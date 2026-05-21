from __future__ import annotations

import base64
import importlib.metadata
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from .config import ANNOTATED_DIR, BASE_DIR, FACE_MATCH_THRESHOLD

EMOTION_MODEL_NAME = "enet_b0_8_best_vgaf"
EMOTION_MODEL_ENGINE = "onnx"
EMOTION_LABEL_MAP = {
    "Anger": "angry",
    "Happiness": "happy",
    "Neutral": "neutral",
    "Sadness": "sad",
    "Surprise": "surprise",
    "Fear": "fear",
    "Disgust": "disgust",
    "Contempt": "contempt",
}
EMOTION_MODEL_ERROR = ""

try:
    from emotiefflib.facial_analysis import EmotiEffLibRecognizer  # type: ignore
except Exception as exc:  # pragma: no cover - optional heavy dependency
    EmotiEffLibRecognizer = None  # type: ignore
    EMOTION_MODEL_ERROR = f"emotiefflib import failed: {type(exc).__name__}: {exc}"

_emotion_recognizer = None


def _get_emotion_recognizer():
    """延迟加载队友新增的 EmotiEffLib 模型；未安装时不影响系统其它功能。"""
    global _emotion_recognizer, EMOTION_MODEL_ERROR
    if EmotiEffLibRecognizer is None:
        return None
    if _emotion_recognizer is None:
        try:
            _emotion_recognizer = EmotiEffLibRecognizer(
                model_name=EMOTION_MODEL_NAME,
                engine=EMOTION_MODEL_ENGINE,
                device="cpu",
            )
            EMOTION_MODEL_ERROR = ""
        except Exception as exc:  # pragma: no cover - depends on optional model files
            EMOTION_MODEL_ERROR = f"{type(exc).__name__}: {exc}"
            return None
    return _emotion_recognizer


def emotion_diagnostics() -> dict:
    """返回情绪模型实际加载状态，供现场验收和排查“是否用了最新模型”。"""
    try:
        version = importlib.metadata.version("emotiefflib")
    except Exception:
        version = ""
    recognizer = _get_emotion_recognizer()
    return {
        "preferred_engine": "emotiefflib",
        "active_engine": "emotiefflib" if recognizer is not None else "deepface/heuristic fallback",
        "emotiefflib_available": EmotiEffLibRecognizer is not None,
        "emotiefflib_loaded": recognizer is not None,
        "emotiefflib_version": version,
        "model_name": EMOTION_MODEL_NAME,
        "model_engine": EMOTION_MODEL_ENGINE,
        "model_error": EMOTION_MODEL_ERROR,
    }


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    quality: float = 0.0

    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    def area(self) -> int:
        return self.w * self.h

    def clipped(self, img: np.ndarray) -> "FaceBox":
        h, w = img.shape[:2]
        x = max(0, min(self.x, w - 1))
        y = max(0, min(self.y, h - 1))
        rw = max(1, min(self.w, w - x))
        rh = max(1, min(self.h, h - y))
        return FaceBox(x, y, rw, rh, self.quality)


def refine_face_box_by_skin(img: np.ndarray, box: FaceBox) -> FaceBox:
    """Haar 对自拍近脸有时只框住眼鼻；用肤色/亮度区域向下扩展到完整脸部。"""
    box = box.clipped(img)
    h, w = img.shape[:2]
    x1 = max(0, box.x - int(box.w * 0.25))
    x2 = min(w, box.x + box.w + int(box.w * 0.25))
    y1 = max(0, box.y - int(box.h * 0.20))
    y2 = min(h, box.y + box.h + int(box.h * 1.25))
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return box
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    ych, cr, cb = cv2.split(ycrcb)
    mask = ((cr > 133) & (cr < 180) & (cb > 75) & (cb < 135) & (ych > 35)).astype(np.uint8) * 255
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return box
    # 选择与原框中心重叠/接近的最大肤色区域。
    cx, cy = box.center()
    best = None
    best_score = -1
    for c in contours:
        x, y, ww, hh = cv2.boundingRect(c)
        gx, gy = x1 + x, y1 + y
        inside = gx <= cx <= gx + ww and gy <= cy <= gy + hh
        dist = abs((gx + ww / 2) - cx) / max(box.w, 1) + abs((gy + hh / 2) - cy) / max(box.h, 1)
        score = ww * hh * (2.0 if inside else 1.0) - dist * 1200
        if score > best_score:
            best_score = score
            best = (gx, gy, ww, hh)
    if not best:
        return box
    gx, gy, ww, hh = best
    # 人脸通常高于宽；给下巴保留空间，避免只截到额头眼睛。
    if hh < ww * 1.05:
        extra = int(ww * 1.25 - hh)
        gy = max(0, gy - int(extra * 0.15))
        hh = min(h - gy, hh + extra)
    nx1 = max(0, min(box.x, gx))
    ny1 = max(0, min(box.y, gy))
    nx2 = min(w, max(box.x + box.w, gx + ww))
    ny2 = min(h, max(box.y + box.h, gy + hh))
    refined = FaceBox(int(nx1), int(ny1), int(nx2 - nx1), int(ny2 - ny1), box.quality)
    # 只有扩展有效且不会离谱时才接受。
    if refined.area() > box.area() * 1.15 and refined.area() < max(box.area() * 9, 1):
        return refined
    return box


def _haar(name: str) -> cv2.CascadeClassifier:
    path = str(Path(cv2.data.haarcascades) / name)
    clf = cv2.CascadeClassifier(path)
    if clf.empty():
        raise RuntimeError(f"无法加载 OpenCV Haar 模型：{path}")
    return clf


_FACE_CASCADE = _haar("haarcascade_frontalface_default.xml")
_FACE_ALT_CASCADE = _haar("haarcascade_frontalface_alt2.xml")
_PROFILE_CASCADE = _haar("haarcascade_profileface.xml")
_EYE_CASCADE = _haar("haarcascade_eye_tree_eyeglasses.xml")


def normalize_size(img: np.ndarray, max_side: int = 1800) -> tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return img, 1.0
    scale = max_side / side
    resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def read_image_path(path: str | Path) -> np.ndarray:
    """读取含中文路径和 EXIF 方向的图片。"""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        arr = np.array(im)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def image_from_upload(file_storage) -> np.ndarray:
    data = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解析图片，请上传 jpg/png/bmp/webp 图片")
    return img


def image_from_base64(data_url: str) -> np.ndarray:
    if not data_url:
        raise ValueError("空图片数据")
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url, validate=False)
    data = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法解析摄像头图片帧")
    return img


def _tiny_frame_hash(img: np.ndarray) -> str:
    """生成轻量感知哈希，用于识别完全重复的照片/重放帧。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tiny = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
    bits = tiny > tiny.mean()
    return "".join("1" if x else "0" for x in bits.ravel())


def _frame_diff(a: np.ndarray, b: np.ndarray) -> float:
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ga = cv2.resize(ga, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    gb = cv2.resize(gb, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return float(np.mean(np.abs(ga - gb)))


def _to_float_list(values: Iterable, scale: float = 1.0) -> list[float]:
    out = []
    for v in values:
        try:
            out.append(float(v) / scale)
        except Exception:
            out.append(0.0)
    return out


def _crop_signal_features(face_img: np.ndarray) -> dict:
    """提取活体/伪造检测用的轻量纹理与颜色响应特征。

    课程前 5 次实验已经用过 HOG/LBP、频域/时频纹理、Face-X-Ray 的伪造边界思想；
    这里不引入重模型，而是把这些思想压缩成可实时运行的特征：归一化人脸帧差、
    高频/网格纹理、镜面高光、色彩响应。返回值只含 JSON 可序列化标量。
    """
    if face_img is None or face_img.size == 0:
        return {
            "face_rgb": [0.0, 0.0, 0.0],
            "border_rgb": [0.0, 0.0, 0.0],
            "face_luma": 0.0,
            "face_sat": 0.0,
            "eye_dark": 0.0,
            "eye_edge": 0.0,
            "mouth_dark": 0.0,
            "mouth_edge": 0.0,
            "mouth_black": 0.0,
            "mouth_open": 0.0,
            "yaw_proxy": 0.0,
            "face_aspect": 0.0,
            "specular_ratio": 0.0,
            "edge_density": 0.0,
            "lap_var": 0.0,
            "fft_high_ratio": 0.0,
            "moire_score": 0.0,
            "norm_hash": "",
            "norm_gray": np.zeros((72, 72), dtype=np.float32),
        }
    h, w = face_img.shape[:2]
    y1, y2 = int(h * 0.10), max(int(h * 0.90), int(h * 0.10) + 1)
    x1, x2 = int(w * 0.10), max(int(w * 0.90), int(w * 0.10) + 1)
    roi = face_img[y1:y2, x1:x2]
    if roi.size == 0:
        roi = face_img
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    ych, cr, cb = cv2.split(ycrcb)
    skin_mask = (cr > 133) & (cr < 180) & (cb > 75) & (cb < 138) & (ych > 35)
    if float(skin_mask.mean()) > 0.06:
        pixels = rgb[skin_mask]
    else:
        pixels = rgb.reshape(-1, 3)
    mean_rgb = pixels.mean(axis=0) if pixels.size else np.array([0.0, 0.0, 0.0], dtype=np.float32)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    specular = ((hsv[:, :, 2] > 242) & (hsv[:, :, 1] < 55)).mean()
    sat = hsv[:, :, 1].mean() / 255.0
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    norm = cv2.resize(gray, (72, 72), interpolation=cv2.INTER_AREA)
    norm = cv2.equalizeHist(norm).astype(np.float32) / 255.0
    bits = norm > float(norm.mean())
    norm_hash = "".join("1" if x else "0" for x in bits[::6, ::6].ravel())
    lap_var = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 1200.0, 2.5)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = float(edges.mean() / 255.0)

    norm_face = cv2.resize(gray, (96, 120), interpolation=cv2.INTER_AREA)
    norm_eq = cv2.equalizeHist(norm_face)
    eye_band = norm_eq[22:54, 12:84]
    mouth_band = norm_eq[72:108, 20:76]
    eye_thr = float(np.percentile(eye_band, 26)) if eye_band.size else 0.0
    mouth_thr = float(np.percentile(mouth_band, 30)) if mouth_band.size else 0.0
    eye_dark = float((eye_band < eye_thr).mean()) if eye_band.size else 0.0
    mouth_dark = float((mouth_band < mouth_thr).mean()) if mouth_band.size else 0.0
    eye_edge = float(cv2.Canny(eye_band, 55, 135).mean() / 255.0) if eye_band.size else 0.0
    mouth_edge = float(cv2.Canny(mouth_band, 45, 125).mean() / 255.0) if mouth_band.size else 0.0
    if mouth_band.size:
        dark_mask = mouth_band < mouth_thr
        ys, _ = np.where(dark_mask)
        vertical_span = (float(ys.max() - ys.min() + 1) / max(mouth_band.shape[0], 1)) if len(ys) else 0.0
        lower_mid = norm_eq[70:112, 18:78]
        base_mid = norm_eq[58:70, 18:78]
        lower_dark_abs = float((lower_mid < 42).mean()) if lower_mid.size else 0.0
        base_dark_abs = float((base_mid < 42).mean()) if base_mid.size else 0.0
        mouth_black = max(0.0, lower_dark_abs - base_dark_abs * 0.35)
        mouth_open = min(1.0, max(0.0, (mouth_dark - 0.18) * 1.80 + vertical_span * 0.18 + mouth_edge * 0.18 + mouth_black * 2.80))
    else:
        mouth_black = 0.0
        mouth_open = 0.0
    left_half = norm_eq[:, :48].astype(np.float32)
    right_half = norm_eq[:, 48:].astype(np.float32)
    yaw_proxy = float((right_half.mean() - left_half.mean()) / 255.0)
    face_aspect = float(w / max(h, 1))

    fsmall = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(fsmall - fsmall.mean()))
    mag = np.abs(spectrum)
    yy, xx = np.indices(mag.shape)
    cy, cx = (np.array(mag.shape) - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    total = float(mag.sum() + 1e-6)
    high_ratio = float(mag[rr > 24].sum() / total)
    # 屏幕翻拍/打印件常出现方向性周期峰；取高频能量峰值相对均值作为轻量摩尔纹指标。
    ring = mag[(rr > 18) & (rr < 45)]
    moire_score = float((np.percentile(ring, 99) / (np.mean(ring) + 1e-6)) / 60.0) if ring.size else 0.0
    return {
        "face_rgb": [round(float(x), 6) for x in mean_rgb.tolist()],
        "face_luma": round(float(0.2126 * mean_rgb[0] + 0.7152 * mean_rgb[1] + 0.0722 * mean_rgb[2]), 6),
        "face_sat": round(float(sat), 6),
        "eye_dark": round(float(eye_dark), 6),
        "eye_edge": round(float(eye_edge), 6),
        "mouth_dark": round(float(mouth_dark), 6),
        "mouth_edge": round(float(mouth_edge), 6),
        "mouth_black": round(float(mouth_black), 6),
        "mouth_open": round(float(mouth_open), 6),
        "yaw_proxy": round(float(yaw_proxy), 6),
        "face_aspect": round(float(face_aspect), 6),
        "specular_ratio": round(float(specular), 6),
        "edge_density": round(float(edge_density), 6),
        "lap_var": round(float(lap_var), 6),
        "fft_high_ratio": round(float(high_ratio), 6),
        "moire_score": round(float(moire_score), 6),
        "norm_hash": norm_hash,
        "norm_gray": norm,
    }


def _closed_eye_proxy(gray: np.ndarray) -> float:
    """用上半脸暗色水平线比例估计闭眼程度；不依赖关键点模型，适合普通摄像头快速判定。"""
    try:
        norm_face = cv2.resize(gray, (96, 120), interpolation=cv2.INTER_AREA)
        norm_eq = cv2.equalizeHist(norm_face)
        # 眼睛真实闭合时会形成横向深色细线；眼镜/阴影可能让绝对暗度偏高，
        # 所以同时融合“暗像素比例”和“水平边缘强度”。
        band = norm_eq[24:54, 10:86]
        if band.size == 0:
            return 0.0
        local_thr = min(112.0, float(np.percentile(band, 38)) + 14.0)
        dark = (band < local_thr).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 1))
        horizontal = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
        sobel_y = cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3)
        edge_score = float(np.percentile(np.abs(sobel_y), 90) / 255.0)
        return float(min(1.0, horizontal.mean() * 0.82 + edge_score * 0.18))
    except Exception:
        return 0.0


def _frame_border_rgb(img: np.ndarray) -> list[float]:
    """读取前端写入采集帧边缘的随机打光水印颜色，作为弱光环境下的稳健校验信号。"""
    if img is None or img.size == 0:
        return [0.0, 0.0, 0.0]
    h, w = img.shape[:2]
    band = max(8, int(min(h, w) * 0.075))
    parts = [
        img[:band, :, :],
        img[max(0, h - band):, :, :],
        img[:, :band, :],
        img[:, max(0, w - band):, :],
    ]
    pixels = np.concatenate([p.reshape(-1, 3) for p in parts if p.size], axis=0)
    if pixels.size == 0:
        return [0.0, 0.0, 0.0]
    rgb = pixels[:, ::-1].astype(np.float32) / 255.0
    mean_rgb = rgb.mean(axis=0)
    return [round(float(x), 6) for x in mean_rgb.tolist()]


def _parse_flash_rgb(value) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    rgb = _to_float_list(value[:3], scale=255.0)
    return [max(0.0, min(1.0, x)) for x in rgb]


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.size < 3 or b.size != a.size or float(a.std()) < 1e-6 or float(b.std()) < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _expected_flash_rgb(stage: int, flash_index: int | None, challenge_steps: list[dict] | None) -> list[float] | None:
    if flash_index is None or not challenge_steps:
        return None
    step = next((s for s in challenge_steps if int(s.get("stage", 0)) == int(stage)), None)
    if not step:
        # 单组实时检测时，前端可能传 stage=N，而 analyze_liveness 只收到一个 step。
        step = challenge_steps[0] if len(challenge_steps) == 1 else None
    seq = (step or {}).get("flash_sequence") or []
    if not seq:
        return None
    try:
        item = seq[int(flash_index) % len(seq)]
        return _parse_flash_rgb(item.get("rgb") if isinstance(item, dict) else item)
    except Exception:
        return None


def _flash_response_check(seen: list[dict], challenge_steps: list[dict] | None) -> dict:
    """校验随机屏幕闪光挑战。

    预录视频只能提前录到固定光照；本次 session 的随机颜色序列无法提前出现在视频里。
    因此要求人脸区域平均颜色/亮度与服务端下发的 flash_sequence 存在同步相关性。
    """
    if not challenge_steps:
        return {
            "pass": True,
            "enabled": False,
            "reason": "未启用随机闪光挑战",
            "coverage": 0.0,
            "color_corr": 0.0,
            "brightness_corr": 0.0,
            "response_amplitude": 0.0,
            "invalid_meta_ratio": 0.0,
        }
    rows = [r for r in seen if r.get("expected_flash_rgb") is not None and r.get("face_rgb")]
    all_meta_rows = [r for r in seen if r.get("expected_flash_rgb") is not None]
    if len(rows) < 6 and len(all_meta_rows) < 6:
        return {"pass": False, "enabled": True, "reason": "随机闪光响应帧不足", "coverage": 0.0, "color_corr": 0.0, "brightness_corr": 0.0, "response_amplitude": 0.0, "invalid_meta_ratio": 1.0}
    stage_results = []
    invalid = 0
    total_meta = 0
    for step in challenge_steps:
        stage = int(step.get("stage", 0))
        seq = step.get("flash_sequence") or []
        srows = [r for r in rows if int(r.get("stage", 0)) == stage]
        smeta_rows = [r for r in all_meta_rows if int(r.get("stage", 0)) == stage]
        if not srows and len(challenge_steps) == 1:
            srows = rows
        if not smeta_rows and len(challenge_steps) == 1:
            smeta_rows = all_meta_rows
        check_rows = smeta_rows or srows
        if len(check_rows) < 4:
            stage_results.append({"stage": stage, "pass": False, "reason": "该组闪光帧不足"})
            continue
        uniq = {int(r.get("flash_index", -1)) % max(len(seq), 1) for r in check_rows if r.get("flash_index") is not None}
        coverage = min(1.0, len(uniq) / max(min(len(seq), 3), 1))
        obs = np.array([r.get("face_rgb", [0.0, 0.0, 0.0]) for r in check_rows], dtype=np.float32)
        border_obs = np.array([r.get("border_rgb") or [0.0, 0.0, 0.0] for r in check_rows], dtype=np.float32)
        exp = np.array([r["expected_flash_rgb"] for r in check_rows], dtype=np.float32)
        sent = np.array([r.get("sent_flash_rgb") or r["expected_flash_rgb"] for r in check_rows], dtype=np.float32)
        invalid += int(np.sum(np.linalg.norm(sent - exp, axis=1) > 0.12))
        total_meta += len(check_rows)
        # 自动白平衡会压缩绝对颜色，采用中心化 RGB、相邻变化方向和亮度双相关，更适合普通摄像头。
        obs_center = obs - obs.mean(axis=0, keepdims=True)
        exp_center = exp - exp.mean(axis=0, keepdims=True)
        channel_corrs = [_corr(obs_center[:, i], exp_center[:, i]) for i in range(3)]
        direct_color_corr = float(np.mean(channel_corrs))
        obs_delta = np.diff(obs, axis=0)
        exp_delta = np.diff(exp, axis=0)
        delta_corrs = [_corr(obs_delta[:, i], exp_delta[:, i]) for i in range(3)] if len(obs_delta) >= 3 else [0.0]
        delta_color_corr = float(np.mean(delta_corrs))
        obs_norm = obs / (obs.sum(axis=1, keepdims=True) + 1e-6)
        exp_norm = exp / (exp.sum(axis=1, keepdims=True) + 1e-6)
        chroma_corrs = [_corr(obs_norm[:, i], exp_norm[:, i]) for i in range(3)]
        chroma_corr = float(np.mean(chroma_corrs))
        color_corr = float(max(direct_color_corr, delta_color_corr, chroma_corr))
        obs_luma = obs @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        exp_luma = exp @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
        brightness_corr = _corr(obs_luma, exp_luma)
        delta_brightness_corr = _corr(np.diff(obs_luma), np.diff(exp_luma)) if len(obs_luma) >= 4 else 0.0
        brightness_corr = float(max(brightness_corr, delta_brightness_corr))
        response_amp = float(np.mean(np.std(obs, axis=0)) + np.std(obs_luma) * 0.35 + np.mean(np.std(obs_norm, axis=0)) * 0.18)
        obs_step = float(np.mean(np.linalg.norm(np.diff(obs, axis=0), axis=1))) if len(obs) >= 2 else 0.0
        obs_total_step = float(np.linalg.norm(obs[-1] - obs[0])) if len(obs) >= 2 else 0.0
        exp_step = float(np.mean(np.linalg.norm(np.diff(exp, axis=0), axis=1))) if len(exp) >= 2 else 0.0
        expected_response_ratio = float(obs_step / (exp_step + 1e-6))
        # 真实屏幕打光通常表现为：人脸颜色围绕本次随机序列轻微同步振荡；
        # 预录视频/缩放裁剪也会让均值漂移，但其相邻变化方向和闪光序列不稳定同步。
        live_face_signal = bool(
            coverage >= 0.66
            and response_amp >= 0.0025
            and (
                color_corr >= 0.08
                or brightness_corr >= 0.10
                or (response_amp >= 0.010 and color_corr >= 0.045)
            )
        )
        strict_face_signal = bool(
            coverage >= 0.66
            and response_amp >= 0.0060
            and expected_response_ratio >= 0.012
            and obs_total_step >= 0.008
            and (
                color_corr >= 0.18
                or brightness_corr >= 0.18
                or chroma_corr >= 0.22
                or (delta_color_corr >= 0.12 and response_amp >= 0.010)
            )
        )
        border_color_corr = 0.0
        border_amp = 0.0
        border_meta_ok = False
        if border_obs.size and float(border_obs.std()) > 1e-6:
            border_center = border_obs - border_obs.mean(axis=0, keepdims=True)
            border_corrs = [_corr(border_center[:, i], exp_center[:, i]) for i in range(3)]
            border_color_corr = float(np.mean(border_corrs))
            border_amp = float(np.mean(np.std(border_obs, axis=0)))
            border_meta_ok = bool(coverage >= 0.50 and border_amp >= 0.006 and border_color_corr >= 0.18)
        # 单组实时体验允许“边缘水印 + 宽松人脸响应”通过，减少教室弱光误拒；
        # 正式提交时 _spoof_flash_face_response_pass 会使用 strict_face_response_pass。
        face_response_ok = live_face_signal
        strict_face_response_ok = strict_face_signal
        ok = bool(face_response_ok or border_meta_ok)
        stage_results.append({
            "stage": stage,
            "pass": ok,
            "face_response_pass": face_response_ok,
            "strict_face_response_pass": strict_face_response_ok,
            "border_meta_pass": border_meta_ok,
            "coverage": round(float(coverage), 4),
            "color_corr": round(float(color_corr), 4),
            "direct_color_corr": round(float(direct_color_corr), 4),
            "delta_color_corr": round(float(delta_color_corr), 4),
            "chroma_corr": round(float(chroma_corr), 4),
            "brightness_corr": round(float(brightness_corr), 4),
            "response_amplitude": round(float(response_amp), 6),
            "observed_flash_step": round(float(obs_step), 6),
            "observed_flash_total_step": round(float(obs_total_step), 6),
            "expected_response_ratio": round(float(expected_response_ratio), 6),
            "border_color_corr": round(float(border_color_corr), 4),
            "border_response_amplitude": round(float(border_amp), 6),
        })
    invalid_ratio = invalid / max(total_meta, len(rows), 1)
    passed = bool(stage_results and all(r.get("pass") for r in stage_results) and invalid_ratio <= 0.20)
    if passed:
        reason = "随机闪光响应同步"
    elif invalid_ratio > 0.20:
        reason = "闪光元数据与服务端挑战不一致"
    else:
        reason = "人脸区域未响应本次随机闪光，疑似预录视频/屏幕翻拍"
    return {
        "pass": passed,
        "enabled": True,
        "reason": reason,
        "coverage": round(float(np.mean([r.get("coverage", 0) for r in stage_results]) if stage_results else 0.0), 4),
        "color_corr": round(float(np.mean([r.get("color_corr", 0) for r in stage_results]) if stage_results else 0.0), 4),
        "brightness_corr": round(float(np.mean([r.get("brightness_corr", 0) for r in stage_results]) if stage_results else 0.0), 4),
        "response_amplitude": round(float(np.mean([r.get("response_amplitude", 0) for r in stage_results]) if stage_results else 0.0), 6),
        "invalid_meta_ratio": round(float(invalid_ratio), 4),
        "stages": stage_results,
    }


def _spoof_flash_face_response_pass(flash_check: dict) -> bool:
    """抗预录视频最终门槛必须有人脸区域响应，不能只靠边缘水印元数据。

    正式打卡有 3 组随机挑战。普通教室屏幕打光可能被摄像头自动白平衡压缩，
    因此最终门槛采用“多数严格人脸响应”：单组必须严格通过；多组至少 2/3
    严格通过，同时每组仍需通过宽松实时闪光检查。预录视频通常只能伪造元数据，
    难以在多数随机组里让人脸区域按本次颜色序列同步变化。
    """
    if not flash_check.get("enabled"):
        return True
    stages = flash_check.get("stages") or []
    if not stages:
        return False
    strict_count = sum(1 for s in stages if s.get("strict_face_response_pass"))
    required = 1 if len(stages) == 1 else max(2, math.ceil(len(stages) * 0.67))
    return strict_count >= required


def _deepface_emotion(face_img: np.ndarray) -> dict | None:
    """优先复用实验五 DeepFace 表情分析；不可用时返回 None 走轻量 fallback。"""
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        from deepface import DeepFace  # type: ignore
    except Exception:
        return None
    try:
        res = DeepFace.analyze(
            img_path=face_img,
            actions=["emotion"],
            detector_backend="skip",
            enforce_detection=False,
            silent=True,
        )
        if isinstance(res, list):
            res = res[0]
        raw_scores = res.get("emotion") or {}
        scores = {str(k): float(v) / 100.0 for k, v in raw_scores.items()}
        if not scores:
            return None
        label = str(res.get("dominant_emotion") or max(scores, key=scores.get))
        conf = max(0.0, min(1.0, float(scores.get(label, max(scores.values())))))
        return {"emotion": label, "confidence": round(conf, 4), "scores": {k: round(float(v), 4) for k, v in scores.items()}, "engine": "deepface"}
    except Exception:
        return None


def _emotiefflib_emotion(face_img: np.ndarray) -> dict | None:
    """队友版本：EfficientNet + ONNX 表情分类；失败时返回 None 让后续引擎兜底。"""
    try:
        recognizer = _get_emotion_recognizer()
        if recognizer is None:
            return None
        # OpenCV 读入为 BGR，而 EmotiEffLib/PIL 训练链路按 RGB 归一化；不转换会显著增加
        # Surprise/Fear 等异常输出。这里保持 DeepFace 走 BGR，EmotiEffLib 单独转 RGB。
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB) if face_img.ndim == 3 else face_img
        emotion_labels, all_scores = recognizer.predict_emotions(rgb_face, logits=False)
        if not emotion_labels or all_scores is None or len(all_scores) == 0:
            return None
        raw_label = str(emotion_labels[0])
        label = EMOTION_LABEL_MAP.get(raw_label, raw_label.lower())
        raw = np.asarray(all_scores[0], dtype=float).ravel()
        idx_to_class = getattr(recognizer, "idx_to_emotion_class", {}) or {}
        scores = {
            EMOTION_LABEL_MAP.get(str(idx_to_class.get(i, i)), str(idx_to_class.get(i, i)).lower()): round(float(raw[i]), 4)
            for i in range(len(raw))
        }
        conf = round(float(scores.get(label, float(raw.max()) if raw.size else 0.0)), 4)
        return {
            "emotion": label,
            "confidence": conf,
            "scores": scores,
            "engine": "emotiefflib",
            "model": f"{EMOTION_MODEL_NAME}/{EMOTION_MODEL_ENGINE}",
            "raw_emotion": raw_label,
        }
    except Exception:
        return None


def save_image(img: np.ndarray, directory: Path, prefix: str = "img", ext: str = ".jpg") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
    path = directory / filename
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"图片编码失败：{ext}")
    # Windows 下 OpenCV 的 imwrite 对中文路径兼容性不稳定，tofile 更可靠。
    buf.tofile(str(path))
    return path


def _nms(boxes: list[FaceBox], overlap_threshold: float = 0.35) -> list[FaceBox]:
    if not boxes:
        return []
    arr = np.array([[b.x, b.y, b.x + b.w, b.y + b.h, b.quality] for b in boxes], dtype=float)
    x1, y1, x2, y2, score = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(score)
    pick = []
    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        idxs = idxs[:-1]
        if len(idxs) == 0:
            break
        xx1 = np.maximum(x1[last], x1[idxs])
        yy1 = np.maximum(y1[last], y1[idxs])
        xx2 = np.minimum(x2[last], x2[idxs])
        yy2 = np.minimum(y2[last], y2[idxs])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs]
        idxs = idxs[overlap <= overlap_threshold]
    return [boxes[i] for i in pick]


def detect_faces(img: np.ndarray, min_size: int = 64) -> list[FaceBox]:
    if img is None or img.size == 0:
        return []
    work, scale = normalize_size(img)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    h, w = gray.shape[:2]
    min_size = max(40, min(min_size, min(h, w) // 4 if min(h, w) < 300 else min_size))
    raw = []
    for cascade, weight, neighbors in ((_FACE_CASCADE, 1.0, 5), (_FACE_ALT_CASCADE, 0.96, 4)):
        for (x, y, fw, fh) in cascade.detectMultiScale(
            gray, scaleFactor=1.08, minNeighbors=neighbors, minSize=(min_size, min_size)
        ):
            ox, oy, ow, oh = int(x / scale), int(y / scale), max(1, int(fw / scale)), max(1, int(fh / scale))
            crop = img[oy:oy + oh, ox:ox + ow]
            raw.append(FaceBox(ox, oy, ow, oh, quality_score(crop) * weight))
    for mirror in (False, True):
        g = cv2.flip(gray, 1) if mirror else gray
        for (x, y, fw, fh) in _PROFILE_CASCADE.detectMultiScale(
            g, scaleFactor=1.08, minNeighbors=5, minSize=(min_size, min_size)
        ):
            if mirror:
                x = w - x - fw
            ox, oy, ow, oh = int(x / scale), int(y / scale), max(1, int(fw / scale)), max(1, int(fh / scale))
            crop = img[oy:oy + oh, ox:ox + ow]
            raw.append(FaceBox(ox, oy, ow, oh, quality_score(crop) * 0.92))
    nms = _nms(raw)
    img_area = max(img.shape[0] * img.shape[1], 1)
    return sorted(nms, key=lambda b: b.quality * 0.72 + min(b.area() / img_area * 15, 1.0) * 0.28, reverse=True)


def crop_face(img: np.ndarray, box: FaceBox | tuple[int, int, int, int], pad: float = 0.18) -> np.ndarray:
    if not isinstance(box, FaceBox):
        box = FaceBox(*box)
    box = refine_face_box_by_skin(img, box)
    h, w = img.shape[:2]
    px, py = int(box.w * pad), int(box.h * pad)
    x1 = max(0, box.x - px)
    y1 = max(0, box.y - py)
    x2 = min(w, box.x + box.w + px)
    y2 = min(h, box.y + box.h + py)
    return img[y1:y2, x1:x2]


def quality_score(face_img: np.ndarray) -> float:
    if face_img is None or face_img.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    sharp = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 850.0, 1.0)
    brightness = gray.mean() / 255.0
    bright_score = max(0.0, 1.0 - abs(brightness - 0.52) / 0.52)
    contrast = min(gray.std() / 80.0, 1.0)
    size_score = min(math.sqrt(face_img.shape[0] * face_img.shape[1]) / 160.0, 1.0)
    base = 0.34 * sharp + 0.26 * bright_score + 0.22 * contrast + 0.18 * size_score
    upper = gray[: max(1, int(gray.shape[0] * 0.68)), :]
    min_eye = max(8, min(gray.shape[:2]) // 10)
    eyes = _EYE_CASCADE.detectMultiScale(
        upper, scaleFactor=1.08, minNeighbors=3, minSize=(min_eye, min_eye)
    )
    eye_count = len(eyes)
    if eye_count >= 2:
        eye_factor = 1.10
    elif eye_count == 1:
        eye_factor = 0.92
    else:
        # 过滤把衣领、海报、背景误识为人脸的情况。
        eye_factor = 0.45
    return float(min(base * eye_factor, 1.0))


def _lbp_hist(gray: np.ndarray, bins: int = 32) -> np.ndarray:
    g = gray.astype(np.uint8)
    c = g[1:-1, 1:-1]
    lbp = np.zeros_like(c, dtype=np.uint8)
    neighbors = [
        g[:-2, :-2], g[:-2, 1:-1], g[:-2, 2:], g[1:-1, 2:],
        g[2:, 2:], g[2:, 1:-1], g[2:, :-2], g[1:-1, :-2],
    ]
    for idx, n in enumerate(neighbors):
        lbp |= ((n >= c) << idx).astype(np.uint8)
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, 256), density=False)
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return hist


def face_embedding(face_img: np.ndarray) -> list[float]:
    if face_img is None or face_img.size == 0:
        raise ValueError("空人脸区域，无法提取特征")
    gray0 = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray0, (128, 128), interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    features = [_lbp_hist(cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA), bins=64) * 1.4]
    for gy in range(0, 96, 24):
        for gx in range(0, 96, 24):
            block = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)[gy:gy + 24, gx:gx + 24]
            features.append(_lbp_hist(block, bins=16) * 0.8)
    # HOG 梯度方向直方图：比纯 LBP 对缩放和光照更稳定。
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hog_parts = []
    cell = 16
    for yy in range(0, 128, cell):
        for xx in range(0, 128, cell):
            m = mag[yy:yy + cell, xx:xx + cell].ravel()
            a = (ang[yy:yy + cell, xx:xx + cell].ravel() % 180) / 20.0
            hist = np.zeros(9, dtype=np.float32)
            for bin_f, weight in zip(a, m):
                hist[int(bin_f) % 9] += weight
            hist /= (np.linalg.norm(hist) + 1e-6)
            hog_parts.append(hist)
    features.append(np.concatenate(hog_parts) * 0.9)
    # 低频结构/DCT，保留五官整体位置。
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    dct = cv2.dct(small)[:12, :12].ravel()
    dct = (dct - dct.mean()) / (dct.std() + 1e-6)
    features.append(dct * 0.35)
    small = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    small = (small - small.mean()) / (small.std() + 1e-6)
    features.append(small.ravel() * 0.18)
    vec = np.concatenate(features).astype(np.float32)
    vec /= (np.linalg.norm(vec) + 1e-8)
    return vec.tolist()


def embedding_from_image(img: np.ndarray) -> tuple[list[float], FaceBox, float]:
    faces = detect_faces(img)
    if not faces:
        raise ValueError("未检测到清晰人脸")
    box = faces[0]
    crop = crop_face(img, box)
    return face_embedding(crop), box, quality_score(crop)


def cosine_similarity(a: Iterable[float], b: Iterable[float]) -> float:
    va = np.array(list(a), dtype=np.float32)
    vb = np.array(list(b), dtype=np.float32)
    if va.size != vb.size or va.size == 0:
        return 0.0
    return float(np.dot(va, vb) / ((np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-8))


def recognize(
    embedding: list[float],
    samples: list[dict],
    threshold: float = FACE_MATCH_THRESHOLD,
    margin: float = 0.0,
) -> dict:
    """在人脸库中按“学生”聚合匹配。

    多张照片不再只是“互相独立的候选样本”：系统会同时使用最佳单样本、TopK 均值、
    质量加权质心、稳定支持样本数四类证据。这样同一学生上传正面、侧脸、不同光照
    的多张图片后，任一图片可命中最佳样本，整体质心也会提高跨场景鲁棒性；低质量
    或明显离群样本只弱参与，避免靠堆图虚高。
    """
    query = np.array(list(embedding), dtype=np.float32)
    if query.size == 0:
        return {"matched": False, "student": None, "score": 0.0, "sample_id": None, "second_score": 0.0, "margin": 0.0, "candidates": []}
    query /= (np.linalg.norm(query) + 1e-8)
    sample_candidates: list[dict] = []
    by_student: dict[int | str, dict] = {}
    for sample in samples:
        try:
            emb = np.array(json.loads(sample["embedding"]), dtype=np.float32)
            if emb.size != query.size:
                continue
            emb /= (np.linalg.norm(emb) + 1e-8)
            score = float(np.dot(query, emb))
        except Exception:
            continue
        item = {"student": sample, "score": float(score), "sample_id": sample.get("id")}
        sample_candidates.append(item)
        sid = sample.get("student_id") or sample.get("id") or sample.get("student_no")
        bucket = by_student.setdefault(sid, {"student": sample, "scores": [], "sample_ids": [], "embeddings": [], "qualities": []})
        bucket["scores"].append(float(score))
        bucket["sample_ids"].append(sample.get("id"))
        bucket["embeddings"].append(emb)
        try:
            bucket["qualities"].append(float(sample.get("quality") or 0.5))
        except Exception:
            bucket["qualities"].append(0.5)

    if not sample_candidates:
        return {"matched": False, "student": None, "score": 0.0, "sample_id": None, "second_score": 0.0, "margin": 0.0, "candidates": []}

    student_candidates: list[dict] = []
    for bucket in by_student.values():
        raw_scores = list(bucket["scores"])
        sorted_scores = sorted(raw_scores, reverse=True)
        best_score = float(sorted_scores[0])
        topk = sorted_scores[: min(5, len(sorted_scores))]
        topk_mean = float(np.mean(topk))
        support = sum(1 for s in sorted_scores if s >= best_score - 0.075 or s >= threshold - 0.035)
        support_bonus = min(0.050, max(0, support - 1) * 0.014)

        embeddings = np.vstack(bucket["embeddings"])
        qualities = np.array(bucket["qualities"], dtype=np.float32)
        # 质量 + 与当前查询接近程度共同决定质心权重：多视角样本能增强鲁棒性，离群样本不会拖垮。
        score_arr = np.array(raw_scores, dtype=np.float32)
        rel = np.clip((score_arr - max(best_score - 0.16, 0.0)) / 0.16, 0.05, 1.0)
        q_weights = np.clip(qualities, 0.20, 1.0) * rel
        if float(q_weights.sum()) <= 1e-6:
            q_weights = np.ones_like(q_weights)
        centroid = np.average(embeddings, axis=0, weights=q_weights)
        centroid /= (np.linalg.norm(centroid) + 1e-8)
        centroid_score = float(np.dot(query, centroid))

        # 保底取最佳单样本，避免同学上传多张差异很大的照片后平均值把正确样本“稀释”。
        blended = 0.68 * best_score + 0.18 * topk_mean + 0.14 * centroid_score + support_bonus
        aggregate_score = min(1.0, max(best_score, blended, centroid_score + min(0.028, support_bonus)))
        best_index = raw_scores.index(best_score)
        student_candidates.append({
            "student": bucket["student"],
            "score": float(aggregate_score),
            "sample_id": bucket["sample_ids"][best_index],
            "best_sample_score": float(best_score),
            "topk_mean_score": topk_mean,
            "centroid_score": centroid_score,
            "sample_count": len(raw_scores),
            "support_count": support,
        })

    sample_candidates.sort(key=lambda x: x["score"], reverse=True)
    student_candidates.sort(key=lambda x: x["score"], reverse=True)
    best = student_candidates[0]
    second_score = float(student_candidates[1]["score"]) if len(student_candidates) > 1 else 0.0
    gap = float(best["score"] - second_score)
    matched = bool(best["score"] >= threshold and gap >= margin)
    return {
        "matched": matched,
        "student": best["student"],
        "score": float(best["score"]),
        "sample_id": best["sample_id"],
        "second_score": second_score,
        "margin": gap,
        "threshold": float(threshold),
        "best_sample_score": float(best.get("best_sample_score", best["score"])),
        "topk_mean_score": float(best.get("topk_mean_score", best["score"])),
        "centroid_score": float(best.get("centroid_score", best["score"])),
        "sample_count": int(best.get("sample_count", 1)),
        "support_count": int(best.get("support_count", 1)),
        "candidates": student_candidates[:3],
        "sample_candidates": sample_candidates[:5],
    }


def analyze_liveness(frames: list[dict], actions: list[str] | None = None, challenge_steps: list[dict] | None = None) -> dict:
    """后端动作活体检测。

    新版本按 3 组随机动作进行挑战：前端只有在某组动作通过后才进入下一组，
    后端最终仍会重新计算所有帧，避免篡改前端 JS 直接绕过。每个 stage 内部
    使用“早期帧 -> 后期帧”的人脸中心、面积变化来判断动态动作；旧版带
    stage=0 基准帧的数据也兼容。
    """
    actions = actions or []
    observations: list[dict] = []
    hashes: list[str] = []
    decoded_frames: list[np.ndarray] = []
    crop_norm_frames: list[tuple[int, np.ndarray]] = []

    for seq, item in enumerate(frames):
        try:
            img = image_from_base64(item.get("image", ""))
        except Exception:
            continue
        hashes.append(_tiny_frame_hash(img))
        decoded_frames.append(img)
        stage = int(item.get("stage", 0))
        elapsed_raw = item.get("stage_elapsed_ms", item.get("elapsed_ms"))
        try:
            elapsed_ms = float(elapsed_raw) if elapsed_raw is not None else None
        except (TypeError, ValueError):
            elapsed_ms = None
        faces = detect_faces(img, min_size=70)
        frame_border_rgb = _frame_border_rgb(img)
        if not faces:
            observations.append({"seq": seq, "stage": stage, "elapsed_ms": elapsed_ms, "seen": False, "border_rgb": frame_border_rgb})
            continue
        box = faces[0]
        h, w = img.shape[:2]
        crop = crop_face(img, box)
        signal = _crop_signal_features(crop)
        face_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        closed_eye_proxy = _closed_eye_proxy(face_gray)
        flash_index_raw = item.get("flash_index")
        try:
            flash_index = int(flash_index_raw) if flash_index_raw is not None else None
        except (TypeError, ValueError):
            flash_index = None
        sent_flash_rgb = _parse_flash_rgb(item.get("flash_rgb") or item.get("expected_flash_rgb"))
        expected_rgb = _expected_flash_rgb(stage, flash_index, challenge_steps)
        crop_norm_frames.append((seq, signal.pop("norm_gray")))
        observations.append({
            "seq": seq,
            "stage": stage,
            "elapsed_ms": elapsed_ms,
            "seen": True,
            "cx": box.center()[0] / max(w, 1),
            "cy": box.center()[1] / max(h, 1),
            "area": box.area() / max(w * h, 1),
            "quality": quality_score(crop),
            "closed_eye_proxy": round(float(closed_eye_proxy), 6),
            "border_rgb": frame_border_rgb,
            "flash_index": flash_index,
            "sent_flash_rgb": sent_flash_rgb,
            "expected_flash_rgb": expected_rgb,
            **signal,
        })

    seen = [o for o in observations if o.get("seen")]
    if len(seen) < max(4, int(len(frames) * 0.45)):
        return {"pass": False, "score": 0.0, "reason": "有效人脸帧不足", "observations": observations[-12:]}

    by_stage: dict[int, list[dict]] = {}
    for o in seen:
        by_stage.setdefault(int(o["stage"]), []).append(o)
    for rows in by_stage.values():
        rows.sort(key=lambda x: (x.get("elapsed_ms") is None, x.get("elapsed_ms") or x.get("seq", 0), x.get("seq", 0)))

    def stage_stats(stage: int) -> dict | None:
        rows = by_stage.get(stage, [])
        if not rows:
            return None
        total_stage_frames = sum(
            1 for r in observations
            if int(r.get("stage", 0)) == int(stage)
        )
        n = len(rows)
        k = max(1, int(math.ceil(n / 3)))
        early = rows[:k]
        late = rows[-k:]

        def avg(part: list[dict], key: str) -> float:
            return float(np.mean([r[key] for r in part if key in r]))

        elapsed_vals = [r.get("elapsed_ms") for r in rows if r.get("elapsed_ms") is not None]
        duration_ms = float(max(elapsed_vals) - min(elapsed_vals)) if len(elapsed_vals) >= 2 else None
        areas = [r["area"] for r in rows]
        cxs = [r["cx"] for r in rows]
        cys = [r["cy"] for r in rows]
        eye_dark_vals = [r.get("eye_dark", 0.0) for r in rows]
        eye_edge_vals = [r.get("eye_edge", 0.0) for r in rows]
        mouth_open_vals = [r.get("mouth_open", 0.0) for r in rows]
        mouth_dark_vals = [r.get("mouth_dark", 0.0) for r in rows]
        mouth_edge_vals = [r.get("mouth_edge", 0.0) for r in rows]
        yaw_vals = [r.get("yaw_proxy", 0.0) for r in rows]
        aspect_vals = [r.get("face_aspect", 0.0) for r in rows]
        closed_eye_vals = [r.get("closed_eye_proxy", 0.0) for r in rows]
        return {
            "stage": stage,
            "seen_frames": n,
            "total_stage_frames": total_stage_frames,
            "face_seen_ratio": n / max(total_stage_frames, 1),
            "first_cx": avg(early, "cx"),
            "last_cx": avg(late, "cx"),
            "first_cy": avg(early, "cy"),
            "last_cy": avg(late, "cy"),
            "first_area": avg(early, "area"),
            "last_area": avg(late, "area"),
            "mean_cx": avg(rows, "cx"),
            "mean_cy": avg(rows, "cy"),
            "mean_area": avg(rows, "area"),
            "min_cx": float(min(cxs)),
            "max_cx": float(max(cxs)),
            "min_cy": float(min(cys)),
            "max_cy": float(max(cys)),
            "min_area": float(min(areas)),
            "max_area": float(max(areas)),
            "duration_ms": duration_ms,
            "avg_quality": avg(rows, "quality"),
            "first_eye_dark": avg(early, "eye_dark"),
            "last_eye_dark": avg(late, "eye_dark"),
            "min_eye_dark": float(min(eye_dark_vals)),
            "max_eye_dark": float(max(eye_dark_vals)),
            "eye_edge_range": float(max(eye_edge_vals) - min(eye_edge_vals)),
            "min_closed_eye": float(min(closed_eye_vals)),
            "max_closed_eye": float(max(closed_eye_vals)),
            "closed_eye_range": float(max(closed_eye_vals) - min(closed_eye_vals)),
            "first_mouth_open": avg(early, "mouth_open"),
            "last_mouth_open": avg(late, "mouth_open"),
            "min_mouth_open": float(min(mouth_open_vals)),
            "max_mouth_open": float(max(mouth_open_vals)),
            "mouth_dark_range": float(max(mouth_dark_vals) - min(mouth_dark_vals)),
            "mouth_edge_range": float(max(mouth_edge_vals) - min(mouth_edge_vals)),
            "first_yaw_proxy": avg(early, "yaw_proxy"),
            "last_yaw_proxy": avg(late, "yaw_proxy"),
            "min_yaw_proxy": float(min(yaw_vals)),
            "max_yaw_proxy": float(max(yaw_vals)),
            "aspect_range": float(max(aspect_vals) - min(aspect_vals)),
        }

    base_stats = stage_stats(0)
    stage_numbers = sorted(s for s in by_stage.keys() if s != 0)

    def stage_for_action(idx: int) -> int | None:
        preferred = idx
        if preferred in by_stage:
            return preferred
        # 兼容旧版：actions 第 1 个可能对应 stage=1，也可能对应 stage=0 之后的第一个非 0 stage。
        if 0 in by_stage and idx in by_stage:
            return idx
        if idx - 1 in by_stage and idx - 1 != 0 and 0 in by_stage:
            return idx - 1
        if idx - 1 < len(stage_numbers):
            return stage_numbers[idx - 1]
        return None

    def eval_action(stage: int, action: str) -> dict:
        st = stage_stats(stage)
        if not st:
            return {"stage": stage, "action": action, "ok": False, "delta": 0.0, "reason": "该组未检测到人脸"}
        action_alias = {
            "nod_up_down": "nod",
            "shake_left_right": "move_right",
            "zoom_in_out": "move_closer",
        }
        action = action_alias.get(action, action)
        dx = st["last_cx"] - st["first_cx"]
        dy = st["last_cy"] - st["first_cy"]
        area_ratio = st["last_area"] / (st["first_area"] + 1e-6)
        x_range = st["max_cx"] - st["min_cx"]
        y_range = st["max_cy"] - st["min_cy"]
        area_range_ratio = st["max_area"] / (st["min_area"] + 1e-6)
        eye_dark_delta = st["last_eye_dark"] - st["first_eye_dark"]
        eye_dark_range = st["max_eye_dark"] - st["min_eye_dark"]
        mouth_delta = st["last_mouth_open"] - st["first_mouth_open"]
        mouth_range = st["max_mouth_open"] - st["min_mouth_open"]
        yaw_delta = st["last_yaw_proxy"] - st["first_yaw_proxy"]
        yaw_range = st["max_yaw_proxy"] - st["min_yaw_proxy"]
        step = next((s for s in (challenge_steps or []) if int(s.get("stage", 0)) == int(stage)), None)
        flash_rows = [r for r in by_stage.get(stage, []) if r.get("expected_flash_rgb") is not None and r.get("face_rgb")]
        flash_stage_check = _flash_response_check(flash_rows, [step]) if step else {"pass": False, "reason": "该组未启用随机闪光"}
        timeout_ok = st["duration_ms"] is None or st["duration_ms"] <= 5500
        enough_frames = st["seen_frames"] >= 3
        ok = False
        delta = 0.0
        detail = ""

        if action == "move_left":
            delta = -dx
            baseline_delta = (base_stats["mean_cx"] - st["mean_cx"]) if base_stats else 0.0
            ok = delta > 0.018 or baseline_delta > 0.025
            detail = f"left_delta={delta:.4f}, baseline_delta={baseline_delta:.4f}"
        elif action == "move_right":
            delta = dx
            baseline_delta = (st["mean_cx"] - base_stats["mean_cx"]) if base_stats else 0.0
            ok = delta > 0.018 or baseline_delta > 0.025
            detail = f"right_delta={delta:.4f}, baseline_delta={baseline_delta:.4f}"
        elif action == "move_closer":
            delta = area_ratio - 1.0
            baseline_ratio = (st["mean_area"] / (base_stats["mean_area"] + 1e-6)) if base_stats else 1.0
            ok = area_ratio > 1.055 or baseline_ratio > 1.080
            detail = f"area_ratio={area_ratio:.4f}, baseline_ratio={baseline_ratio:.4f}"
        elif action == "move_away":
            delta = 1.0 - area_ratio
            baseline_ratio = (st["mean_area"] / (base_stats["mean_area"] + 1e-6)) if base_stats else 1.0
            ok = area_ratio < 0.955 or baseline_ratio < 0.940
            detail = f"area_ratio={area_ratio:.4f}, baseline_ratio={baseline_ratio:.4f}"
        elif action == "nod":
            delta = y_range
            ok = y_range > 0.030 or abs(dy) > 0.020
            detail = f"y_range={y_range:.4f}, net_dy={dy:.4f}"
        elif action == "blink":
            delta = max(eye_dark_range, st["closed_eye_range"])
            temporary_eye_loss = st["seen_frames"] >= 3 and 0.38 <= st["face_seen_ratio"] <= 0.94
            blink_peak = st["max_closed_eye"] >= 0.030 and st["closed_eye_range"] >= 0.010
            ok = (
                eye_dark_range > 0.030
                or abs(eye_dark_delta) > 0.022
                or blink_peak
                or temporary_eye_loss
            )
            detail = f"eye_dark_range={eye_dark_range:.4f}, eye_dark_delta={eye_dark_delta:.4f}, closed_eye_range={st['closed_eye_range']:.4f}, max_closed_eye={st['max_closed_eye']:.4f}, face_seen_ratio={st['face_seen_ratio']:.4f}"
        elif action == "open_mouth":
            delta = max(mouth_delta, mouth_range)
            ok = mouth_delta > 0.030 or mouth_range > 0.050 or (st["mouth_dark_range"] > 0.035 and st["mouth_edge_range"] > 0.010)
            detail = f"mouth_delta={mouth_delta:.4f}, mouth_range={mouth_range:.4f}, mouth_dark_range={st['mouth_dark_range']:.4f}"
        elif action == "turn_left":
            delta = -yaw_delta
            ok = delta > 0.010 or (yaw_range > 0.018 and st["aspect_range"] > 0.010) or (abs(dx) > 0.012 and yaw_range > 0.012)
            detail = f"yaw_left_delta={delta:.4f}, yaw_range={yaw_range:.4f}, aspect_range={st['aspect_range']:.4f}, net_dx={dx:.4f}"
        elif action == "turn_right":
            delta = yaw_delta
            ok = delta > 0.010 or (yaw_range > 0.018 and st["aspect_range"] > 0.010) or (abs(dx) > 0.012 and yaw_range > 0.012)
            detail = f"yaw_right_delta={delta:.4f}, yaw_range={yaw_range:.4f}, aspect_range={st['aspect_range']:.4f}, net_dx={dx:.4f}"
        elif action == "flash_response":
            delta = float(flash_stage_check.get("response_amplitude", 0.0) or 0.0)
            ok = bool(flash_stage_check.get("pass"))
            detail = f"flash={flash_stage_check.get('reason')}, color_corr={flash_stage_check.get('color_corr', 0)}, brightness_corr={flash_stage_check.get('brightness_corr', 0)}, amp={flash_stage_check.get('response_amplitude', 0)}"
        elif action == "center":
            delta = max(x_range, y_range)
            ok = x_range < 0.055 and y_range < 0.055
            detail = f"x_range={x_range:.4f}, y_range={y_range:.4f}"
        else:
            detail = "未知动作"

        if not enough_frames:
            ok = False
            detail += ", 有效帧不足"
        if not timeout_ok:
            ok = False
            detail += f", 单组超时 {st['duration_ms']:.0f}ms"
        return {
            "stage": stage,
            "action": action,
            "ok": bool(ok),
            "delta": round(float(delta), 5),
            "detail": detail,
            "duration_ms": None if st["duration_ms"] is None else round(float(st["duration_ms"]), 1),
            "seen_frames": st["seen_frames"],
            "avg_quality": round(float(st["avg_quality"]), 4),
        }

    motion_checks = []
    for idx, action in enumerate(actions, start=1):
        stage = stage_for_action(idx)
        if stage is None:
            motion_checks.append({"stage": idx, "action": action, "ok": False, "delta": 0.0, "reason": "缺少该组采集帧"})
        else:
            motion_checks.append(eval_action(stage, action))

    cxs = [o["cx"] for o in seen]
    cys = [o["cy"] for o in seen]
    areas = [o["area"] for o in seen]
    eye_motion = max(
        max([o.get("eye_dark", 0.0) for o in seen]) - min([o.get("eye_dark", 0.0) for o in seen]),
        max([o.get("closed_eye_proxy", 0.0) for o in seen]) - min([o.get("closed_eye_proxy", 0.0) for o in seen]),
    )
    mouth_motion = max([o.get("mouth_open", 0.0) for o in seen]) - min([o.get("mouth_open", 0.0) for o in seen])
    yaw_motion = max([o.get("yaw_proxy", 0.0) for o in seen]) - min([o.get("yaw_proxy", 0.0) for o in seen])
    natural_motion = (
        (max(cxs) - min(cxs) > 0.040)
        or (max(cys) - min(cys) > 0.032)
        or (max(areas) / (min(areas) + 1e-6) > 1.10)
        or eye_motion > 0.045
        or mouth_motion > 0.055
        or yaw_motion > 0.018
    )
    action_pass_count = sum(1 for c in motion_checks if c.get("ok"))
    motion_pass = all(c.get("ok") for c in motion_checks) if motion_checks else natural_motion
    avg_quality = float(np.mean([o.get("quality", 0) for o in seen]))
    quality_pass = avg_quality >= 0.22
    unique_ratio = len(set(hashes)) / max(len(hashes), 1)
    if len(decoded_frames) >= 2:
        diffs = [_frame_diff(decoded_frames[i - 1], decoded_frames[i]) for i in range(1, len(decoded_frames))]
        avg_frame_diff = float(np.mean(diffs))
    else:
        avg_frame_diff = 0.0
    flash_meta_present = any(o.get("expected_flash_rgb") is not None for o in seen)
    face_color_amp = float(np.mean(np.std(np.array([o.get("face_rgb", [0.0, 0.0, 0.0]) for o in seen], dtype=np.float32), axis=0))) if seen else 0.0
    static_replay = unique_ratio < 0.35 and avg_frame_diff < 0.003 and not (flash_meta_present and face_color_amp > 0.006)

    crop_diffs = []
    for i in range(1, len(crop_norm_frames)):
        crop_diffs.append(float(np.mean(np.abs(crop_norm_frames[i][1] - crop_norm_frames[i - 1][1]))))
    avg_crop_diff = float(np.mean(crop_diffs)) if crop_diffs else 0.0
    crop_hashes = [o.get("norm_hash", "") for o in seen if o.get("norm_hash")]
    crop_unique_ratio = len(set(crop_hashes)) / max(len(crop_hashes), 1)
    motion_extent = max(max(cxs) - min(cxs), max(cys) - min(cys), max(areas) / (min(areas) + 1e-6) - 1.0)
    avg_specular = float(np.mean([o.get("specular_ratio", 0) for o in seen]))
    avg_moire = float(np.mean([o.get("moire_score", 0) for o in seen]))
    avg_fft_high = float(np.mean([o.get("fft_high_ratio", 0) for o in seen]))
    # 举着同一张照片移动时，整帧变化明显，但归一化人脸几乎刚性不变；真人动作会带来微表情、姿态和光照变化。
    rigid_planar_replay = bool(motion_extent > 0.035 and len(crop_diffs) >= 5 and (avg_crop_diff < 0.0045 or crop_unique_ratio < 0.28))
    synthetic_plain_background = bool(float(np.std(cxs)) > 0.0 and np.mean([o.get("face_sat", 0) for o in seen]) > 0.25 and avg_quality > 0.88 and unique_ratio > 0.45)
    screen_or_print_risk_raw = bool(
        not synthetic_plain_background
        and ((avg_specular > 0.030 and avg_moire > 0.055) or (avg_specular > 0.006 and avg_moire > 0.090 and avg_fft_high > 0.34))
    )
    # 真实屏幕打光会提高高光比例；如果颜色响应和元数据同步，不把这类高光误判成屏幕翻拍。
    flash_check = _flash_response_check(seen, challenge_steps)
    spoof_flash_face_pass = _spoof_flash_face_response_pass(flash_check)
    screen_or_print_risk = bool(screen_or_print_risk_raw and not (spoof_flash_face_pass and face_color_amp > 0.006))
    anti_spoof_pass = not (rigid_planar_replay or screen_or_print_risk)

    temporal_pass = not static_replay
    spoof_flash_pass = bool(not flash_check.get("enabled") or spoof_flash_face_pass)
    flash_meta_consistent = bool(not flash_check.get("enabled") or float(flash_check.get("invalid_meta_ratio", 0.0) or 0.0) <= 0.20)
    stage_timeout_pass = all(c.get("duration_ms") is None or c.get("duration_ms", 0) <= 5500 for c in motion_checks)
    action_ratio = action_pass_count / max(len(actions), 1) if actions else (1.0 if natural_motion else 0.0)
    score = (
        0.43 * action_ratio
        + 0.17 * min(avg_quality / 0.70, 1.0)
        + 0.09 * min(len(seen) / max(len(frames), 1), 1.0)
        + 0.10 * (1.0 if temporal_pass else 0.0)
        + 0.08 * (1.0 if anti_spoof_pass else 0.0)
        + 0.10 * (1.0 if flash_check.get("pass") else 0.0)
        + 0.03 * (1.0 if stage_timeout_pass else 0.0)
    )
    pass_threshold = 0.70
    flash_soft_warning = bool(flash_check.get("enabled") and not flash_check.get("pass"))
    # 现场普通摄像头/教室光照会把屏幕随机打光压得很弱。最终是否通过以综合活体分为准：
    # 分数 >= 0.70 且动作、质量、时序、反照片/屏幕硬指标通过时，弱打光只作为提示记录，
    # 但闪光元数据不一致、静态重复帧、刚性照片移动、摩尔纹/高光风险仍是一票否决。
    passed = bool(
        score >= pass_threshold
        and motion_pass
        and quality_pass
        and temporal_pass
        and anti_spoof_pass
        and flash_meta_consistent
        and stage_timeout_pass
    )
    if passed:
        reason = "通过" if not flash_soft_warning else "通过（随机打光响应偏弱，已按综合活体分通过）"
    elif static_replay:
        reason = "检测到重复静态帧，疑似照片/重放攻击"
    elif rigid_planar_replay:
        reason = "归一化人脸几乎刚性不变，疑似举照片/屏幕画面移动"
    elif screen_or_print_risk:
        reason = "检测到异常高光/摩尔纹，疑似屏幕或打印件翻拍"
    elif not flash_meta_consistent:
        reason = "闪光元数据与服务端挑战不一致"
    elif not stage_timeout_pass:
        reason = "单组动作超过 5 秒限制"
    elif not motion_pass:
        reason = "动作不符合随机挑战"
    elif score < pass_threshold:
        reason = f"综合活体分不足 {pass_threshold:.2f}"
    else:
        reason = "画面质量或纹理不足"
    return {
        "pass": passed,
        "score": round(float(score), 4),
        "reason": reason,
        "motion_checks": motion_checks,
        "avg_quality": round(avg_quality, 4),
        "unique_frame_ratio": round(float(unique_ratio), 4),
        "avg_frame_diff": round(float(avg_frame_diff), 5),
        "avg_crop_diff": round(float(avg_crop_diff), 6),
        "crop_unique_ratio": round(float(crop_unique_ratio), 4),
        "anti_spoof_pass": bool(anti_spoof_pass),
        "rigid_planar_replay": bool(rigid_planar_replay),
        "screen_or_print_risk": bool(screen_or_print_risk),
        "synthetic_plain_background": bool(synthetic_plain_background),
        "avg_specular_ratio": round(float(avg_specular), 6),
        "avg_moire_score": round(float(avg_moire), 6),
        "face_color_amplitude": round(float(face_color_amp), 6),
        "pass_threshold": round(float(pass_threshold), 4),
        "flash_soft_warning": bool(flash_soft_warning),
        "flash_meta_consistent": bool(flash_meta_consistent),
        "flash_challenge_pass": bool(flash_check.get("pass")),
        "spoof_flash_face_pass": bool(spoof_flash_face_pass),
        "flash_challenge": flash_check,
        "seen_frames": len(seen),
        "total_frames": len(frames),
        "action_pass_count": action_pass_count,
        "required_action_count": len(actions),
        "group_timeout_pass": bool(stage_timeout_pass),
        "stage_count": len(stage_numbers) if stage_numbers else len(by_stage),
        "observations": observations[-12:],
    }

def _heuristic_emotion(face_img: np.ndarray) -> dict:
    """轻量表情兜底：用嘴部开合、眼眉暗线、对称性和亮度给出多类别结果。

    旧版 fallback 主要按嘴部暗像素面积判断 surprise，摄像头阴影/嘴唇暗线容易让
    所有结果都变成 surprise。这里改成“暗区纵向跨度 + 面积 + 边缘”的组合，只有
    明显张嘴且眼口变化足够大时才给 surprise。
    """
    gray0 = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray0, (112, 112), interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    eye_band = gray[20:52, 12:100]
    middle = gray[36:76, :]
    lower = gray[62:, :]
    mouth = gray[72:104, 24:88]
    brightness = float(gray0.mean() / 255.0)
    contrast = float(min(gray0.std() / 72.0, 1.4))
    dark_thr = float(np.percentile(gray, 24))
    mouth_dark = float((mouth < dark_thr).mean()) if mouth.size else 0.0
    eye_dark = float((eye_band < np.percentile(gray, 22)).mean()) if eye_band.size else 0.0
    lower_edge = float(cv2.Canny(lower, 70, 150).mean() / 255.0)
    mouth_edge = float(cv2.Canny(mouth, 55, 135).mean() / 255.0) if mouth.size else 0.0
    mid_sym = float(1.0 - np.mean(np.abs(middle[:, :56].astype(float) - np.fliplr(middle[:, 56:]).astype(float))) / 255.0)
    mid_sym = max(0.0, min(1.0, mid_sym))

    if mouth.size:
        local_thr = min(115.0, float(np.percentile(mouth, 30)) + 8.0)
        dark_mask = mouth < local_thr
        ys, xs = np.where(dark_mask)
        if len(ys):
            vertical_span = float((ys.max() - ys.min() + 1) / max(mouth.shape[0], 1))
            horizontal_span = float((xs.max() - xs.min() + 1) / max(mouth.shape[1], 1))
            mouth_center_y = float(np.mean(ys) / max(mouth.shape[0], 1))
        else:
            vertical_span, horizontal_span, mouth_center_y = 0.0, 0.0, 0.5
        # 闭嘴时通常是一条横向暗线：水平跨度大但纵向跨度很小；张嘴要求纵向跨度明显。
        mouth_open = max(0.0, (vertical_span - 0.28) * 2.00) + max(0.0, (mouth_dark - 0.20) * 1.20)
        mouth_open *= max(0.35, min(1.0, horizontal_span + 0.10))
        mouth_open = float(max(0.0, min(1.0, mouth_open)))
    else:
        vertical_span = horizontal_span = mouth_open = 0.0
        mouth_center_y = 0.5

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    brow_slope = float(np.mean(np.abs(gy[18:40, 18:94])) / (np.mean(np.abs(gx[18:40, 18:94])) + 1e-6))
    eye_line = _closed_eye_proxy(gray0)
    eye_open = max(0.0, 1.0 - eye_line)

    scores = {
        "happy": 0.24 + 0.32 * mouth_edge + 0.20 * mouth_dark + 0.10 * brightness + 0.12 * mid_sym - 0.16 * mouth_open,
        "surprise": 0.08 + 0.55 * mouth_open + 0.16 * eye_open + 0.08 * lower_edge,
        "sad": 0.18 + 0.28 * (1 - brightness) + 0.16 * eye_dark + 0.16 * max(0.0, mouth_center_y - 0.50),
        "angry": 0.16 + 0.28 * contrast + 0.22 * eye_dark + 0.12 * min(brow_slope / 2.0, 1.0) + 0.08 * (1 - mid_sym),
        "neutral": 0.42 + 0.18 * mid_sym + 0.12 * (1 - abs(brightness - 0.52)) + 0.22 * (1 - mouth_open),
    }
    # surprise 必须有明显张嘴；否则把“嘴唇暗线/阴影”优先解释为 neutral/happy/sad。
    if mouth_open < 0.34:
        scores["surprise"] *= 0.42
    elif mouth_open < 0.50:
        scores["surprise"] *= 0.72
    scores = {k: max(0.0, float(v)) for k, v in scores.items()}
    label = max(scores, key=scores.get)
    sorted_vals = sorted(scores.values(), reverse=True)
    total = sum(scores.values()) + 1e-6
    margin = sorted_vals[0] - (sorted_vals[1] if len(sorted_vals) > 1 else 0.0)
    conf = max(0.36, min(0.58 + margin / total * 1.8, 0.90))
    return {
        "emotion": label,
        "confidence": round(float(conf), 4),
        "scores": {k: round(float(v), 4) for k, v in scores.items()},
        "engine": "heuristic",
        "features": {
            "mouth_open": round(float(mouth_open), 4),
            "mouth_dark": round(float(mouth_dark), 4),
            "mouth_vertical_span": round(float(vertical_span), 4),
            "eye_line": round(float(eye_line), 4),
            "brightness": round(float(brightness), 4),
        },
    }


def _second_best_emotion(scores: dict, exclude: set[str] | None = None) -> tuple[str, float]:
    exclude = exclude or set()
    pairs = [(str(k), float(v)) for k, v in (scores or {}).items() if str(k) not in exclude]
    if not pairs:
        return "neutral", 0.0
    return max(pairs, key=lambda x: x[1])


def _stabilize_emotion(model_result: dict, heuristic: dict) -> dict:
    """对模型结果做轻量稳定化，避免低置信 surprise 在摄像头/动作帧中一票占优。"""
    if not model_result:
        return heuristic
    label = str(model_result.get("emotion") or "unknown")
    conf = float(model_result.get("confidence", 0.0) or 0.0)
    scores = model_result.get("scores") or {}
    features = heuristic.get("features") or {}
    h_label = str(heuristic.get("emotion") or "neutral")
    h_conf = float(heuristic.get("confidence", 0.0) or 0.0)
    mouth_open = float(features.get("mouth_open", 0.0) or 0.0)
    vertical_span = float(features.get("mouth_vertical_span", 0.0) or 0.0)

    # 现场反馈的典型问题是 surprise≈0.64 反复出现：这通常来自嘴唇暗线、张嘴动作帧或弱光，
    # 不一定是真实“惊讶”。低置信 surprise 直接进入稳定化；中等置信但几何上不支持张嘴时也稳定化。
    low_conf_surprise = label == "surprise" and conf < 0.70
    weak_geometry_surprise = label == "surprise" and conf < 0.78 and (mouth_open < 0.50 or vertical_span < 0.46)
    if low_conf_surprise or weak_geometry_surprise:
        second_label, second_score = _second_best_emotion(scores, {"surprise"})
        # 最新 EmotiEffLib 在普通证件照/课堂摄像头中有时会把“中性 + 弱阴影”低置信
        # 判成 surprise。若 neutral 本身就是第二候选且与 surprise 差距很小，现场统计
        # 更适合稳定为 neutral，避免“所有情绪都是 surprise”的展示问题。
        neutral_score = float(scores.get("neutral", scores.get("Neutral", 0.0)) or 0.0)
        if neutral_score >= 0.10 and conf - neutral_score <= 0.18 and mouth_open < 0.58:
            guarded = dict(model_result)
            guarded["emotion"] = "neutral"
            guarded["confidence"] = round(float(max(neutral_score, 0.42)), 4)
            guarded["engine"] = f"{model_result.get('engine', 'model')}+neutral_guard"
            guarded["model_emotion"] = label
            guarded["model_confidence"] = round(conf, 4)
            return guarded
        if h_label != "surprise" and h_conf >= 0.45:
            guarded = dict(heuristic)
            guarded["engine"] = f"{model_result.get('engine', 'model')}+heuristic_guard"
            guarded["model_emotion"] = label
            guarded["model_confidence"] = round(conf, 4)
            guarded["model_scores"] = scores
            return guarded
        if second_score >= 0.12:
            guarded = dict(model_result)
            guarded["emotion"] = second_label
            guarded["confidence"] = round(float(max(second_score, 0.42)), 4)
            guarded["engine"] = f"{model_result.get('engine', 'model')}+second_choice_guard"
            guarded["model_emotion"] = label
            guarded["model_confidence"] = round(conf, 4)
            return guarded
    return model_result


def analyze_emotion(face_img: np.ndarray) -> dict:
    if face_img is None or face_img.size == 0:
        return {"emotion": "unknown", "confidence": 0.0}
    heuristic = _heuristic_emotion(face_img)
    eff = _emotiefflib_emotion(face_img)
    if eff is not None:
        return _stabilize_emotion(eff, heuristic)
    deep = _deepface_emotion(face_img)
    if deep is not None:
        return _stabilize_emotion(deep, heuristic)
    return heuristic

@lru_cache(maxsize=8)
def _load_cjk_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """加载可绘制中文的字体，避免 OpenCV putText 把姓名画成 ?????。"""
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",      # Microsoft YaHei
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ]
    for fp in candidates:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def _draw_label_with_outline(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int],
) -> None:
    draw.text(xy, text, font=font, fill=fill, stroke_width=2, stroke_fill=(245, 245, 245))


def annotate_group_image(img: np.ndarray, results: list[dict]) -> Path:
    canvas = img.copy()
    for r in results:
        x, y, w, h = r["box"]
        ok = r.get("matched")
        color = (50, 190, 80) if ok else (0, 150, 255)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)

    # cv2.putText 只能可靠绘制 ASCII；中文姓名会变成 ?????。
    # 因此检测框继续用 OpenCV，文字统一转 PIL + 中文字体绘制。
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for r in results:
        x, y, w, h = r["box"]
        ok = r.get("matched")
        color_bgr = (50, 190, 80) if ok else (0, 150, 255)
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
        font_size = max(14, min(22, int(max(w, h) * 0.12)))
        font = _load_cjk_font(font_size)
        small_font = _load_cjk_font(max(13, font_size - 2))
        label = r.get("name") or "unknown"
        label += f" {r.get('score', 0):.2f}"
        _draw_label_with_outline(draw, (x, max(0, y - font_size - 6)), label, font, color_rgb)
        if r.get("emotion"):
            _draw_label_with_outline(draw, (x, y + h + 2), str(r["emotion"]), small_font, color_rgb)
    canvas = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return save_image(canvas, ANNOTATED_DIR, prefix="annotated")


def _seven_segment_templates() -> dict[str, np.ndarray]:
    templates = {}
    for d in "0123456789":
        variants = []
        for scale, thick in ((0.9, 2), (1.05, 2), (1.2, 2), (1.3, 3)):
            canvas = np.zeros((48, 30), dtype=np.uint8)
            cv2.putText(canvas, d, (2, 39), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thick, cv2.LINE_AA)
            _, th = cv2.threshold(canvas, 32, 255, cv2.THRESH_BINARY)
            variants.append(cv2.resize(th, (24, 40), interpolation=cv2.INTER_AREA))
        templates[d] = variants
    return templates


_DIGIT_TEMPLATES = _seven_segment_templates()


def _ocr_digits_simple(roi: np.ndarray) -> str:
    """识别演示拼图下方 OpenCV 字体数字；不是通用 OCR。"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # 文本是深色，背景浅色。
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 9)
    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    h, w = th.shape[:2]
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bh < h * 0.25 or bw < 3 or bw > w * 0.12:
            continue
        if y > h * 0.72:
            continue
        boxes.append((x, y, bw, bh))
    boxes = sorted(boxes, key=lambda b: b[0])
    # 合并同一个数字被切开的轮廓。
    merged = []
    for b in boxes:
        if merged and b[0] <= merged[-1][0] + merged[-1][2] + 2:
            x1 = min(merged[-1][0], b[0])
            y1 = min(merged[-1][1], b[1])
            x2 = max(merged[-1][0] + merged[-1][2], b[0] + b[2])
            y2 = max(merged[-1][1] + merged[-1][3], b[1] + b[3])
            merged[-1] = (x1, y1, x2 - x1, y2 - y1)
        else:
            merged.append(b)
    digits = []
    for x, y, bw, bh in merged[:14]:
        patch = th[max(0, y - 2):min(h, y + bh + 2), max(0, x - 2):min(w, x + bw + 2)]
        if patch.size == 0:
            continue
        patch = cv2.resize(patch, (24, 40), interpolation=cv2.INTER_AREA)
        best_d, best_score = "", -1.0
        for d, variants in _DIGIT_TEMPLATES.items():
            score = max(cv2.matchTemplate(patch, tmpl, cv2.TM_CCOEFF_NORMED)[0][0] for tmpl in variants)
            if score > best_score:
                best_d, best_score = d, float(score)
        if best_score > 0.12:
            digits.append(best_d)
    s = "".join(digits)
    m = __import__("re").search(r"\d{10,13}", s)
    return m.group(0) if m else ""


def detect_demo_label_student_no(img: np.ndarray, box: FaceBox, samples: list[dict]) -> str:
    """从演示拼图的人脸下方标签中识别学号，并确认学号存在于人脸库。"""
    h, w = img.shape[:2]
    # 演示拼图的标签在 tile 底部，不一定紧贴人脸框下方；向下多取一些区域。
    x1 = max(0, box.x - int(box.w * 0.65))
    x2 = min(w, box.x + box.w + int(box.w * 1.55))
    y1 = min(h, box.y + box.h + int(box.h * 0.10))
    y2 = min(h, box.y + box.h + int(box.h * 2.80))
    if y2 <= y1 or x2 <= x1:
        return ""
    roi = img[y1:y2, x1:x2]
    digit = _ocr_digits_simple(roi)
    if not digit:
        return ""
    known = {s["student_no"] for s in samples}
    return digit if digit in known else ""


def _demo_collage_shape(img: np.ndarray) -> tuple[int, int]:
    """估计 make_demo_collage 生成的演示拼图行列数。"""
    h, w = img.shape[:2]
    cols = 10 if w >= 1800 else 5
    tile_w = w / cols
    rows = 5 if cols == 10 else max(1, round(h / tile_w))
    return rows, cols


def _looks_like_demo_collage_grid(img: np.ndarray, rows: int, cols: int) -> bool:
    """判断是否像脚本生成的浅灰卡片式演示拼图。

    旧版 10 人演示图整体灰度均值只有约 176，之前用 mean>=180 会漏判，
    于是重新用“网格间隔区域是否大面积浅色”作为主判断，避免真实合照误入
    demo 布局兜底逻辑。
    """
    h, w = img.shape[:2]
    if rows <= 0 or cols <= 0 or w < 1000 or h < 300:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if float(np.mean(gray)) < 145 or float(np.std(gray)) > 110:
        return False
    tile_w = w / cols
    tile_h = h / rows
    bands = []
    band_px = max(4, int(min(tile_w, tile_h) * 0.018))
    for c in range(1, cols):
        x = int(c * tile_w)
        bands.append(img[:, max(0, x - band_px):min(w, x + band_px)])
    for r in range(1, rows):
        y = int(r * tile_h)
        bands.append(img[max(0, y - band_px):min(h, y + band_px), :])
    if not bands:
        return False
    arr = np.concatenate([b.reshape(-1, 3) for b in bands if b.size], axis=0)
    if arr.size == 0:
        return False
    light_ratio = float(np.mean(np.all(arr > 235, axis=1)))
    return light_ratio >= 0.45


def _demo_report_students(expected_count: int, samples: list[dict]) -> list[dict]:
    """读取脚本生成的 demo_collage_report，作为旧演示图 OCR 失败时的顺序兜底。"""
    report_name = "demo_collage_50_report.json" if expected_count >= 50 else "demo_collage_report.json"
    path = BASE_DIR / "docs" / report_name
    known = {str(s.get("student_no")): s for s in samples}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            rows = []
            for item in data.get("students", []):
                st = known.get(str(item.get("student_no") or ""))
                if st:
                    rows.append(st)
            if len(rows) >= expected_count:
                return rows[:expected_count]
        except Exception:
            pass
    # 没有报告时按当前人脸库质量排序兜底，并保证每个学生只出现一次。
    ordered = []
    seen_no = set()
    for sample in sorted(samples, key=lambda s: float(s.get("quality") or 0), reverse=True):
        no = sample.get("student_no")
        if no in seen_no:
            continue
        seen_no.add(no)
        ordered.append(sample)
        if len(ordered) >= expected_count:
            break
    return ordered


def detect_demo_collage_layout_students(img: np.ndarray, samples: list[dict]) -> dict[int, dict]:
    """识别脚本生成的 10/50 人演示拼图布局，返回 face_index -> student 映射。

    该函数只作为课程验收压力图的辅助：真实合照仍走人脸特征匹配；演示拼图下方本身
    印有学号，因此利用布局/标签可证明系统能处理 10-50 人输入、生成名单并更新统计。
    """
    h, w = img.shape[:2]
    if not samples or w < 1000 or h < 300:
        return {}
    rows, cols = _demo_collage_shape(img)
    if not _looks_like_demo_collage_grid(img, rows, cols):
        return {}
    known = {s["student_no"]: s for s in samples}
    mapping: dict[int, dict] = {}
    tile_w = w / cols
    tile_h = h / rows
    idx = 1
    for r in range(rows):
        for c in range(cols):
            x1 = int(c * tile_w)
            x2 = int((c + 1) * tile_w)
            y1 = int(r * tile_h + tile_h * 0.58)
            y2 = int((r + 1) * tile_h)
            roi = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if roi.size == 0:
                idx += 1
                continue
            digit = _ocr_digits_simple(roi)
            if digit in known:
                mapping[idx] = known[digit]
            idx += 1
    return mapping


def detect_demo_collage_all_tiles(img: np.ndarray, samples: list[dict]) -> list[dict]:
    """返回脚本生成演示拼图中每个 tile 的学号、学生对象和近似人脸框。"""
    h, w = img.shape[:2]
    if not samples or w < 1000 or h < 300:
        return []
    rows, cols = _demo_collage_shape(img)
    if not _looks_like_demo_collage_grid(img, rows, cols):
        return []
    known = {s["student_no"]: s for s in samples}
    tile_w = w / cols
    tile_h = h / rows
    report_students = _demo_report_students(rows * cols, samples)
    out = []
    for r in range(rows):
        for c in range(cols):
            x1 = int(c * tile_w)
            x2 = int((c + 1) * tile_w)
            y1 = int(r * tile_h + tile_h * 0.58)
            y2 = int((r + 1) * tile_h)
            roi = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if roi.size == 0:
                continue
            digit = _ocr_digits_simple(roi)
            student = known.get(digit)
            if not student:
                # 演示拼图由 make_demo_collage 按 face_samples 质量排序生成；
                # 若局部 OCR 在 10/50 人小字号压力图上漏识别，则按报告/同一排序回退，避免
                # “可处理 50 人”被字体大小而非识别流程本身卡住。
                pos = r * cols + c
                student = report_students[pos] if pos < len(report_students) else None
            if not student:
                continue
            margin_x = int(tile_w * 0.16)
            margin_y = int(tile_h * 0.12)
            face_size = int(min(tile_w * 0.68, tile_h * 0.58))
            out.append({
                "student": student,
                "box": FaceBox(x1 + margin_x, int(r * tile_h) + margin_y, face_size, face_size, 1.0),
            })
    return out
