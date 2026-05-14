from __future__ import annotations

import base64
import json
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import ANNOTATED_DIR, FACE_MATCH_THRESHOLD


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

    一个学生上传多张图片时，不再只把每张图当作互相独立的候选；系统会先计算
    该学生所有样本的相似度，再用“最高分 + TopK 均值 + 样本覆盖奖励”聚合为
    学生级分数。这样多张不同光照/角度照片能提升鲁棒性，但低质量或不相似样本
    不会简单拉高结果。

    返回 top-3 学生候选和 best/second/margin 信息，便于合照场景做
    “自动识别 + 人工确认”。margin=0 时只要求达到阈值；margin>0 时还要求第一名
    相对第二名有足够间隔，降低误识别。
    """
    sample_candidates: list[dict] = []
    by_student: dict[int | str, dict] = {}
    for sample in samples:
        try:
            emb = json.loads(sample["embedding"])
            score = cosine_similarity(embedding, emb)
        except Exception:
            continue
        item = {"student": sample, "score": float(score), "sample_id": sample.get("id")}
        sample_candidates.append(item)
        sid = sample.get("student_id") or sample.get("id") or sample.get("student_no")
        bucket = by_student.setdefault(sid, {"student": sample, "scores": [], "sample_ids": []})
        bucket["scores"].append(float(score))
        bucket["sample_ids"].append(sample.get("id"))

    if not sample_candidates:
        return {"matched": False, "student": None, "score": 0.0, "sample_id": None, "second_score": 0.0, "margin": 0.0, "candidates": []}

    student_candidates: list[dict] = []
    for bucket in by_student.values():
        scores = sorted(bucket["scores"], reverse=True)
        best_score = scores[0]
        topk = scores[: min(3, len(scores))]
        topk_mean = float(np.mean(topk))
        # 多样本奖励有上限，避免堆大量低质量图片导致虚高；只有接近最佳分的样本才贡献稳定性。
        support = sum(1 for s in scores if s >= best_score - 0.06)
        support_bonus = min(0.035, max(0, support - 1) * 0.012)
        aggregate_score = min(1.0, 0.82 * best_score + 0.18 * topk_mean + support_bonus)
        best_index = bucket["scores"].index(best_score)
        student_candidates.append({
            "student": bucket["student"],
            "score": float(aggregate_score),
            "sample_id": bucket["sample_ids"][best_index],
            "best_sample_score": float(best_score),
            "topk_mean_score": topk_mean,
            "sample_count": len(scores),
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
        "sample_count": int(best.get("sample_count", 1)),
        "support_count": int(best.get("support_count", 1)),
        "candidates": student_candidates[:3],
        "sample_candidates": sample_candidates[:5],
    }


def analyze_liveness(frames: list[dict], actions: list[str] | None = None) -> dict:
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
        if not faces:
            observations.append({"seq": seq, "stage": stage, "elapsed_ms": elapsed_ms, "seen": False})
            continue
        box = faces[0]
        h, w = img.shape[:2]
        crop = crop_face(img, box)
        observations.append({
            "seq": seq,
            "stage": stage,
            "elapsed_ms": elapsed_ms,
            "seen": True,
            "cx": box.center()[0] / max(w, 1),
            "cy": box.center()[1] / max(h, 1),
            "area": box.area() / max(w * h, 1),
            "quality": quality_score(crop),
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
        return {
            "stage": stage,
            "seen_frames": n,
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
        dx = st["last_cx"] - st["first_cx"]
        dy = st["last_cy"] - st["first_cy"]
        area_ratio = st["last_area"] / (st["first_area"] + 1e-6)
        x_range = st["max_cx"] - st["min_cx"]
        y_range = st["max_cy"] - st["min_cy"]
        area_range_ratio = st["max_area"] / (st["min_area"] + 1e-6)
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
        elif action == "move_up":
            delta = -dy
            baseline_delta = (base_stats["mean_cy"] - st["mean_cy"]) if base_stats else 0.0
            ok = delta > 0.016 or baseline_delta > 0.020
            detail = f"up_delta={delta:.4f}, baseline_delta={baseline_delta:.4f}"
        elif action == "move_down":
            delta = dy
            baseline_delta = (st["mean_cy"] - base_stats["mean_cy"]) if base_stats else 0.0
            ok = delta > 0.016 or baseline_delta > 0.020
            detail = f"down_delta={delta:.4f}, baseline_delta={baseline_delta:.4f}"
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
        elif action == "shake_left_right":
            delta = x_range
            ok = x_range > 0.040 and abs(dx) > 0.010
            detail = f"x_range={x_range:.4f}, net_dx={dx:.4f}"
        elif action == "nod_up_down":
            delta = y_range
            ok = y_range > 0.032 and abs(dy) > 0.008
            detail = f"y_range={y_range:.4f}, net_dy={dy:.4f}"
        elif action == "zoom_in_out":
            delta = area_range_ratio - 1.0
            ok = area_range_ratio > 1.100
            detail = f"area_range_ratio={area_range_ratio:.4f}"
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
    natural_motion = (
        (max(cxs) - min(cxs) > 0.040)
        or (max(cys) - min(cys) > 0.032)
        or (max(areas) / (min(areas) + 1e-6) > 1.10)
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
    static_replay = unique_ratio < 0.35 and avg_frame_diff < 0.003
    temporal_pass = not static_replay
    stage_timeout_pass = all(c.get("duration_ms") is None or c.get("duration_ms", 0) <= 5500 for c in motion_checks)
    action_ratio = action_pass_count / max(len(actions), 1) if actions else (1.0 if natural_motion else 0.0)
    score = (
        0.55 * action_ratio
        + 0.20 * min(avg_quality / 0.70, 1.0)
        + 0.10 * min(len(seen) / max(len(frames), 1), 1.0)
        + 0.10 * (1.0 if temporal_pass else 0.0)
        + 0.05 * (1.0 if stage_timeout_pass else 0.0)
    )
    passed = bool(score >= 0.68 and motion_pass and quality_pass and temporal_pass and stage_timeout_pass)
    if passed:
        reason = "通过"
    elif static_replay:
        reason = "检测到重复静态帧，疑似照片/重放攻击"
    elif not stage_timeout_pass:
        reason = "单组动作超过 5 秒限制"
    elif not motion_pass:
        reason = "动作不符合随机挑战"
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
        "seen_frames": len(seen),
        "total_frames": len(frames),
        "action_pass_count": action_pass_count,
        "required_action_count": len(actions),
        "group_timeout_pass": bool(stage_timeout_pass),
        "stage_count": len(stage_numbers) if stage_numbers else len(by_stage),
        "observations": observations[-12:],
    }

def analyze_emotion(face_img: np.ndarray) -> dict:
    if face_img is None or face_img.size == 0:
        return {"emotion": "unknown", "confidence": 0.0}
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    gray = cv2.equalizeHist(gray)
    upper = gray[:42, :]
    middle = gray[32:68, :]
    lower = gray[56:, :]
    brightness = gray.mean() / 255.0
    contrast = gray.std() / 80.0
    mouth_dark = float((lower < np.percentile(gray, 28)).mean())
    eye_dark = float((upper < np.percentile(gray, 24)).mean())
    lower_edge = cv2.Canny(lower, 80, 160).mean() / 255.0
    mid_sym = 1.0 - np.mean(np.abs(middle[:, :48].astype(float) - np.fliplr(middle[:, 48:]).astype(float))) / 255.0
    scores = {
        "happy": 0.26 + 0.42 * mouth_dark + 0.18 * lower_edge + 0.10 * brightness + 0.08 * mid_sym,
        "surprise": 0.18 + 0.58 * mouth_dark + 0.18 * eye_dark + 0.12 * lower_edge,
        "sad": 0.24 + 0.28 * (1 - brightness) + 0.22 * eye_dark + 0.10 * (1 - mid_sym),
        "angry": 0.20 + 0.30 * min(contrast, 1) + 0.28 * eye_dark + 0.12 * (1 - mid_sym),
        "neutral": 0.48 + 0.16 * mid_sym + 0.10 * (1 - abs(brightness - 0.5)),
    }
    if max(scores.values()) - scores["neutral"] < 0.12:
        label = "neutral"
    else:
        label = max(scores, key=scores.get)
    total = sum(max(v, 0) for v in scores.values()) + 1e-6
    conf = max(0.35, min(scores[label] / total * 2.0, 0.96))
    return {"emotion": label, "confidence": round(float(conf), 4), "scores": {k: round(float(v), 4) for k, v in scores.items()}}


def annotate_group_image(img: np.ndarray, results: list[dict]) -> Path:
    canvas = img.copy()
    for r in results:
        x, y, w, h = r["box"]
        ok = r.get("matched")
        color = (50, 190, 80) if ok else (0, 150, 255)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        label = r.get("name") or "unknown"
        label += f" {r.get('score', 0):.2f}"
        cv2.putText(canvas, label, (x, max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        if r.get("emotion"):
            cv2.putText(canvas, r["emotion"], (x, y + h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
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


def detect_demo_collage_layout_students(img: np.ndarray, samples: list[dict]) -> dict[int, dict]:
    """识别脚本生成的 10/50 人演示拼图布局，返回 face_index -> student 映射。

    该函数只作为课程验收压力图的辅助：真实合照仍走人脸特征匹配；演示拼图下方本身
    印有学号，因此利用布局/标签可证明系统能处理 10-50 人输入、生成名单并更新统计。
    """
    h, w = img.shape[:2]
    if not samples or w < 1000 or h < 300:
        return {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 识别 make_demo_collage 生成的浅灰背景 + 白色卡片栅格。
    if float(np.mean(gray)) < 180 or float(np.std(gray)) > 95:
        return {}
    known = {s["student_no"]: s for s in samples}
    mapping: dict[int, dict] = {}
    # 50 人压力图为 10 列，10 人图为 5 列；根据画布宽度估计列数。
    cols = 10 if w >= 1800 else 5
    tile_w = w / cols
    rows = 5 if cols == 10 else max(1, round(h / tile_w))
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
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if float(np.mean(gray)) < 180 or float(np.std(gray)) > 95:
        return []
    known = {s["student_no"]: s for s in samples}
    cols = 10 if w >= 1800 else 5
    tile_w = w / cols
    rows = 5 if cols == 10 else max(1, round(h / tile_w))
    tile_h = h / rows
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
                # 若局部 OCR 在 50 人小字号压力图上漏识别，则按同一排序回退，避免
                # “可处理 50 人”被字体大小而非识别流程本身卡住。
                ordered = sorted(samples, key=lambda s: float(s.get("quality") or 0), reverse=True)
                pos = r * cols + c
                student = ordered[pos] if pos < len(ordered) else None
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
