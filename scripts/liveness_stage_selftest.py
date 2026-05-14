from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from app import create_app  # noqa: E402
from core.config import BASE_DIR  # noqa: E402
from core.db import db, init_db  # noqa: E402
from core.vision import analyze_liveness, read_image_path  # noqa: E402


def encode_jpeg(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("图片编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def load_seed_face() -> np.ndarray:
    """优先使用人脸库最佳样本；旧的 glob 顺序可能取到低质量/强反光样本。"""
    init_db(seed=True)
    with db() as conn:
        row = conn.execute("SELECT image_path FROM face_samples ORDER BY quality DESC LIMIT 1").fetchone()
    if row:
        return read_image_path(BASE_DIR / row["image_path"])
    faces = list((ROOT / "storage" / "faces").glob("**/*.jpg"))
    if not faces:
        raise SystemExit("缺少 storage/faces 样本，请先运行 python scripts/prepare_demo.py --source ..\\face_data")
    return read_image_path(faces[0])

def make_canvas(face: np.ndarray, *, x: int = 0, y: int = 0, scale: float = 1.0) -> np.ndarray:
    h, w = face.shape[:2]
    canvas_h, canvas_w = max(360, h * 2), max(480, w * 3)
    nw, nh = max(20, int(w * scale)), max(20, int(h * scale))
    resized = cv2.resize(face, (nw, nh))
    canvas = np.full((canvas_h, canvas_w, 3), 92, dtype=np.uint8)
    px = canvas_w // 2 - nw // 2 + int(x)
    py = canvas_h // 2 - nh // 2 + int(y)
    px = max(0, min(canvas_w - nw, px))
    py = max(0, min(canvas_h - nh, py))
    canvas[py:py + nh, px:px + nw] = resized
    return canvas


def _simulate_face_action(face: np.ndarray, action: str, idx: int, total: int) -> np.ndarray:
    """生成可被轻量动作特征识别的合成帧，用于回归测试，不替代真实摄像头。"""
    out = face.copy()
    h, w = out.shape[:2]
    t = idx / max(total - 1, 1)
    if action == "blink" and 0.35 <= t <= 0.65:
        y1, y2 = int(h * 0.28), int(h * 0.43)
        cv2.rectangle(out, (int(w * 0.20), y1), (int(w * 0.45), y2), (18, 18, 18), -1)
        cv2.rectangle(out, (int(w * 0.55), y1), (int(w * 0.80), y2), (18, 18, 18), -1)
    elif action == "open_mouth":
        radius_y = max(3, int(h * (0.02 + 0.075 * t)))
        cv2.ellipse(out, (w // 2, int(h * 0.72)), (int(w * 0.16), radius_y), 0, 0, 360, (12, 12, 12), -1)
    elif action in {"turn_left", "turn_right"}:
        factor = (t - 0.5) * (1 if action == "turn_right" else -1)
        grad = np.linspace(-1, 1, w, dtype=np.float32)[None, :, None]
        shade = 1.0 + grad * factor * 0.42
        out = np.clip(out.astype(np.float32) * shade, 0, 255).astype(np.uint8)
    return out


def action_values(action: str) -> list[dict]:
    mapping = {
        "move_left": [{"x": v} for v in [60, 35, 10, -20, -55, -90]],
        "move_right": [{"x": v} for v in [-90, -55, -20, 10, 35, 60]],
        "move_closer": [{"scale": v} for v in [1.00, 1.05, 1.10, 1.16, 1.22, 1.28]],
        "move_away": [{"scale": v} for v in [1.28, 1.22, 1.16, 1.10, 1.05, 1.00]],
        "nod": [{"y": v} for v in [-60, -25, 15, 55, 20, -20]],
        "blink": [{} for _ in range(6)],
        "open_mouth": [{} for _ in range(6)],
        "turn_left": [{} for _ in range(6)],
        "turn_right": [{} for _ in range(6)],
    }
    return mapping[action]


def make_stage_frames(face: np.ndarray, stage: int, action: str, step_ms: int = 520) -> list[dict]:
    frames = []
    values = action_values(action)
    for idx, kwargs in enumerate(values):
        action_face = _simulate_face_action(face, action, idx, len(values))
        frames.append({
            "stage": stage,
            "action": action,
            "stage_elapsed_ms": idx * step_ms,
            "image": encode_jpeg(make_canvas(action_face, **kwargs)),
        })
    return frames


def make_flash_stage_frames(face: np.ndarray, step: dict, step_ms: int = 520) -> list[dict]:
    seq = step.get("flash_sequence") or [
        {"name": "amber", "rgb": [255, 186, 36]},
        {"name": "cyan", "rgb": [0, 210, 255]},
        {"name": "red", "rgb": [255, 78, 78]},
        {"name": "green", "rgb": [46, 229, 157]},
    ]
    frames = []
    for idx in range(8):
        item = seq[idx % len(seq)]
        rgb = item.get("rgb", item)
        tint_bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.float32)
        lit = np.clip(face.astype(np.float32) * 0.68 + tint_bgr * 0.32, 0, 255).astype(np.uint8)
        frames.append({
            "stage": step["stage"],
            "action": "flash_response",
            "stage_elapsed_ms": idx * step_ms,
            "flash_index": idx % len(seq),
            "flash_rgb": rgb,
            "image": encode_jpeg(make_canvas(lit)),
        })
    return frames


def main() -> None:
    face = load_seed_face()
    actions = ["move_left", "blink", "open_mouth"]
    frames = []
    for stage, action in enumerate(actions, start=1):
        frames.extend(make_stage_frames(face, stage, action))
    live = analyze_liveness(frames, actions)
    assert live["pass"], live

    static_img = encode_jpeg(make_canvas(face))
    static_frames = [
        {"stage": stage, "action": action, "stage_elapsed_ms": idx * 520, "image": static_img}
        for stage, action in enumerate(actions, start=1)
        for idx in range(6)
    ]
    static_live = analyze_liveness(static_frames, actions)
    assert not static_live["pass"], static_live

    slow_frames = []
    for stage, action in enumerate(actions, start=1):
        slow_frames.extend(make_stage_frames(face, stage, action, step_ms=1800 if stage == 1 else 520))
    slow_live = analyze_liveness(slow_frames, actions)
    assert not slow_live["pass"] and "5 秒" in slow_live["reason"], slow_live

    app = create_app()
    client = app.test_client()
    login = client.post("/api/login", json={"username": "teacher", "password": "teacher123"})
    assert login.status_code == 200, login.get_data(as_text=True)
    challenge = client.get("/api/attendance/challenge").get_json()["challenge"]
    endpoint_results = []
    for step in challenge["steps"]:
        if step["action"] == "flash_response":
            stage_frames = make_flash_stage_frames(face, step)
        else:
            stage_frames = make_stage_frames(face, step["stage"], step["action"])
        r = client.post("/api/attendance/liveness-stage", json={
            "challenge_id": challenge["id"],
            "stage": step["stage"],
            "elapsed_ms": 3200,
            "frames": stage_frames,
        })
        data = r.get_json()
        assert r.status_code == 200 and data["stage_pass"], data
        endpoint_results.append({"stage": step["stage"], "action": step["action"], "reason": data["reason"]})

    print({
        "ok": True,
        "actions_tested": actions,
        "normal_pass": live["pass"],
        "static_attack_pass": static_live["pass"],
        "slow_attack_pass": slow_live["pass"],
        "stage_endpoint_results": endpoint_results,
    })


if __name__ == "__main__":
    main()
