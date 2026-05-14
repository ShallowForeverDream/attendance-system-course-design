from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from app import create_app  # noqa: E402
from core.vision import analyze_liveness, read_image_path  # noqa: E402


def encode_jpeg(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        raise RuntimeError("图片编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def load_seed_face() -> np.ndarray:
    faces = list((ROOT / "storage" / "faces").glob("**/*.jpg"))
    if not faces:
        raise SystemExit("缺少 storage/faces 样本，请先运行 python scripts/prepare_demo.py --source ..\\face_data")
    return read_image_path(faces[0])


def make_canvas(face: np.ndarray, *, x: int = 0, y: int = 0, scale: float = 1.0) -> np.ndarray:
    h, w = face.shape[:2]
    canvas_h, canvas_w = max(360, h * 2), max(480, w * 3)
    nw, nh = max(20, int(w * scale)), max(20, int(h * scale))
    resized = cv2.resize(face, (nw, nh))
    canvas = np.full((canvas_h, canvas_w, 3), 240, dtype=np.uint8)
    px = canvas_w // 2 - nw // 2 + int(x)
    py = canvas_h // 2 - nh // 2 + int(y)
    px = max(0, min(canvas_w - nw, px))
    py = max(0, min(canvas_h - nh, py))
    canvas[py:py + nh, px:px + nw] = resized
    return canvas


def action_values(action: str) -> list[dict]:
    mapping = {
        "move_left": [{"x": v} for v in [60, 35, 10, -20, -55, -90]],
        "move_right": [{"x": v} for v in [-90, -55, -20, 10, 35, 60]],
        "move_up": [{"y": v} for v in [50, 30, 10, -10, -30, -55]],
        "move_down": [{"y": v} for v in [-55, -30, -10, 10, 30, 50]],
        "move_closer": [{"scale": v} for v in [1.00, 1.05, 1.10, 1.16, 1.22, 1.28]],
        "move_away": [{"scale": v} for v in [1.28, 1.22, 1.16, 1.10, 1.05, 1.00]],
        "shake_left_right": [{"x": v} for v in [-85, -35, 20, 75, 25, -25]],
        "nod_up_down": [{"y": v} for v in [-60, -25, 15, 55, 20, -20]],
        "zoom_in_out": [{"scale": v} for v in [1.00, 1.10, 1.22, 1.14, 1.04, 0.96]],
    }
    return mapping[action]


def make_stage_frames(face: np.ndarray, stage: int, action: str, step_ms: int = 520) -> list[dict]:
    frames = []
    for idx, kwargs in enumerate(action_values(action)):
        frames.append({
            "stage": stage,
            "action": action,
            "stage_elapsed_ms": idx * step_ms,
            "image": encode_jpeg(make_canvas(face, **kwargs)),
        })
    return frames


def main() -> None:
    face = load_seed_face()
    actions = ["move_left", "move_closer", "shake_left_right"]
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
