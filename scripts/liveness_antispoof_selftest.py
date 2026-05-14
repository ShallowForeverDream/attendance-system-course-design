from __future__ import annotations

import base64
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from core.config import BASE_DIR
from core.db import db, init_db
from core.vision import analyze_liveness, read_image_path


def _b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode('.jpg', img)
    if not ok:
        raise RuntimeError('encode failed')
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode()


def _best_sample_image() -> tuple[np.ndarray, dict]:
    init_db(seed=True)
    with db() as conn:
        row = conn.execute(
            """SELECT f.image_path,s.student_no,s.name
               FROM face_samples f JOIN students s ON s.id=f.student_id
               ORDER BY f.quality DESC LIMIT 1"""
        ).fetchone()
    if not row:
        raise RuntimeError('人脸库为空，请先运行 scripts/import_face_data.py 导入 face_data')
    return read_image_path(BASE_DIR / row['image_path']), row


def moving_photo_attack(img: np.ndarray) -> list[dict]:
    base_h, base_w = 720, 960
    photo = cv2.resize(img, (260, 320), interpolation=cv2.INTER_AREA)
    positions = {
        1: [(370, 200), (340, 200), (300, 200), (260, 200), (230, 200), (210, 200)],
        2: [(350, 210, 0.92), (340, 200, 1.00), (330, 190, 1.08), (320, 180, 1.16), (310, 170, 1.24), (300, 160, 1.32)],
        3: [(340, 200), (355, 200), (370, 200), (385, 200), (400, 200), (415, 200)],
    }
    frames = []
    for stage in range(1, 4):
        for idx, pos in enumerate(positions[stage]):
            if len(pos) == 3:
                x, y, scale = pos
                patch = cv2.resize(photo, (int(photo.shape[1] * scale), int(photo.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            else:
                x, y = pos
                patch = photo
            canvas = np.full((base_h, base_w, 3), 32, dtype=np.uint8)
            pad = 18
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(base_w, x + patch.shape[1] + pad), min(base_h, y + patch.shape[0] + pad)
            canvas[y1:y2, x1:x2] = (230, 230, 230)
            canvas[y:y + patch.shape[0], x:x + patch.shape[1]] = patch
            frames.append({'stage': stage, 'stage_elapsed_ms': idx * 650, 'image': _b64(canvas)})
    return frames


def prerecorded_video_attack(img: np.ndarray, steps: list[dict]) -> list[dict]:
    base_h, base_w = 720, 960
    face = cv2.resize(img, (280, 360), interpolation=cv2.INTER_AREA)
    frames = []
    for stage in range(1, 4):
        for idx in range(6):
            canvas = np.full((base_h, base_w, 3), 70, dtype=np.uint8)
            if stage == 1:
                x, y = 390 - idx * 26, 190
                scale = 1.0
            elif stage == 2:
                x, y = 340 - idx * 7, 190 - idx * 5
                scale = 0.92 + idx * 0.07
            else:
                x, y = 330 + idx * 16, 190
                scale = 1.0
            patch = cv2.resize(face, (int(face.shape[1] * scale), int(face.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            x = max(0, min(base_w - patch.shape[1], int(x)))
            y = max(0, min(base_h - patch.shape[0], int(y)))
            canvas[y:y + patch.shape[0], x:x + patch.shape[1]] = patch
            flash_idx = idx % len(steps[stage - 1]['flash_sequence'])
            frames.append({
                'stage': stage,
                'stage_elapsed_ms': idx * 650,
                'flash_index': flash_idx,
                'flash_rgb': steps[stage - 1]['flash_sequence'][flash_idx]['rgb'],
                'image': _b64(canvas),
            })
    return frames


def main() -> None:
    img, row = _best_sample_image()
    actions = ['move_left', 'open_mouth', 'turn_right']
    live_photo = analyze_liveness(moving_photo_attack(img), actions)
    steps = [
        {'stage': 1, 'action': 'move_left', 'flash_sequence': [{'name': 'amber', 'rgb': [255, 186, 36]}, {'name': 'cyan', 'rgb': [0, 210, 255]}, {'name': 'red', 'rgb': [255, 78, 78]}, {'name': 'green', 'rgb': [46, 229, 157]}]},
        {'stage': 2, 'action': 'open_mouth', 'flash_sequence': [{'name': 'cyan', 'rgb': [0, 210, 255]}, {'name': 'white', 'rgb': [245, 248, 255]}, {'name': 'amber', 'rgb': [255, 186, 36]}, {'name': 'red', 'rgb': [255, 78, 78]}]},
        {'stage': 3, 'action': 'turn_right', 'flash_sequence': [{'name': 'green', 'rgb': [46, 229, 157]}, {'name': 'red', 'rgb': [255, 78, 78]}, {'name': 'cyan', 'rgb': [0, 210, 255]}, {'name': 'white', 'rgb': [245, 248, 255]}]},
    ]
    live_video = analyze_liveness(prerecorded_video_attack(img, steps), actions, challenge_steps=steps)
    result = {
        'sample': {'student_no': row['student_no'], 'name': row['name']},
        'moving_photo_passed': live_photo['pass'],
        'moving_photo_reason': live_photo['reason'],
        'moving_photo': live_photo,
        'prerecorded_video_passed': live_video['pass'],
        'prerecorded_video_reason': live_video['reason'],
        'prerecorded_video': live_video,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if live_photo['pass'] or live_video['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
