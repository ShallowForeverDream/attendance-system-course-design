from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from core.vision import crop_face, detect_faces, face_embedding, read_image_path, recognize

FACE_DATA = Path(r'C:\大学\大三\大三下\内容安全实践\实验六-课程设计\face_data')


def augment_face(face: np.ndarray) -> list[np.ndarray]:
    variants = [face]
    variants.append(cv2.convertScaleAbs(face, alpha=1.08, beta=12))
    variants.append(cv2.convertScaleAbs(face, alpha=0.92, beta=-8))
    variants.append(cv2.GaussianBlur(face, (3, 3), 0))
    h, w = face.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 4, 1.0)
    variants.append(cv2.warpAffine(face, m, (w, h), borderMode=cv2.BORDER_REFLECT))
    return variants


def main() -> None:
    paths = [p for p in FACE_DATA.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
    paths = paths[:8]
    samples = []
    queries = []
    sid = 1
    for p in paths:
        img = read_image_path(p)
        faces = detect_faces(img)
        if not faces:
            continue
        face = crop_face(img, faces[0])
        student_no, name = p.stem.split('-')[:2]
        variants = augment_face(face)
        for i, v in enumerate(variants[:4], start=1):
            samples.append({
                'id': len(samples) + 1,
                'student_id': sid,
                'student_no': student_no,
                'name': name,
                'class_name': 'selftest',
                'quality': 0.62 + i * 0.06,
                'embedding': json.dumps(face_embedding(v)),
            })
        queries.append({'student_id': sid, 'student_no': student_no, 'name': name, 'embedding': face_embedding(variants[-1])})
        sid += 1
    ok = 0
    details = []
    for q in queries:
        match = recognize(q['embedding'], samples, threshold=0.70)
        got = match.get('student') or {}
        success = match.get('matched') and got.get('student_no') == q['student_no']
        ok += int(bool(success))
        details.append({
            'query': q['student_no'],
            'matched': bool(success),
            'got': got.get('student_no'),
            'score': round(match.get('score', 0), 4),
            'best_sample_score': round(match.get('best_sample_score', 0), 4),
            'centroid_score': round(match.get('centroid_score', 0), 4),
            'sample_count': match.get('sample_count'),
            'support_count': match.get('support_count'),
        })
    result = {'queries': len(queries), 'matched': ok, 'accuracy': ok / max(len(queries), 1), 'details': details}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if ok < len(queries):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
