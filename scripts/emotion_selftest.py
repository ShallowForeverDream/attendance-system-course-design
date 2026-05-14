from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.vision import analyze_emotion, crop_face, detect_faces, emotion_diagnostics, read_image_path

FACE_DATA = Path(r'C:\大学\大三\大三下\内容安全实践\实验六-课程设计\face_data')


def main() -> None:
    files = [p for p in FACE_DATA.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
    counter = Counter()
    examples = []
    for p in files[:60]:
        try:
            img = read_image_path(p)
            faces = detect_faces(img)
            if not faces:
                counter['no_face'] += 1
                continue
            emo = analyze_emotion(crop_face(img, faces[0]))
            counter[emo['emotion']] += 1
            if len(examples) < 12:
                examples.append({'file': p.name, **emo})
        except Exception as exc:
            counter['error'] += 1
            if len(examples) < 12:
                examples.append({'file': p.name, 'error': str(exc)})
    result = {'sample_size': min(60, len(files)), 'diagnostics': emotion_diagnostics(), 'distribution': dict(counter), 'examples': examples}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # 至少出现 2 类有效情绪，证明不再“全部一种情绪”。
    valid = [k for k in counter if k not in {'no_face', 'error', 'unknown'}]
    if len(valid) < 2:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
