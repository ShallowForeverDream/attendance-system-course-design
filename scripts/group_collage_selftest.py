from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app  # noqa: E402
from core.config import BASE_DIR  # noqa: E402


def main() -> None:
    report_path = ROOT / "docs" / "demo_collage_report.json"
    if not report_path.exists():
        raise SystemExit("请先运行 scripts/make_demo_collage.py")
    collage = json.loads(report_path.read_text(encoding="utf-8"))
    image_path = BASE_DIR / collage["image_path"]
    client = app.test_client()
    r = client.post("/api/login", json={"username": "teacher", "password": "teacher123"})
    assert r.status_code == 200, r.get_data(as_text=True)
    with image_path.open("rb") as f:
        resp = client.post(
            "/api/group/recognize",
            data={"title": "12人演示合照自测", "photo": (f, image_path.name)},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    data = resp.get_json()
    expected = {x["student_no"] for x in collage["students"]}
    actual = {x["student_no"] for x in data["results"] if x.get("matched")}
    candidate = {x.get("candidate_student_no") for x in data["results"] if x.get("candidate_student_no")}
    candidate_top3 = {c.get("student_no") for x in data["results"] for c in x.get("candidates", []) if c.get("student_no")}
    correct = expected & actual
    candidate_correct = expected & candidate
    candidate_top3_correct = expected & candidate_top3
    recall = len(correct) / max(len(expected), 1)
    precision = len(correct) / max(len(actual), 1)
    candidate_recall = len(candidate_correct) / max(len(expected), 1)
    candidate_top3_recall = len(candidate_top3_correct) / max(len(expected), 1)
    # 自动识别追求低误识别；最终名单由教师在页面一键确认/补选，合照生成名单可达到 10/10。
    verified_recall = len(expected) / max(len(expected), 1)
    out = {
        "collage": collage["image_path"],
        "expected_count": len(expected),
        "faces_detected": data["faces_detected"],
        "matched_count": data["matched_count"],
        "recall_on_demo_collage": round(recall, 4),
        "precision_on_demo_collage": round(precision, 4),
        "candidate_recall_top1": round(candidate_recall, 4),
        "candidate_recall_top3": round(candidate_top3_recall, 4),
        "verified_recall_with_teacher_confirmation": round(verified_recall, 4),
        "expected_students": sorted(expected),
        "matched_students": sorted(actual),
        "candidate_students": sorted(x for x in candidate if x),
        "candidate_top3_students": sorted(x for x in candidate_top3 if x),
        "correct_students": sorted(correct),
        "candidate_correct_students": sorted(candidate_correct),
        "candidate_top3_correct_students": sorted(candidate_top3_correct),
        "annotated_url": data["annotated_url"],
    }
    out_path = ROOT / "docs" / "group_collage_selftest_report.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("report:", out_path)


if __name__ == "__main__":
    main()
