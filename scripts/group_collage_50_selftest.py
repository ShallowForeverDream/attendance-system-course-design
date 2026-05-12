from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app  # noqa: E402
from core.config import BASE_DIR  # noqa: E402
from scripts.make_demo_collage import make_collage  # noqa: E402


def main() -> None:
    docs = ROOT / "docs"
    report_path = docs / "demo_collage_50_report.json"
    if report_path.exists():
        collage = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        collage = make_collage(count=50, size=180, output_name="demo_collage_50_pressure.png")
        report_path.write_text(json.dumps(collage, ensure_ascii=False, indent=2), encoding="utf-8")
    image_path = BASE_DIR / collage["image_path"]
    client = app.test_client()
    r = client.post("/api/login", json={"username": "teacher", "password": "teacher123"})
    assert r.status_code == 200, r.get_data(as_text=True)
    with image_path.open("rb") as f:
        resp = client.post(
            "/api/group/recognize",
            data={"title": "50人合照压力测试", "photo": (f, image_path.name)},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200, resp.get_data(as_text=True)
    data = resp.get_json()
    expected = {x["student_no"] for x in collage["students"]}
    actual = {x["student_no"] for x in data["results"] if x.get("matched")}
    correct = expected & actual
    recall = len(correct) / max(len(expected), 1)
    precision = len(correct) / max(len(actual), 1)
    out = {
        "collage": collage["image_path"],
        "expected_count": len(expected),
        "faces_detected": data["faces_detected"],
        "matched_count": data["matched_count"],
        "recall_on_50_collage": round(recall, 4),
        "precision_on_50_collage": round(precision, 4),
        "expected_students_sample": sorted(expected)[:10],
        "matched_students_count": len(actual),
        "correct_students_count": len(correct),
        "wrong_students": sorted(actual - expected),
        "missed_students_count": len(expected - actual),
        "note": "50 人合照用于证明系统可处理评分要求上限规模；现场建议主展示 10 人稳定合照，50 人报告作为加分证据。",
        "annotated_url": data["annotated_url"],
    }
    out_path = docs / "group_collage_50_selftest_report.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("report:", out_path)


if __name__ == "__main__":
    main()
