from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app  # noqa: E402
from core.config import BASE_DIR  # noqa: E402


def assert_ok(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def main() -> None:
    client = app.test_client()
    r = client.post("/api/login", json={"username": "teacher", "password": "teacher123"})
    assert_ok(r.status_code == 200, "教师登录失败")
    summary = client.get("/api/summary").get_json()["counts"]
    checklist = client.get("/api/demo/checklist").get_json()
    assert_ok(summary["students"] >= 100, "学生数量不足，需导入 face_data")
    assert_ok(summary["face_samples"] >= 100, "人脸样本不足")

    collage_report = ROOT / "docs" / "demo_collage_report.json"
    assert_ok(collage_report.exists(), "缺少 demo_collage_report.json")
    collage = json.loads(collage_report.read_text(encoding="utf-8"))
    img_path = BASE_DIR / collage["image_path"]
    assert_ok(img_path.exists(), f"演示合照不存在：{img_path}")
    with img_path.open("rb") as f:
        resp = client.post(
            "/api/group/recognize",
            data={"title": "最终验收10人合照", "photo": (f, img_path.name)},
            content_type="multipart/form-data",
        )
    assert_ok(resp.status_code == 200, resp.get_data(as_text=True))
    group = resp.get_json()
    expected = {x["student_no"] for x in collage["students"]}
    matched = {x["student_no"] for x in group["results"] if x.get("matched")}
    assert_ok(group["faces_detected"] >= 10, "合照检测人脸数不足 10")
    assert_ok(len(expected & matched) / len(expected) >= 0.85, f"合照自动召回不足：{expected & matched}")
    assert_ok(len(matched - expected) == 0, f"合照有误识别：{matched - expected}")

    # 教师确认最终名单，验证活动频次闭环。
    id_by_no = {x["student_no"]: x["student_id"] for x in group["results"] if x.get("candidate_student_no") for _ in [None]}
    # 如果自动结果里没有某个学生 id，就从 /api/students 查表补齐。
    students = client.get("/api/students").get_json()["students"]
    id_by_no.update({s["student_no"]: s["id"] for s in students})
    ids = [id_by_no[no] for no in sorted(expected) if no in id_by_no]
    confirm = client.post(f"/api/group/{group['activity_id']}/participants", json={"student_ids": ids})
    assert_ok(confirm.status_code == 200, confirm.get_data(as_text=True))
    assert_ok(confirm.get_json()["count"] == len(expected), "教师确认名单人数不等于 10")

    collage_50_report = ROOT / "docs" / "demo_collage_50_report.json"
    if collage_50_report.exists():
        collage_50 = json.loads(collage_50_report.read_text(encoding="utf-8"))
        img_50_path = BASE_DIR / collage_50["image_path"]
        assert_ok(img_50_path.exists(), f"50人演示合照不存在：{img_50_path}")
        assert_ok(len(collage_50.get("students", [])) >= 50, "50人合照名单不足 50")
    else:
        collage_50 = {}

    static_report = ROOT / "docs" / "group_collage_selftest_report.json"
    if static_report.exists():
        selftest = json.loads(static_report.read_text(encoding="utf-8"))
    else:
        selftest = {}
    out = {
        "summary": summary,
        "checklist_items": len(checklist["items"]),
        "group_faces_detected": group["faces_detected"],
        "group_matched_count": group["matched_count"],
        "group_expected_count": len(expected),
        "group_recall_runtime": round(len(expected & matched) / len(expected), 4),
        "group_precision_runtime": round(len(expected & matched) / max(len(matched), 1), 4),
        "confirmed_participants": confirm.get_json()["count"],
        "collage_50_expected_count": len(collage_50.get("students", [])) if collage_50 else 0,
        "selftest_report": selftest,
    }
    out_path = ROOT / "docs" / "final_acceptance_report.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("FINAL ACCEPTANCE OK")


if __name__ == "__main__":
    main()
