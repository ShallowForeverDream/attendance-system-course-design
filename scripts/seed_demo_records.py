from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.db import db, init_db, now_iso, log_action, upsert_metric  # noqa: E402


def seed_demo_records() -> dict:
    init_db(seed=True)
    with db() as conn:
        # 选取质量最高的已入库学生，生成一条成功考勤样例，保证记录查询/Excel/情绪统计可现场展示。
        student = conn.execute(
            """SELECT s.id,s.student_no,s.name,f.quality,f.image_path
               FROM face_samples f JOIN students s ON s.id=f.student_id
               WHERE s.status='active' ORDER BY f.quality DESC LIMIT 1"""
        ).fetchone()
        if not student:
            return {"ok": False, "error": "没有可用人脸样本"}
        exists = conn.execute("SELECT id FROM attendance_records WHERE source='demo_seed' LIMIT 1").fetchone()
        if not exists:
            conn.execute(
                """INSERT INTO attendance_records(student_id,student_no,name,status,liveness_pass,liveness_score,face_score,emotion,captured_at,source,note)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (student["id"], student["student_no"], student["name"], "success", 1, 0.936, 0.982, "neutral", now_iso(), "demo_seed", "演示预置成功考勤；现场仍可用摄像头重新打卡"),
            )
            conn.execute(
                """INSERT INTO emotion_records(student_id,student_no,name,emotion,confidence,scene,image_path,captured_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (student["id"], student["student_no"], student["name"], "neutral", 0.91, "attendance", student["image_path"], now_iso()),
            )
        fail_exists = conn.execute("SELECT id FROM attendance_records WHERE source='demo_attack' LIMIT 1").fetchone()
        if not fail_exists:
            conn.execute(
                """INSERT INTO attendance_records(student_id,student_no,name,status,liveness_pass,liveness_score,face_score,emotion,captured_at,source,note)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (None, "", "", "failed", 0, 0.214, 0.0, "unknown", now_iso(), "demo_attack", "静态照片/重复帧攻击被活体检测拒绝"),
            )
        log_action(conn, None, "seed_demo_records", {"student_no": student["student_no"]})
        upsert_metric(conn, "demo_seed_attendance_records", 2, 2, {"success_student": student["student_no"]})
    return {"ok": True, "success_student": {"student_no": student["student_no"], "name": student["name"]}, "records_seeded": 2}


def main() -> None:
    report = seed_demo_records()
    out = ROOT / "docs" / "demo_seed_records_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("report:", out)


if __name__ == "__main__":
    main()
