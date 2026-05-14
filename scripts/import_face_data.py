from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from core.config import BASE_DIR, FACES_DIR  # noqa: E402
from core.db import db, init_db, now_iso, upsert_metric  # noqa: E402
from core.face_import import IMAGE_EXTS, parse_student_image_filename  # noqa: E402
from core.vision import crop_face, embedding_from_image, read_image_path, save_image  # noqa: E402


def import_face_data(source: Path, reset_demo: bool = False, limit: int | None = None) -> dict:
    init_db(seed=True)
    images = [p for p in sorted(source.rglob("*")) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if limit:
        images = images[:limit]
    summary = {
        "source": str(source),
        "total_images": len(images),
        "students_added": 0,
        "students_updated": 0,
        "samples_added": 0,
        "failed": [],
        "quality": [],
    }
    ts = now_iso()
    with db() as conn:
        if reset_demo:
            conn.execute("DELETE FROM face_samples")
            conn.execute("DELETE FROM attendance_records")
            conn.execute("DELETE FROM emotion_records")
            conn.execute("DELETE FROM activity_participants")
            conn.execute("DELETE FROM activities")
            conn.execute("DELETE FROM students")
            conn.execute("DELETE FROM users WHERE role='student'")
        known_sources = {
            row["image_path"]
            for row in conn.execute(
                "SELECT image_path FROM face_samples WHERE image_path LIKE 'storage/faces/%'"
            ).fetchall()
        }
        for img_path in images:
            meta = parse_student_image_filename(img_path)
            student_no = meta["student_no"]
            name = meta["name"]
            try:
                old = conn.execute("SELECT id FROM students WHERE student_no=?", (student_no,)).fetchone()
                if old:
                    student_id = old["id"]
                    conn.execute(
                        "UPDATE students SET name=?,class_name=?,gender=?,status='active',updated_at=? WHERE id=?",
                        (name, meta["class_name"], meta["gender"], ts, student_id),
                    )
                    summary["students_updated"] += 1
                else:
                    cur = conn.execute(
                        """INSERT INTO students(student_no,name,class_name,gender,status,created_at,updated_at)
                           VALUES(?,?,?,?,?,?,?)""",
                        (student_no, name, meta["class_name"], meta["gender"], "active", ts, ts),
                    )
                    student_id = cur.lastrowid
                    summary["students_added"] += 1
                img = read_image_path(img_path)
                emb, box, quality = embedding_from_image(img)
                face_crop = crop_face(img, box, pad=0.22)
                saved = save_image(face_crop, FACES_DIR / str(student_id), prefix=f"{student_no}_{img_path.stem[:24]}")
                rel_saved = str(saved.relative_to(BASE_DIR))
                # 以新保存路径为准，避免 Windows 中文/长文件名前缀导致的 LIKE 误判。
                if rel_saved in known_sources:
                    continue
                conn.execute(
                    "INSERT INTO face_samples(student_id,image_path,embedding,quality,created_at) VALUES(?,?,?,?,?)",
                    (student_id, rel_saved, json.dumps(emb), quality, ts),
                )
                summary["samples_added"] += 1
                summary["quality"].append(round(float(quality), 4))
            except Exception as exc:
                # 失败图片仍可在报告中说明：常见原因是非正脸、多人、遮挡或分辨率太低。
                summary["failed"].append({"file": str(img_path), "student_no": student_no, "name": name, "error": str(exc)})
        ok = summary["samples_added"]
        total = summary["total_images"]
        avg_q = sum(summary["quality"]) / max(len(summary["quality"]), 1)
        upsert_metric(conn, "face_data_import_success_rate", ok / max(total, 1), total, summary)
        upsert_metric(conn, "face_data_average_quality", avg_q, ok, {"quality": summary["quality"]})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="导入老师提供的 face_data 班级照片到系统人脸库")
    parser.add_argument("--source", default=str(PROJECT_ROOT / "face_data"), help="face_data 目录")
    parser.add_argument("--reset-demo", action="store_true", help="清空现有学生/样本/记录后重新导入")
    parser.add_argument("--limit", type=int, default=None, help="只导入前 N 张，便于调试")
    parser.add_argument("--out", default=str(ROOT / "docs" / "face_data_import_report.json"), help="导入报告路径")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise SystemExit(f"face_data 目录不存在：{source}")
    summary = import_face_data(source, reset_demo=args.reset_demo, limit=args.limit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "total_images": summary["total_images"],
        "students_added": summary["students_added"],
        "students_updated": summary["students_updated"],
        "samples_added": summary["samples_added"],
        "failed": len(summary["failed"]),
        "report": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
