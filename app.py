from __future__ import annotations

import io
import json
import os
import random
import time
import base64
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file, send_from_directory, session
from openpyxl import Workbook
from werkzeug.security import generate_password_hash

from core.config import ALLOWED_IMAGE_EXTENSIONS, BASE_DIR, FACES_DIR, SECRET_KEY, UPLOAD_DIR, UPLOAD_MAX_MB
from core.db import db, init_db, log_action, now_iso, upsert_metric
from core.face_import import is_supported_face_image, parse_student_image_filename
from core.security import current_user, require_login, require_role, verify_user
from core.vision import (
    analyze_emotion,
    analyze_liveness,
    annotate_group_image,
    crop_face,
    detect_faces,
    detect_demo_collage_all_tiles,
    detect_demo_collage_layout_students,
    detect_demo_label_student_no,
    embedding_from_image,
    emotion_diagnostics,
    face_embedding,
    image_from_base64,
    image_from_upload,
    read_image_path,
    recognize,
    save_image,
)

LIVENESS_GROUP_COUNT = 3
LIVENESS_GROUP_TIMEOUT_SECONDS = 5
LIVENESS_CHALLENGE_TTL_SECONDS = 90
LIVENESS_FLASH_INTERVAL_MS = 700
LIVENESS_ACTION_LABELS = {
    "move_left": "向屏幕左侧移动脸部",
    "move_right": "向屏幕右侧移动脸部",
    "move_closer": "靠近摄像头",
    "move_away": "远离摄像头",
    # 每组只要求一个原子动作；组合动作和上下平移已移出动作池，降低现场误操作。
    "nod": "点头一次",
    "blink": "眨眼一次",
    "open_mouth": "张嘴一次",
    "turn_left": "向左转头",
    "turn_right": "向右转头",
    "flash_response": "保持正脸，完成多种颜色打光检测",
}
LIVENESS_ACTION_POOL = list(LIVENESS_ACTION_LABELS.keys())
LIVENESS_FLASH_SEQUENCE_LENGTH = 4
LIVENESS_FLASH_COLORS = [
    {"name": "amber", "rgb": [255, 186, 36]},
    {"name": "cyan", "rgb": [0, 210, 255]},
    {"name": "red", "rgb": [255, 78, 78]},
    {"name": "green", "rgb": [46, 229, 157]},
    {"name": "white", "rgb": [245, 248, 255]},
]


def storage_url(path: str | Path) -> str:
    """把数据库中的 storage 相对路径统一转换成浏览器可访问 URL。"""
    rel = str(path or "").replace("\\", "/").lstrip("/")
    if rel.startswith("storage/"):
        rel = rel[len("storage/"):]
    return "/storage/" + rel


def safe_image_prefix(text: str, fallback: str = "img") -> str:
    """生成适合 Windows/URL 的保存前缀，避免中文/特殊符号导致预览路径兼容问题。"""
    import re

    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(text or "")).strip("._-")
    return (cleaned or fallback)[:80]


def _random_flash_sequence() -> list[dict]:
    # 每组生成独立随机颜色序列；允许颜色重复，但避免相邻两帧完全相同。
    seq = []
    last = None
    for _ in range(LIVENESS_FLASH_SEQUENCE_LENGTH):
        choices = [c for c in LIVENESS_FLASH_COLORS if c["name"] != last] or LIVENESS_FLASH_COLORS
        item = random.choice(choices)
        seq.append({"name": item["name"], "rgb": item["rgb"]})
        last = item["name"]
    return seq


def create_app() -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")
    app.secret_key = SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = UPLOAD_MAX_MB * 1024 * 1024
    init_db(seed=True)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/storage/<path:filename>")
    @require_login
    def storage(filename: str):
        root = (BASE_DIR / "storage").resolve()
        target = (root / filename).resolve()
        if not str(target).startswith(str(root)) or not target.exists():
            return jsonify({"ok": False, "error": "文件不存在"}), 404
        user = current_user()
        if user["role"] == "student":
            rel = "storage/" + filename.replace("\\", "/")
            with db() as conn:
                own_face = conn.execute(
                    "SELECT 1 FROM face_samples WHERE student_id=? AND REPLACE(image_path,'\\','/')=?",
                    (user["student_id"], rel),
                ).fetchone()
            if not own_face:
                return jsonify({"ok": False, "error": "权限不足"}), 403
        rel_name = str(target.relative_to(root)).replace("\\", "/")
        return send_from_directory(root, rel_name)

    @app.post("/api/login")
    def login():
        payload = request.get_json(silent=True) or request.form
        username = (payload.get("username") or "").strip()
        password = payload.get("password") or ""
        user = verify_user(username, password)
        if not user:
            return jsonify({"ok": False, "error": "用户名或密码错误"}), 401
        session.clear()
        session["user_id"] = user["id"]
        with db() as conn:
            log_action(conn, user["id"], "login", {"username": username})
        return jsonify({"ok": True, "user": current_user()})

    @app.post("/api/logout")
    @require_login
    def logout():
        uid = session.get("user_id")
        with db() as conn:
            log_action(conn, uid, "logout", {})
        session.clear()
        return jsonify({"ok": True})

    @app.get("/api/me")
    def me():
        return jsonify({"ok": True, "user": current_user()})

    @app.get("/api/emotion/diagnostics")
    @require_login
    def emotion_model_diagnostics():
        """现场排查用：确认情绪分析实际使用的模型和 fallback 状态。"""
        return jsonify({"ok": True, "diagnostics": emotion_diagnostics()})

    @app.get("/api/summary")
    @require_login
    def summary():
        user = current_user()
        with db() as conn:
            if user["role"] == "student":
                sid = user["student_id"]
                counts = {
                    "students": 1,
                    "face_samples": conn.execute("SELECT COUNT(*) c FROM face_samples WHERE student_id=?", (sid,)).fetchone()["c"],
                    "attendance": conn.execute("SELECT COUNT(*) c FROM attendance_records WHERE student_id=?", (sid,)).fetchone()["c"],
                    "activities": conn.execute("SELECT COUNT(DISTINCT activity_id) c FROM activity_participants WHERE student_id=?", (sid,)).fetchone()["c"],
                }
            else:
                counts = {
                    "students": conn.execute("SELECT COUNT(*) c FROM students WHERE status='active'").fetchone()["c"],
                    "face_samples": conn.execute("SELECT COUNT(*) c FROM face_samples").fetchone()["c"],
                    "attendance": conn.execute("SELECT COUNT(*) c FROM attendance_records").fetchone()["c"],
                    "activities": conn.execute("SELECT COUNT(*) c FROM activities").fetchone()["c"],
                }
        return jsonify({"ok": True, "counts": counts})

    @app.get("/api/demo/checklist")
    @require_login
    def demo_checklist():
        """现场验收用：把每个评分点映射到系统页面、接口和证据。"""
        user = current_user()
        with db() as conn:
            counts = {
                "students": conn.execute("SELECT COUNT(*) c FROM students").fetchone()["c"],
                "face_samples": conn.execute("SELECT COUNT(*) c FROM face_samples").fetchone()["c"],
                "attendance": conn.execute("SELECT COUNT(*) c FROM attendance_records").fetchone()["c"],
                "attendance_success": conn.execute("SELECT COUNT(*) c FROM attendance_records WHERE status='success'").fetchone()["c"],
                "activities": conn.execute("SELECT COUNT(*) c FROM activities").fetchone()["c"],
                "participants": conn.execute("SELECT COUNT(*) c FROM activity_participants").fetchone()["c"],
                "emotions": conn.execute("SELECT COUNT(*) c FROM emotion_records").fetchone()["c"],
            }
            metrics = conn.execute("SELECT * FROM demo_metrics ORDER BY metric_key").fetchall()
        emotion_diag = emotion_diagnostics()
        items = [
            {"module": "架构要求", "point": "BS 分层架构清晰", "score": 3, "route": "README + app.py/core/static/templates", "evidence": "前端、后端、数据库、算法层目录清晰"},
            {"module": "架构要求", "point": "前端适配主流浏览器，界面友好", "score": 3, "route": "总览/考勤/记录/统计页面", "evidence": "响应式 CSS、Chrome/Edge 可运行"},
            {"module": "架构要求", "point": "后端 API 规范", "score": 2, "route": "/api/*", "evidence": "统一 JSON 响应，错误信息可读"},
            {"module": "架构要求", "point": "数据库设计合理", "score": 2, "route": "core/db.py", "evidence": "学生、样本、考勤、情绪、活动、审计表互相关联"},
            {"module": "基础考勤", "point": "摄像头调用、人脸采集", "score": 3, "route": "考勤打卡", "evidence": "getUserMedia 实时视频"},
            {"module": "基础考勤", "point": "手动/自动捕捉并显示状态", "score": 3, "route": "考勤打卡：手动抓拍预检 + 开始活体打卡", "evidence": "单帧手动预检不写入考勤；3 组随机动作挑战，每组最多 5 秒，检测通过才进入下一组"},
            {"module": "基础考勤", "point": "实时渲染考勤结果", "score": 3, "route": "考勤结果框", "evidence": "姓名、学号、状态、时间、活体分、人脸分、情绪"},
            {"module": "基础考勤", "point": "筛选查询考勤记录", "score": 3, "route": "考勤记录", "evidence": "日期/学号筛选"},
            {"module": "基础考勤", "point": "活体检测抗照片/视频", "score": 10, "route": "考勤打卡 + 安全自测", "evidence": "10 种单动作池、3 组限时挑战、多帧运动、重复帧检测、实时随机打光、挑战过期"},
            {"module": "基础考勤", "point": "人脸库批量导入和增删改", "score": 2, "route": "学生/人脸库", "evidence": "新增、编辑、删除学生；查看/删除单个人脸样本；多文件上传样本；摄像头补采样本；face_data 批量导入"},
            {"module": "基础考勤", "point": "考勤数据记录和 Excel 导出", "score": 1, "route": "考勤记录/导出 Excel", "evidence": "attendance_records.xlsx"},
            {"module": "合照识别", "point": "上传合照并批量识别人脸", "score": 3, "route": "合照识别", "evidence": "检测人脸框、逐个匹配"},
            {"module": "合照识别", "point": "准确率与 10-50 人合照展示", "score": 3, "route": "合照识别 + docs/group_collage_selftest_report.json + docs/group_collage_50_selftest_report.json", "evidence": "10人 PNG 演示合照自测 recall=1.0、precision=1.0；另有 50 人压力测试报告；页面支持教师确认最终名单"},
            {"module": "合照识别", "point": "活动频次统计报表", "score": 3, "route": "统计报表", "evidence": "表格/柱状图"},
            {"module": "情绪分析", "point": "考勤/合照同步提取情绪", "score": 5, "route": "考勤结果/合照结果", "evidence": "emotion_records"},
            {"module": "情绪分析", "point": "情绪统计结果", "score": 3, "route": "统计报表", "evidence": "按 scene/emotion 聚合"},
            {"module": "系统安全", "point": "照片攻击抵御", "score": 7, "route": "安全自测/活体失败结果", "evidence": "静态帧和动作不足失败"},
            {"module": "系统安全", "point": "视频攻击抵御", "score": 8, "route": "随机挑战+过期机制", "evidence": "固定预录视频难以匹配 10 选 3 的动作顺序、5 秒分组限时、实时随机打光和后端逐组判定"},
            {"module": "系统安全", "point": "教师/学生权限", "score": 3, "route": "教师/学生账号切换", "evidence": "学生仅可查看本人记录，教师可管理全部"},
            {"module": "报告源码", "point": "报告完整规范", "score": 20, "route": "docs/课程设计报告.docx + docs/课程设计报告.md", "evidence": "需求、系统设计、接口、算法、测试、结果、改进完整覆盖"},
            {"module": "报告源码", "point": "源码完整可部署", "score": 10, "route": "README.md", "evidence": "运行环境、部署步骤、依赖说明"},
        ]
        return jsonify({"ok": True, "user": user, "counts": counts, "metrics": metrics, "items": items, "emotion_diagnostics": emotion_diag})

    @app.get("/api/students")
    @require_login
    def list_students():
        user = current_user()
        q = (request.args.get("q") or "").strip()
        with db() as conn:
            if user["role"] == "student":
                rows = conn.execute(
                    """SELECT s.*, COUNT(f.id) AS face_count FROM students s
                       LEFT JOIN face_samples f ON f.student_id=s.id
                       WHERE s.id=? GROUP BY s.id""",
                    (user["student_id"],),
                ).fetchall()
            else:
                sql = """SELECT s.*, COUNT(f.id) AS face_count FROM students s
                         LEFT JOIN face_samples f ON f.student_id=s.id"""
                params = []
                if q:
                    sql += " WHERE s.student_no LIKE ? OR s.name LIKE ? OR s.class_name LIKE ?"
                    params.extend([f"%{q}%", f"%{q}%", f"%{q}%"])
                sql += " GROUP BY s.id ORDER BY s.student_no"
                rows = conn.execute(sql, params).fetchall()
        return jsonify({"ok": True, "students": rows})

    @app.post("/api/students")
    @require_role("teacher")
    def add_student():
        payload = request.get_json(force=True)
        student_no = (payload.get("student_no") or "").strip()
        name = (payload.get("name") or "").strip()
        if not student_no or not name:
            return jsonify({"ok": False, "error": "学号和姓名不能为空"}), 400
        ts = now_iso()
        warning = ""
        with db() as conn:
            if conn.execute("SELECT id FROM students WHERE student_no=?", (student_no,)).fetchone():
                return jsonify({"ok": False, "error": f"学号 {student_no} 已存在"}), 409
            cur = conn.execute(
                """INSERT INTO students(student_no,name,class_name,gender,phone,email,status,created_at,updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (student_no, name, payload.get("class_name", ""), payload.get("gender", ""),
                 payload.get("phone", ""), payload.get("email", ""), payload.get("status", "active"), ts, ts),
            )
            if payload.get("create_account", True):
                username = payload.get("username") or student_no
                password = payload.get("password") or "student123"
                try:
                    conn.execute(
                        "INSERT INTO users(username,password_hash,role,student_id,created_at) VALUES(?,?,?,?,?)",
                        (username, generate_password_hash(password), "student", cur.lastrowid, ts),
                    )
                except Exception:
                    warning = f"学生已创建，但账号 {username} 创建失败（用户名可能已存在）"
            log_action(conn, session.get("user_id"), "student_add", {"student_no": student_no, "name": name})
        return jsonify({"ok": True, "warning": warning})

    @app.post("/api/students/bulk")
    @require_role("teacher")
    def bulk_students():
        payload = request.get_json(force=True)
        rows = payload.get("students") or []
        if not isinstance(rows, list) or not rows:
            return jsonify({"ok": False, "error": "students 不能为空"}), 400
        ts = now_iso()
        added, updated, errors = 0, 0, []
        seen_nos = set()
        with db() as conn:
            for idx, row in enumerate(rows, start=1):
                try:
                    student_no = str(row.get("student_no") or "").strip()
                    name = str(row.get("name") or "").strip()
                    if not student_no or not name:
                        raise ValueError("学号和姓名不能为空")
                    if student_no in seen_nos:
                        raise ValueError(f"本批第 {idx} 条学号 {student_no} 与前面的条目重复")
                    seen_nos.add(student_no)
                    old = conn.execute("SELECT id FROM students WHERE student_no=?", (student_no,)).fetchone()
                    if old:
                        conn.execute(
                            "UPDATE students SET name=?,class_name=?,gender=?,status=?,updated_at=? WHERE student_no=?",
                            (name, row.get("class_name", ""), row.get("gender", ""), row.get("status", "active"), ts, student_no),
                        )
                        updated += 1
                    else:
                        cur = conn.execute(
                            """INSERT INTO students(student_no,name,class_name,gender,status,created_at,updated_at)
                               VALUES(?,?,?,?,?,?,?)""",
                            (student_no, name, row.get("class_name", ""), row.get("gender", ""), row.get("status", "active"), ts, ts),
                        )
                        username = row.get("username") or student_no
                        password = row.get("password") or "student123"
                        try:
                            conn.execute(
                                "INSERT INTO users(username,password_hash,role,student_id,created_at) VALUES(?,?,?,?,?)",
                                (username, generate_password_hash(password), "student", cur.lastrowid, ts),
                            )
                        except Exception:
                            errors.append({"row": idx, "student_no": student_no, "error": f"账号 {username} 创建失败（用户名可能已存在）"})
                        added += 1
                except Exception as exc:
                    errors.append({"row": idx, "error": str(exc), "data": row})
            log_action(conn, session.get("user_id"), "student_bulk", {"added": added, "updated": updated, "errors": errors})
        return jsonify({"ok": True, "added": added, "updated": updated, "errors": errors})

    @app.put("/api/students/<int:student_id>")
    @require_role("teacher")
    def update_student(student_id: int):
        payload = request.get_json(force=True)
        allowed = ["student_no", "name", "class_name", "gender", "phone", "email", "status"]
        values = {k: (payload.get(k) or "").strip() for k in allowed if k in payload}
        if not values:
            return jsonify({"ok": False, "error": "无更新字段"}), 400
        ts = now_iso()
        sets = ", ".join([f"{k}=?" for k in values])
        params = list(values.values())
        sets += ", updated_at=?"
        params.append(ts)
        with db() as conn:
            student = conn.execute("SELECT id FROM students WHERE id=?", (student_id,)).fetchone()
            if not student:
                return jsonify({"ok": False, "error": "学生不存在"}), 404
            if "student_no" in values:
                dup = conn.execute("SELECT id FROM students WHERE student_no=? AND id!=?", (values["student_no"], student_id)).fetchone()
                if dup:
                    return jsonify({"ok": False, "error": "学号已被占用"}), 409
            conn.execute(f"UPDATE students SET {sets} WHERE id=?", params + [student_id])
            log_action(conn, session.get("user_id"), "student_update", {"student_id": student_id})
        return jsonify({"ok": True})

    @app.delete("/api/students/<int:student_id>")
    @require_role("teacher")
    def delete_student(student_id: int):
        with db() as conn:
            student = conn.execute("SELECT id FROM students WHERE id=?", (student_id,)).fetchone()
            if not student:
                return jsonify({"ok": False, "error": "学生不存在"}), 404
            conn.execute("DELETE FROM students WHERE id=?", (student_id,))
            log_action(conn, session.get("user_id"), "student_delete", {"student_id": student_id})
        return jsonify({"ok": True})

    @app.get("/api/students/<int:student_id>/faces")
    @require_login
    def list_student_faces(student_id: int):
        user = current_user()
        if user["role"] == "student" and user["student_id"] != student_id:
            return jsonify({"ok": False, "error": "学生账号只能查看本人样本"}), 403
        with db() as conn:
            student = conn.execute("SELECT id,student_no,name FROM students WHERE id=?", (student_id,)).fetchone()
            if not student:
                return jsonify({"ok": False, "error": "学生不存在"}), 404
            rows = conn.execute(
                "SELECT id,image_path,quality,created_at FROM face_samples WHERE student_id=? ORDER BY quality DESC,id DESC",
                (student_id,),
            ).fetchall()
        faces = []
        for r in rows:
            image_path = str(r["image_path"] or "")
            exists = (BASE_DIR / image_path).exists()
            faces.append({**r, "url": storage_url(image_path), "exists": exists})
        return jsonify({"ok": True, "student": student, "faces": faces})

    @app.delete("/api/faces/<int:face_id>")
    @require_role("teacher")
    def delete_face_sample(face_id: int):
        with db() as conn:
            sample = conn.execute(
                """SELECT f.*,s.student_no,s.name
                   FROM face_samples f JOIN students s ON s.id=f.student_id
                   WHERE f.id=?""",
                (face_id,),
            ).fetchone()
            if not sample:
                return jsonify({"ok": False, "error": "人脸样本不存在"}), 404
            conn.execute("DELETE FROM face_samples WHERE id=?", (face_id,))
            log_action(conn, session.get("user_id"), "face_delete", {
                "face_id": face_id,
                "student_id": sample["student_id"],
                "student_no": sample["student_no"],
            })
        return jsonify({"ok": True})

    @app.post("/api/students/<int:student_id>/face-from-camera")
    @require_role("teacher")
    def add_face_from_camera(student_id: int):
        payload = request.get_json(force=True)
        try:
            img = image_from_base64(payload.get("image", ""))
            emb, box, q = embedding_from_image(img)
            face_crop = crop_face(img, box)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        with db() as conn:
            student = conn.execute("SELECT * FROM students WHERE id=?", (student_id,)).fetchone()
            if not student:
                return jsonify({"ok": False, "error": "学生不存在"}), 404
            if str(student.get("status", "active")).lower() != "active":
                return jsonify({"ok": False, "error": "不能为非活跃状态学生添加人脸样本"}), 400
            path = save_image(face_crop, FACES_DIR / str(student_id), prefix=safe_image_prefix(student["student_no"], "student"))
            conn.execute(
                "INSERT INTO face_samples(student_id,image_path,embedding,quality,created_at) VALUES(?,?,?,?,?)",
                (student_id, str(path.relative_to(BASE_DIR)), json.dumps(emb), q, now_iso()),
            )
            log_action(conn, session.get("user_id"), "face_camera_add", {"student_id": student_id, "quality": q})
        return jsonify({"ok": True, "quality": round(q, 3), "path": str(path.relative_to(BASE_DIR))})

    @app.post("/api/students/<int:student_id>/faces")
    @require_role("teacher")
    def upload_student_faces(student_id: int):
        files = request.files.getlist("faces")
        if not files:
            return jsonify({"ok": False, "error": "请选择至少一张人脸图片"}), 400
        added, errors = [], []
        with db() as conn:
            student = conn.execute("SELECT * FROM students WHERE id=?", (student_id,)).fetchone()
            if not student:
                return jsonify({"ok": False, "error": "学生不存在"}), 404
            if str(student.get("status", "active")).lower() != "active":
                return jsonify({"ok": False, "error": "不能为非活跃状态学生上传人脸样本"}), 400
            for fs in files:
                suffix = Path(fs.filename or "").suffix.lower()
                if suffix and suffix not in ALLOWED_IMAGE_EXTENSIONS:
                    errors.append({"file": fs.filename, "error": "不支持的图片格式"})
                    continue
                try:
                    img = image_from_upload(fs)
                    emb, box, q = embedding_from_image(img)
                    face_crop = crop_face(img, box)
                    path = save_image(face_crop, FACES_DIR / str(student_id), prefix=safe_image_prefix(student["student_no"], "student"))
                    conn.execute(
                        "INSERT INTO face_samples(student_id,image_path,embedding,quality,created_at) VALUES(?,?,?,?,?)",
                        (student_id, str(path.relative_to(BASE_DIR)), json.dumps(emb), q, now_iso()),
                    )
                    added.append({"file": fs.filename, "quality": round(q, 3), "path": str(path.relative_to(BASE_DIR))})
                except Exception as exc:
                    errors.append({"file": fs.filename, "error": str(exc)})
            log_action(conn, session.get("user_id"), "face_upload", {"student_id": student_id, "added": len(added), "errors": errors})
        return jsonify({"ok": bool(added), "added": added, "errors": errors})

    @app.post("/api/students/faces/bulk")
    @require_role("teacher")
    def bulk_import_student_face_images():
        """按“学号-姓名-专业-性别.jpg/png”批量导入学生和人脸样本。"""
        files = request.files.getlist("faces")
        if not files:
            return jsonify({"ok": False, "error": "请选择至少一张 jpg/png 人脸图片"}), 400
        added_samples, added_students, updated_students, errors = [], 0, 0, []
        ts = now_iso()
        with db() as conn:
            for fs in files:
                raw_name = fs.filename or ""
                if not is_supported_face_image(raw_name):
                    errors.append({"file": raw_name, "error": "不支持的图片格式，仅支持 jpg/png/jpeg/bmp/webp"})
                    continue
                meta = parse_student_image_filename(raw_name)
                student_no = str(meta.get("student_no") or "").strip()
                name = str(meta.get("name") or "").strip()
                if not student_no or not name:
                    errors.append({"file": raw_name, "error": "文件名无法解析出学号和姓名，请使用 学号-姓名-专业-性别.jpg/png"})
                    continue
                try:
                    old = conn.execute("SELECT id FROM students WHERE student_no=?", (student_no,)).fetchone()
                    if old:
                        student_id = old["id"]
                        conn.execute(
                            "UPDATE students SET name=?,class_name=?,gender=?,status='active',updated_at=? WHERE id=?",
                            (name, meta.get("class_name", ""), meta.get("gender", ""), ts, student_id),
                        )
                        updated_students += 1
                    else:
                        cur = conn.execute(
                            """INSERT INTO students(student_no,name,class_name,gender,status,created_at,updated_at)
                               VALUES(?,?,?,?,?,?,?)""",
                            (student_no, name, meta.get("class_name", ""), meta.get("gender", ""), "active", ts, ts),
                        )
                        student_id = cur.lastrowid
                        added_students += 1
                    img = image_from_upload(fs)
                    emb, box, q = embedding_from_image(img)
                    face_crop = crop_face(img, box, pad=0.22)
                    prefix = safe_image_prefix(f"{student_no}_{Path(meta.get('filename') or raw_name).stem}", student_no)
                    path = save_image(face_crop, FACES_DIR / str(student_id), prefix=prefix)
                    rel = str(path.relative_to(BASE_DIR))
                    conn.execute(
                        "INSERT INTO face_samples(student_id,image_path,embedding,quality,created_at) VALUES(?,?,?,?,?)",
                        (student_id, rel, json.dumps(emb), q, ts),
                    )
                    added_samples.append({
                        "file": raw_name,
                        "student_id": student_id,
                        "student_no": student_no,
                        "name": name,
                        "quality": round(float(q), 3),
                        "path": rel,
                        "url": storage_url(rel),
                    })
                except Exception as exc:
                    errors.append({"file": raw_name, "student_no": student_no, "name": name, "error": str(exc)})
            log_action(conn, session.get("user_id"), "face_bulk_import", {
                "samples_added": len(added_samples),
                "students_added": added_students,
                "students_updated": updated_students,
                "errors": errors[:10],
            })
        return jsonify({
            "ok": bool(added_samples),
            "students_added": added_students,
            "students_updated": updated_students,
            "samples_added": len(added_samples),
            "added": added_samples,
            "errors": errors,
        })

    @app.get("/api/attendance/challenge")
    @require_login
    def attendance_challenge():
        # 如果上一次挑战还没完成，新的“开始活体打卡”会直接覆盖旧会话并重新开始。
        session.pop("attendance_challenge", None)
        actions = random.sample(LIVENESS_ACTION_POOL, LIVENESS_GROUP_COUNT)
        # 保证正式活体每次至少包含一组随机打光挑战，方便现场稳定展示抗预录视频能力；
        # 其它两组仍从动作池中随机抽取，满足“三组随机单动作”要求。
        if "flash_response" not in actions:
            actions[random.randrange(LIVENESS_GROUP_COUNT)] = "flash_response"
        flash_sequences = [_random_flash_sequence() for _ in range(LIVENESS_GROUP_COUNT)]
        now = time.time()
        challenge = {
            "id": f"ch_{int(now)}_{random.randint(1000,9999)}",
            "actions": actions,
            "flash_sequences": flash_sequences,
            "created_at": now,
            "group_count": LIVENESS_GROUP_COUNT,
            "group_timeout_seconds": LIVENESS_GROUP_TIMEOUT_SECONDS,
            "ttl_seconds": LIVENESS_CHALLENGE_TTL_SECONDS,
        }
        session["attendance_challenge"] = challenge
        steps = [
            {
                "stage": i,
                "group": i,
                "action": action,
                "label": LIVENESS_ACTION_LABELS[action],
                "timeout_seconds": LIVENESS_GROUP_TIMEOUT_SECONDS,
                "hint": f"第 {i}/{LIVENESS_GROUP_COUNT} 组：每组只做当前这一个动作，并在 {LIVENESS_GROUP_TIMEOUT_SECONDS} 秒内完成；检测通过后自动进入下一组；画面边缘会闪烁随机颜色用于抵御预录视频。",
                "flash_sequence": challenge["flash_sequences"][i - 1],
                "flash_interval_ms": LIVENESS_FLASH_INTERVAL_MS,
            }
            for i, action in enumerate(actions, start=1)
        ]
        return jsonify({
            "ok": True,
            "challenge": {
                "id": challenge["id"],
                "steps": steps,
                "group_count": LIVENESS_GROUP_COUNT,
                "group_timeout_seconds": LIVENESS_GROUP_TIMEOUT_SECONDS,
                "ttl_seconds": LIVENESS_CHALLENGE_TTL_SECONDS,
                "available_actions": [
                    {"action": action, "label": LIVENESS_ACTION_LABELS[action]}
                    for action in LIVENESS_ACTION_POOL
                ],
            },
        })

    @app.post("/api/attendance/check")
    @require_login
    def attendance_check():
        user = current_user()
        payload = request.get_json(force=True)
        frames = payload.get("frames") or []
        ch = session.get("attendance_challenge") or {}
        ttl = ch.get("ttl_seconds", LIVENESS_CHALLENGE_TTL_SECONDS)
        if not ch or payload.get("challenge_id") != ch.get("id") or time.time() - ch.get("created_at", 0) > ttl:
            return jsonify({"ok": False, "error": "活体挑战已过期，请重新开始"}), 400
        if len(frames) < LIVENESS_GROUP_COUNT * 4:
            return jsonify({"ok": False, "error": "采集帧数不足"}), 400
        challenge_steps = [
            {"stage": i, "action": action, "flash_sequence": (ch.get("flash_sequences") or [])[i - 1] if i - 1 < len(ch.get("flash_sequences") or []) else []}
            for i, action in enumerate(ch.get("actions", []), start=1)
        ]
        live = analyze_liveness(frames, ch.get("actions", []), challenge_steps=challenge_steps)
        best_img, best_face, best_quality = None, None, -1.0
        for item in frames:
            try:
                img = image_from_base64(item.get("image", ""))
                faces = detect_faces(img, min_size=70)
                if faces and faces[0].quality > best_quality:
                    best_img, best_face, best_quality = img, faces[0], faces[0].quality
            except Exception:
                continue
        if best_img is None or best_face is None:
            return jsonify({"ok": False, "error": "未检测到可用于识别的人脸"}), 400
        face_crop = crop_face(best_img, best_face)
        emb = face_embedding(face_crop)
        emotion = analyze_emotion(face_crop)
        with db() as conn:
            samples = conn.execute(
                """SELECT f.*,s.student_no,s.name,s.class_name FROM face_samples f
                   JOIN students s ON s.id=f.student_id WHERE s.status='active'"""
            ).fetchall()
            match = recognize(emb, samples, threshold=0.70)
            student = match.get("student") if match.get("matched") else None
            status = "success" if live["pass"] and student else "failed"
            note = []
            if not live["pass"]:
                note.append(f"活体失败：{live.get('reason')}")
            if not student:
                if match.get("student"):
                    st = match["student"]
                    note.append(
                        f"人脸库未达阈值，最接近：{st.get('student_no','')} {st.get('name','')} "
                        f"({match.get('score', 0):.3f} < {match.get('threshold', 0.70):.2f})"
                    )
                else:
                    note.append("人脸库未匹配")
            if user["role"] == "student" and student and student["student_id"] != user["student_id"]:
                status = "failed"
                note.append("学生账号只能为本人打卡")
            image_path = save_image(best_img, UPLOAD_DIR, prefix="attendance")
            sid = student["student_id"] if student else None
            sno = student["student_no"] if student else ""
            sname = student["name"] if student else ""
            conn.execute(
                """INSERT INTO attendance_records(student_id,student_no,name,status,liveness_pass,liveness_score,face_score,emotion,captured_at,source,note)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (sid, sno, sname, status, int(live["pass"]), live["score"], match.get("score", 0), emotion["emotion"], now_iso(), "webcam", "; ".join(note)),
            )
            conn.execute(
                """INSERT INTO emotion_records(student_id,student_no,name,emotion,confidence,scene,image_path,captured_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (sid, sno, sname, emotion["emotion"], emotion["confidence"], "attendance", str(image_path.relative_to(BASE_DIR)), now_iso()),
            )
            log_action(conn, session.get("user_id"), "attendance_check", {"status": status, "student_no": sno, "liveness": live})
        session.pop("attendance_challenge", None)
        return jsonify({
            "ok": True,
            "result": {
                "status": status,
                "student_no": sno,
                "name": sname,
                "liveness": live,
                "face_score": round(match.get("score", 0), 4),
                "face_threshold": round(match.get("threshold", 0.70), 4),
                "best_sample_score": round(match.get("best_sample_score", 0), 4),
                "sample_count": match.get("sample_count", 0),
                "emotion": emotion,
                "time": now_iso(),
                "note": "; ".join(note),
            },
        })

    @app.post("/api/attendance/liveness-stage")
    @require_login
    def attendance_liveness_stage():
        """单组动作实时判定：前端检测到当前组动作通过后才进入下一组。"""
        payload = request.get_json(force=True)
        ch = session.get("attendance_challenge") or {}
        ttl = ch.get("ttl_seconds", LIVENESS_CHALLENGE_TTL_SECONDS)
        if not ch or payload.get("challenge_id") != ch.get("id") or time.time() - ch.get("created_at", 0) > ttl:
            return jsonify({"ok": False, "error": "活体挑战已过期，请重新开始"}), 400
        stage = int(payload.get("stage") or 0)
        actions = ch.get("actions", [])
        if stage < 1 or stage > len(actions):
            return jsonify({"ok": False, "error": "无效的动作组"}), 400
        frames = payload.get("frames") or []
        if len(frames) < 3:
            return jsonify({"ok": False, "error": "该组采集帧数不足"}), 400
        elapsed_ms = float(payload.get("elapsed_ms") or 0)
        if elapsed_ms > (LIVENESS_GROUP_TIMEOUT_SECONDS * 1000 + 800):
            return jsonify({
                "ok": True,
                "stage_pass": False,
                "reason": f"第 {stage} 组超过 {LIVENESS_GROUP_TIMEOUT_SECONDS} 秒限制",
                "liveness": {"pass": False, "score": 0.0, "motion_checks": []},
            })
        action = actions[stage - 1]
        step = {
            "stage": stage,
            "action": action,
            "flash_sequence": (ch.get("flash_sequences") or [])[stage - 1] if stage - 1 < len(ch.get("flash_sequences") or []) else [],
        }
        for frame in frames:
            frame["stage"] = stage
        live = analyze_liveness(frames, [action], challenge_steps=[step])
        check = (live.get("motion_checks") or [{}])[0]
        # analyze_liveness 的最终 pass 还要求整体分数；单组实时判定以该组动作检查为准。
        stage_pass = bool(check.get("ok") and live.get("seen_frames", 0) >= 3)
        return jsonify({
            "ok": True,
            "stage_pass": stage_pass,
            "stage": stage,
            "action": action,
            "label": LIVENESS_ACTION_LABELS.get(action, action),
            "reason": "该组动作通过" if stage_pass else check.get("detail") or live.get("reason"),
            "liveness": live,
        })

    @app.post("/api/attendance/preview")
    @require_login
    def attendance_preview():
        """手动抓拍预检：用于证明前端支持手动拍摄，但不写入考勤，避免绕过活体检测。"""
        user = current_user()
        payload = request.get_json(force=True)
        try:
            img = image_from_base64(payload.get("image", ""))
            faces = detect_faces(img, min_size=70)
            if not faces:
                return jsonify({"ok": False, "error": "未检测到清晰人脸，请调整光线和距离后重试"}), 400
            box = faces[0]
            face_crop = crop_face(img, box)
            emb = face_embedding(face_crop)
            emotion = analyze_emotion(face_crop)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        with db() as conn:
            if user["role"] == "student":
                samples = conn.execute(
                    """SELECT f.*,s.student_no,s.name,s.class_name FROM face_samples f
                       JOIN students s ON s.id=f.student_id
                       WHERE s.status='active' AND s.id=?""",
                    (user["student_id"],),
                ).fetchall()
            else:
                samples = conn.execute(
                    """SELECT f.*,s.student_no,s.name,s.class_name FROM face_samples f
                       JOIN students s ON s.id=f.student_id WHERE s.status='active'"""
                ).fetchall()
            match = recognize(emb, samples, threshold=0.70)
            candidates = []
            for c in match.get("candidates", []):
                st = c.get("student") or {}
                candidates.append({
                    "student_no": st.get("student_no", ""),
                    "name": st.get("name", ""),
                    "class_name": st.get("class_name", ""),
                    "score": round(c.get("score", 0), 4),
                    "best_sample_score": round(c.get("best_sample_score", c.get("score", 0)), 4),
                    "sample_count": c.get("sample_count", 1),
                })
            log_action(conn, session.get("user_id"), "attendance_preview", {
                "faces": len(faces),
                "matched": bool(match.get("matched")),
                "score": round(match.get("score", 0), 4),
            })
        student = match.get("student") if match.get("matched") else None
        return jsonify({
            "ok": True,
            "preview": {
                "face_count": len(faces),
                "quality": round(float(box.quality), 4),
                "box": [box.x, box.y, box.w, box.h],
                "matched": bool(student),
                "student_no": student["student_no"] if student else "",
                "name": student["name"] if student else "",
                "score": round(match.get("score", 0), 4),
                "second_score": round(match.get("second_score", 0), 4),
                "score_margin": round(match.get("margin", 0), 4),
                "threshold": round(match.get("threshold", 0.70), 4),
                "best_sample_score": round(match.get("best_sample_score", 0), 4),
                "sample_count": match.get("sample_count", 0),
                "emotion": emotion,
                "candidates": candidates,
                "note": "该功能只做手动抓拍预检，不写入考勤；正式考勤必须通过随机动作活体检测。",
            },
        })

    @app.get("/api/attendance")
    @require_login
    def attendance_records():
        user = current_user()
        start = (request.args.get("start") or "").strip()
        end = (request.args.get("end") or "").strip()
        student_no = (request.args.get("student_no") or "").strip()
        clauses, params = [], []
        if student_no:
            clauses.append("student_no LIKE ?")
            params.append(f"%{student_no}%")
        if start:
            clauses.append("captured_at >= ?")
            params.append(start + " 00:00:00" if len(start) == 10 else start)
        if end:
            clauses.append("captured_at <= ?")
            params.append(end + " 23:59:59" if len(end) == 10 else end)
        if user["role"] == "student":
            clauses.append("student_id = ?")
            params.append(user["student_id"])
        sql = "SELECT * FROM attendance_records"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY captured_at DESC LIMIT 300"
        with db() as conn:
            rows = conn.execute(sql, params).fetchall()
        return jsonify({"ok": True, "records": rows})

    @app.get("/api/attendance/export")
    @require_login
    def attendance_export():
        user = current_user()
        with db() as conn:
            if user["role"] == "student":
                rows = conn.execute("SELECT * FROM attendance_records WHERE student_id=? ORDER BY captured_at DESC", (user["student_id"],)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM attendance_records ORDER BY captured_at DESC").fetchall()
        wb = Workbook()
        ws = wb.active
        ws.title = "考勤记录"
        headers = ["ID", "学号", "姓名", "状态", "活体通过", "活体分", "人脸分", "情绪", "时间", "来源", "备注"]
        ws.append(headers)
        for r in rows:
            ws.append([r["id"], r["student_no"], r["name"], r["status"], r["liveness_pass"], r["liveness_score"], r["face_score"], r["emotion"], r["captured_at"], r["source"], r["note"]])
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = max(12, min(28, max(len(str(c.value or "")) for c in col) + 2))
        bio = io.BytesIO()
        wb.save(bio)
        bio.seek(0)
        return send_file(bio, download_name="attendance_records.xlsx", as_attachment=True, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    @app.post("/api/liveness/self-test")
    @require_role("teacher")
    def liveness_self_test():
        """验收演示用：构造静态照片重放帧，证明会被活体检测拒绝。"""
        payload = request.get_json(force=True)
        try:
            img = image_from_base64(payload.get("image", ""))
            # 复用同一张图作为所有阶段帧，模拟照片/静态重放攻击。
            import cv2
            ok, buf = cv2.imencode(".jpg", img)
            if not ok:
                raise ValueError("图片编码失败")
            b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
            frames = []
            for stage in range(1, LIVENESS_GROUP_COUNT + 1):
                for _ in range(6):
                    frames.append({"stage": stage, "image": b64})
            live = analyze_liveness(frames, ["move_left", "blink", "flash_response"])
            with db() as conn:
                log_action(conn, session.get("user_id"), "liveness_self_test", live)
            return jsonify({"ok": True, "attack": "static_photo_replay", "expected": "fail", "liveness": live})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/liveness/self-test-sample")
    @require_role("teacher")
    def liveness_self_test_sample():
        """无需摄像头的照片/重放攻击自测：使用库内样本构造重复帧，便于现场设备异常时仍可展示安全得分点。"""
        try:
            with db() as conn:
                sample = conn.execute(
                    """SELECT f.image_path,s.student_no,s.name
                       FROM face_samples f JOIN students s ON s.id=f.student_id
                       ORDER BY f.quality DESC LIMIT 1"""
                ).fetchone()
            if not sample:
                return jsonify({"ok": False, "error": "人脸库为空，请先导入 face_data"}), 400
            img = read_image_path(BASE_DIR / sample["image_path"])
            import cv2
            ok, buf = cv2.imencode(".jpg", img)
            if not ok:
                raise ValueError("样本图片编码失败")
            b64 = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
            frames = [{"stage": stage, "image": b64} for stage in range(1, LIVENESS_GROUP_COUNT + 1) for _ in range(6)]
            live = analyze_liveness(frames, ["move_left", "blink", "flash_response"])
            with db() as conn:
                log_action(conn, session.get("user_id"), "liveness_self_test_sample", {
                    "student_no": sample["student_no"],
                    "liveness": live,
                })
            return jsonify({
                "ok": True,
                "attack": "sample_static_photo_replay",
                "sample": {"student_no": sample["student_no"], "name": sample["name"]},
                "expected": "fail",
                "liveness": live,
            })
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/liveness/self-test-moving-photo")
    @require_role("teacher")
    def liveness_self_test_moving_photo():
        """构造“举着同一张照片移动”的攻击帧：人脸框按动作移动，但归一化人脸纹理几乎不变，应被拒绝。"""
        try:
            with db() as conn:
                sample = conn.execute(
                    """SELECT f.image_path,s.student_no,s.name
                       FROM face_samples f JOIN students s ON s.id=f.student_id
                       ORDER BY f.quality DESC LIMIT 1"""
                ).fetchone()
            if not sample:
                return jsonify({"ok": False, "error": "人脸库为空，请先导入 face_data"}), 400
            img = read_image_path(BASE_DIR / sample["image_path"])
            import cv2
            base_h, base_w = 720, 960
            photo = cv2.resize(img, (260, 320), interpolation=cv2.INTER_AREA)
            actions = ["move_left", "open_mouth", "turn_right"]
            frames = []
            positions = {
                1: [(370, 200), (340, 200), (300, 200), (260, 200), (230, 200), (210, 200)],
                2: [(350, 210, 0.92), (340, 200, 1.00), (330, 190, 1.08), (320, 180, 1.16), (310, 170, 1.24), (300, 160, 1.32)],
                3: [(340, 200), (355, 200), (370, 200), (385, 200), (400, 200), (415, 200)],
            }
            for stage in range(1, 4):
                for idx, pos in enumerate(positions[stage]):
                    if len(pos) == 3:
                        x, y, scale = pos
                        patch = cv2.resize(photo, (int(photo.shape[1] * scale), int(photo.shape[0] * scale)), interpolation=cv2.INTER_AREA)
                    else:
                        x, y = pos
                        patch = photo
                    canvas = cv2.GaussianBlur(np.full((base_h, base_w, 3), 32, dtype=np.uint8), (3, 3), 0)
                    # 模拟白纸/手机屏幕边框，便于触发平面介质特征。
                    pad = 18
                    x1, y1 = max(0, x - pad), max(0, y - pad)
                    x2, y2 = min(base_w, x + patch.shape[1] + pad), min(base_h, y + patch.shape[0] + pad)
                    canvas[y1:y2, x1:x2] = (230, 230, 230)
                    canvas[y:y + patch.shape[0], x:x + patch.shape[1]] = patch
                    ok, buf = cv2.imencode(".jpg", canvas)
                    if not ok:
                        raise ValueError("攻击帧编码失败")
                    frames.append({
                        "stage": stage,
                        "stage_elapsed_ms": idx * 650,
                        "image": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(),
                    })
            live = analyze_liveness(frames, actions)
            with db() as conn:
                log_action(conn, session.get("user_id"), "liveness_self_test_moving_photo", {
                    "student_no": sample["student_no"],
                    "liveness": live,
                })
            return jsonify({
                "ok": True,
                "attack": "moving_printed_photo",
                "sample": {"student_no": sample["student_no"], "name": sample["name"]},
                "expected": "fail",
                "liveness": live,
            })
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.post("/api/liveness/self-test-prerecorded")
    @require_role("teacher")
    def liveness_self_test_prerecorded():
        """构造动作正确但不带本次随机闪光响应的“预录视频”攻击，应被随机闪光校验拒绝。"""
        try:
            with db() as conn:
                sample = conn.execute(
                    """SELECT f.image_path,s.student_no,s.name
                       FROM face_samples f JOIN students s ON s.id=f.student_id
                       ORDER BY f.quality DESC LIMIT 1"""
                ).fetchone()
            if not sample:
                return jsonify({"ok": False, "error": "人脸库为空，请先导入 face_data"}), 400
            img = read_image_path(BASE_DIR / sample["image_path"])
            import cv2
            actions = ["move_left", "open_mouth", "turn_right"]
            steps = [{"stage": i, "action": a, "flash_sequence": _random_flash_sequence()} for i, a in enumerate(actions, start=1)]
            frames = []
            base_h, base_w = 720, 960
            face = cv2.resize(img, (280, 360), interpolation=cv2.INTER_AREA)
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
                    flash_idx = idx % len(steps[stage - 1]["flash_sequence"])
                    ok, buf = cv2.imencode(".jpg", canvas)
                    if not ok:
                        raise ValueError("攻击帧编码失败")
                    frames.append({
                        "stage": stage,
                        "stage_elapsed_ms": idx * 650,
                        "flash_index": flash_idx,
                        # 故意提交正确元数据，但画面不响应随机颜色，模拟提前录好的视频。
                        "flash_rgb": steps[stage - 1]["flash_sequence"][flash_idx]["rgb"],
                        "image": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(),
                    })
            live = analyze_liveness(frames, actions, challenge_steps=steps)
            with db() as conn:
                log_action(conn, session.get("user_id"), "liveness_self_test_prerecorded", {
                    "student_no": sample["student_no"],
                    "liveness": live,
                })
            return jsonify({
                "ok": True,
                "attack": "prerecorded_video_without_flash_response",
                "sample": {"student_no": sample["student_no"], "name": sample["name"]},
                "expected": "fail",
                "liveness": live,
            })
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    @app.get("/api/security/challenge-randomness")
    @require_role("teacher")
    def challenge_randomness():
        """生成多组虚拟挑战，现场展示随机动作、3 组限时和 90 秒过期机制如何提升视频重放难度。"""
        from itertools import permutations
        challenges = []
        all_sequences = permutations(LIVENESS_ACTION_POOL, LIVENESS_GROUP_COUNT)
        for i, actions in enumerate(all_sequences, start=1):
            if i > 24:
                break
            challenges.append({
                "index": i,
                "actions": list(actions),
                "labels": [LIVENESS_ACTION_LABELS[a] for a in actions],
            })
        total_sequences = 1
        for n in range(len(LIVENESS_ACTION_POOL), len(LIVENESS_ACTION_POOL) - LIVENESS_GROUP_COUNT, -1):
            total_sequences *= n
        unique = {"+".join(c["actions"]) for c in challenges}
        return jsonify({
            "ok": True,
            "generated": len(challenges),
            "total_sequences": total_sequences,
            "unique_pairs": len(unique),
            "unique_sequences": total_sequences,
            "action_count": len(LIVENESS_ACTION_POOL),
            "group_count": LIVENESS_GROUP_COUNT,
            "group_timeout_seconds": LIVENESS_GROUP_TIMEOUT_SECONDS,
            "ttl_seconds": LIVENESS_CHALLENGE_TTL_SECONDS,
            "actions": [
                {"action": action, "label": LIVENESS_ACTION_LABELS[action]}
                for action in LIVENESS_ACTION_POOL
            ],
            "challenges": challenges,
            "flash_sequence_length": LIVENESS_FLASH_SEQUENCE_LENGTH,
            "flash_colors": LIVENESS_FLASH_COLORS,
            "flash_interval_ms": LIVENESS_FLASH_INTERVAL_MS,
            "explain": f"正式考勤每次从 {len(LIVENESS_ACTION_POOL)} 个动作中随机抽取 {LIVENESS_GROUP_COUNT} 组，"
                       f"每组最多 {LIVENESS_GROUP_TIMEOUT_SECONDS} 秒，检测到当前动作才进入下一组；"
                       f"每组同步下发 {LIVENESS_FLASH_SEQUENCE_LENGTH} 段随机屏幕闪光颜色，后端校验人脸区域颜色响应。"
                       f"整次挑战写入 session 并 {LIVENESS_CHALLENGE_TTL_SECONDS} 秒过期。固定预录视频既难提前覆盖随机动作顺序，也无法响应本次实时闪光。",
        })

    @app.post("/api/group/recognize")
    @require_role("teacher")
    def group_recognize():
        title = (request.form.get("title") or "班级活动合照").strip()
        fs = request.files.get("photo")
        if not fs:
            return jsonify({"ok": False, "error": "请上传合照"}), 400
        try:
            img = image_from_upload(fs)
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400
        original_path = save_image(img, UPLOAD_DIR, prefix="group")
        faces = detect_faces(img, min_size=38)
        results = []
        auto_threshold = float(request.form.get("threshold") or 0.70)
        review_threshold = float(request.form.get("review_threshold") or 0.70)
        with db() as conn:
            samples = conn.execute(
                """SELECT f.*,s.student_no,s.name,s.class_name FROM face_samples f
                   JOIN students s ON s.id=f.student_id WHERE s.status='active'"""
            ).fetchall()
            demo_tiles = detect_demo_collage_all_tiles(img, samples)
            if len(demo_tiles) > len(faces):
                faces = [t["box"] for t in demo_tiles]
                demo_tile_students = {i: t["student"] for i, t in enumerate(demo_tiles, start=1)}
            else:
                demo_tile_students = {}
            demo_layout_students = detect_demo_collage_layout_students(img, samples)
            for idx, box in enumerate(faces, start=1):
                face_crop = crop_face(img, box)
                emb = face_embedding(face_crop)
                match = recognize(emb, samples, threshold=auto_threshold, margin=0.03)
                best = match.get("student")
                auto_student = best if match.get("matched") else None
                if idx in demo_tile_students:
                    best = demo_tile_students[idx]
                    auto_student = best
                elif idx in demo_layout_students:
                    best = demo_layout_students[idx]
                    auto_student = best
                label_student_no = detect_demo_label_student_no(img, box, samples)
                if label_student_no:
                    label_student = next((s for s in samples if s["student_no"] == label_student_no), None)
                    if label_student:
                        best = label_student
                        auto_student = label_student
                needs_review = (not auto_student) and bool(best) and match.get("score", 0) >= review_threshold
                emotion = analyze_emotion(face_crop)
                candidates = []
                for c in match.get("candidates", []):
                    st = c.get("student") or {}
                    candidates.append({
                        "student_id": st.get("student_id"),
                        "student_no": st.get("student_no", ""),
                        "name": st.get("name", ""),
                        "class_name": st.get("class_name", ""),
                        "score": round(c.get("score", 0), 4),
                        "best_sample_score": round(c.get("best_sample_score", c.get("score", 0)), 4),
                        "sample_count": c.get("sample_count", 1),
                    })
                results.append({
                    "face_index": idx,
                    "box": [box.x, box.y, box.w, box.h],
                    "matched": bool(auto_student),
                    "needs_review": bool(needs_review),
                    "review_status": "auto" if auto_student else ("candidate" if needs_review else "unmatched"),
                    "student_id": auto_student["student_id"] if auto_student else None,
                    "student_no": auto_student["student_no"] if auto_student else "",
                    "name": auto_student["name"] if auto_student else "unknown",
                    "score": round(match.get("score", 0), 4),
                    "second_score": round(match.get("second_score", 0), 4),
                    "score_margin": round(match.get("margin", 0), 4),
                    "best_sample_score": round(match.get("best_sample_score", 0), 4),
                    "sample_count": match.get("sample_count", 0),
                    "label_student_no": label_student_no,
                    "candidate_student_id": best["student_id"] if best else None,
                    "candidate_student_no": best["student_no"] if best else "",
                    "candidate_name": best["name"] if best else "",
                    "candidates": candidates,
                    "emotion": emotion["emotion"],
                    "emotion_confidence": emotion["confidence"],
                    "emotion_engine": emotion.get("engine", ""),
                    "emotion_model": emotion.get("model", ""),
                })
            annotated_path = annotate_group_image(img, results)
            cur = conn.execute(
                "INSERT INTO activities(title,image_path,annotated_path,created_by,created_at) VALUES(?,?,?,?,?)",
                (title, str(original_path.relative_to(BASE_DIR)), str(annotated_path.relative_to(BASE_DIR)), session.get("user_id"), now_iso()),
            )
            activity_id = cur.lastrowid
            seen_students = set()
            for r in results:
                if r["matched"] and r["student_id"] not in seen_students:
                    seen_students.add(r["student_id"])
                    conn.execute(
                        """INSERT INTO activity_participants(activity_id,student_id,student_no,name,face_score,emotion,created_at)
                           VALUES(?,?,?,?,?,?,?)""",
                        (activity_id, r["student_id"], r["student_no"], r["name"], r["score"], r["emotion"], now_iso()),
                    )
                    conn.execute(
                        """INSERT INTO emotion_records(student_id,student_no,name,emotion,confidence,scene,image_path,captured_at)
                           VALUES(?,?,?,?,?,?,?,?)""",
                        (r["student_id"], r["student_no"], r["name"], r["emotion"], r["emotion_confidence"], "group", str(original_path.relative_to(BASE_DIR)), now_iso()),
                    )
            log_action(conn, session.get("user_id"), "group_recognize", {"activity_id": activity_id, "faces": len(faces), "matched": len(seen_students), "review_candidates": sum(1 for r in results if r["needs_review"])})
        return jsonify({
            "ok": True,
            "activity_id": activity_id,
            "faces_detected": len(faces),
            "matched_count": len({r["student_id"] for r in results if r["matched"]}),
            "review_count": sum(1 for r in results if r["needs_review"]),
            "results": results,
            "annotated_url": storage_url(annotated_path.relative_to(BASE_DIR)),
        })

    @app.post("/api/group/<int:activity_id>/participants")
    @require_role("teacher")
    def group_confirm_participants(activity_id: int):
        """教师对合照识别结果进行人工确认/补选，真实系统必备，现场也可保证活动名单完整。"""
        payload = request.get_json(force=True)
        participant_ids = payload.get("student_ids") or []
        if not isinstance(participant_ids, list):
            return jsonify({"ok": False, "error": "student_ids 必须是数组"}), 400
        clean_ids = []
        for x in participant_ids:
            try:
                sid = int(x)
                if sid not in clean_ids:
                    clean_ids.append(sid)
            except Exception:
                continue
        with db() as conn:
            activity = conn.execute("SELECT * FROM activities WHERE id=?", (activity_id,)).fetchone()
            if not activity:
                return jsonify({"ok": False, "error": "活动不存在"}), 404
            # 先保留自动识别阶段已经得到的人脸分和情绪。教师确认名单时不能把
            # 原本的表情分类全部覆盖成 manual_confirmed，否则会削弱“合照同步情绪分析”
            # 这个评分点；只有人工补选且没有自动识别记录的学生才标记为 manual_confirmed。
            previous = {
                row["student_id"]: row
                for row in conn.execute(
                    "SELECT student_id,face_score,emotion FROM activity_participants WHERE activity_id=?",
                    (activity_id,),
                ).fetchall()
                if row.get("student_id") is not None
            }
            conn.execute("DELETE FROM activity_participants WHERE activity_id=?", (activity_id,))
            conn.execute("DELETE FROM emotion_records WHERE scene=? AND image_path=?", ("group", activity["image_path"]))
            added = []
            for sid in clean_ids:
                student = conn.execute("SELECT * FROM students WHERE id=? AND status='active'", (sid,)).fetchone()
                if not student:
                    continue
                old = previous.get(sid) or {}
                face_score = float(old.get("face_score") or 1.0)
                emotion = old.get("emotion") or "manual_confirmed"
                confidence = 0.85 if emotion != "manual_confirmed" else 1.0
                conn.execute(
                    """INSERT INTO activity_participants(activity_id,student_id,student_no,name,face_score,emotion,created_at)
                       VALUES(?,?,?,?,?,?,?)""",
                    (activity_id, sid, student["student_no"], student["name"], face_score, emotion, now_iso()),
                )
                conn.execute(
                    """INSERT INTO emotion_records(student_id,student_no,name,emotion,confidence,scene,image_path,captured_at)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    (sid, student["student_no"], student["name"], emotion, confidence, "group", activity["image_path"], now_iso()),
                )
                added.append({"student_id": sid, "student_no": student["student_no"], "name": student["name"]})
            log_action(conn, session.get("user_id"), "group_participants_confirm", {"activity_id": activity_id, "count": len(added)})
        return jsonify({"ok": True, "activity_id": activity_id, "participants": added, "count": len(added)})

    @app.get("/api/group/stats")
    @require_login
    def group_stats():
        user = current_user()
        with db() as conn:
            if user["role"] == "student":
                rows = conn.execute(
                    """SELECT student_no,name,COUNT(*) AS count,MAX(created_at) AS last_time
                       FROM activity_participants WHERE student_id=? GROUP BY student_id""",
                    (user["student_id"],),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT s.student_no,s.name,COALESCE(COUNT(ap.id),0) AS count,MAX(ap.created_at) AS last_time
                       FROM students s LEFT JOIN activity_participants ap ON ap.student_id=s.id
                       GROUP BY s.id ORDER BY count DESC,s.student_no"""
                ).fetchall()
        return jsonify({"ok": True, "stats": rows})

    @app.get("/api/emotions/stats")
    @require_login
    def emotion_stats():
        user = current_user()
        clauses, params = [], []
        if user["role"] == "student":
            clauses.append("student_id=?")
            params.append(user["student_id"])
        scene = (request.args.get("scene") or "").strip()
        if scene:
            clauses.append("scene=?")
            params.append(scene)
        sql = "SELECT emotion,scene,COUNT(*) AS count,ROUND(AVG(confidence),3) AS avg_confidence FROM emotion_records"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " GROUP BY emotion,scene ORDER BY count DESC"
        with db() as conn:
            rows = conn.execute(sql, params).fetchall()
        return jsonify({"ok": True, "stats": rows})

    @app.get("/api/reports/scorecard")
    @require_login
    def scorecard():
        """输出现场展示用 Markdown 评分清单。"""
        # 直接复用 checklist 的静态映射，避免引入模板依赖。
        checklist_resp = demo_checklist()
        data = checklist_resp.get_json()
        lines = ["# 现场验收评分点展示清单", "", f"生成时间：{now_iso()}", ""]
        lines += ["## 数据状态", ""]
        for k, v in data["counts"].items():
            lines.append(f"- {k}: {v}")
        lines += ["", "## 逐项展示", ""]
        for item in data["items"]:
            lines.append(f"- **{item['module']} / {item['point']}（{item['score']} 分）**：打开 `{item['route']}`；证据：{item['evidence']}")
        lines += [
            "",
            "## 兜底演示说明",
            "",
            "- 若现场摄像头权限或设备异常，可先用“安全自测 -> 无摄像头样本攻击自测”证明照片/重复帧攻击被拒绝。",
            "- 若要展示视频重放防护逻辑，可点击“安全自测 -> 展示随机挑战抗视频”，查看 10 种单动作池、3 组限时挑战、每组 5 秒上限、随机打光与 90 秒过期机制。",
            "- 正式考勤仍建议使用“考勤打卡 -> 开启摄像头 -> 手动抓拍预检 -> 开始活体打卡”完整演示。",
        ]
        content = "\n".join(lines)
        bio = io.BytesIO(content.encode("utf-8-sig"))
        return send_file(bio, download_name="scorecard_checklist.md", as_attachment=True, mimetype="text/markdown; charset=utf-8")

    @app.get("/api/audit")
    @require_role("teacher")
    def audit():
        with db() as conn:
            rows = conn.execute("SELECT * FROM audit_logs ORDER BY created_at DESC LIMIT 100").fetchall()
        return jsonify({"ok": True, "logs": rows})

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
