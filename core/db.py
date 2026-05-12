from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterable

from werkzeug.security import generate_password_hash

from .config import DATABASE_PATH

SCHEMA = r"""
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_no TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    class_name TEXT DEFAULT '',
    gender TEXT DEFAULT '',
    phone TEXT DEFAULT '',
    email TEXT DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('teacher','student')),
    student_id INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS face_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    image_path TEXT NOT NULL,
    embedding TEXT NOT NULL,
    quality REAL NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS attendance_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    student_no TEXT DEFAULT '',
    name TEXT DEFAULT '',
    status TEXT NOT NULL,
    liveness_pass INTEGER NOT NULL DEFAULT 0,
    liveness_score REAL NOT NULL DEFAULT 0,
    face_score REAL NOT NULL DEFAULT 0,
    emotion TEXT DEFAULT '',
    captured_at TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'webcam',
    note TEXT DEFAULT '',
    FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS emotion_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER,
    student_no TEXT DEFAULT '',
    name TEXT DEFAULT '',
    emotion TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0,
    scene TEXT NOT NULL,
    image_path TEXT DEFAULT '',
    captured_at TEXT NOT NULL,
    FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS activities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    image_path TEXT NOT NULL,
    annotated_path TEXT DEFAULT '',
    created_by INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY(created_by) REFERENCES users(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS activity_participants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    activity_id INTEGER NOT NULL,
    student_id INTEGER,
    student_no TEXT DEFAULT '',
    name TEXT DEFAULT '',
    face_score REAL NOT NULL DEFAULT 0,
    emotion TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY(activity_id) REFERENCES activities(id) ON DELETE CASCADE,
    FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    action TEXT NOT NULL,
    detail TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS demo_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric_key TEXT NOT NULL UNIQUE,
    metric_value REAL NOT NULL,
    sample_size INTEGER NOT NULL DEFAULT 0,
    detail TEXT DEFAULT '',
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_attendance_captured_at ON attendance_records(captured_at);
CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance_records(student_id);
CREATE INDEX IF NOT EXISTS idx_emotion_student ON emotion_records(student_id);
CREATE INDEX IF NOT EXISTS idx_activity_participants_student ON activity_participants(student_id);
"""


def now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def dict_factory(cursor: sqlite3.Cursor, row: sqlite3.Row) -> dict[str, Any]:
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def get_conn() -> sqlite3.Connection:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DATABASE_PATH))
    conn.row_factory = dict_factory
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db() -> Iterable[sqlite3.Connection]:
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(seed: bool = True) -> None:
    with db() as conn:
        conn.executescript(SCHEMA)
        if seed:
            seed_demo_data(conn)


def seed_demo_data(conn: sqlite3.Connection) -> None:
    ts = now_iso()
    teacher = conn.execute("SELECT id FROM users WHERE username=?", ("teacher",)).fetchone()
    if not teacher:
        conn.execute(
            "INSERT INTO users(username,password_hash,role,created_at) VALUES(?,?,?,?)",
            ("teacher", generate_password_hash("teacher123"), "teacher", ts),
        )
    demo_students = [
        ("20260001", "张三", "内容安全实践班", "student01", "student123"),
        ("20260002", "李四", "内容安全实践班", "student02", "student123"),
    ]
    for student_no, name, class_name, username, password in demo_students:
        row = conn.execute("SELECT id FROM students WHERE student_no=?", (student_no,)).fetchone()
        if not row:
            cur = conn.execute(
                """INSERT INTO students(student_no,name,class_name,status,created_at,updated_at)
                   VALUES(?,?,?,?,?,?)""",
                (student_no, name, class_name, "active", ts, ts),
            )
            student_id = cur.lastrowid
        else:
            student_id = row["id"]
        user = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
        if not user:
            conn.execute(
                "INSERT INTO users(username,password_hash,role,student_id,created_at) VALUES(?,?,?,?,?)",
                (username, generate_password_hash(password), "student", student_id, ts),
            )


def log_action(conn: sqlite3.Connection, user_id: int | None, action: str, detail: Any = "") -> None:
    if not isinstance(detail, str):
        detail = json.dumps(detail, ensure_ascii=False)
    conn.execute(
        "INSERT INTO audit_logs(user_id,action,detail,created_at) VALUES(?,?,?,?)",
        (user_id, action, detail, now_iso()),
    )


def upsert_metric(conn: sqlite3.Connection, key: str, value: float, sample_size: int = 0, detail: Any = "") -> None:
    if not isinstance(detail, str):
        detail = json.dumps(detail, ensure_ascii=False)
    conn.execute(
        """INSERT INTO demo_metrics(metric_key,metric_value,sample_size,detail,updated_at)
           VALUES(?,?,?,?,?)
           ON CONFLICT(metric_key) DO UPDATE SET
             metric_value=excluded.metric_value,
             sample_size=excluded.sample_size,
             detail=excluded.detail,
             updated_at=excluded.updated_at""",
        (key, value, sample_size, detail, now_iso()),
    )
