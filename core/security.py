from __future__ import annotations

from functools import wraps
from typing import Callable

from flask import jsonify, session
from werkzeug.security import check_password_hash

from .db import db


def current_user() -> dict | None:
    uid = session.get("user_id")
    if not uid:
        return None
    with db() as conn:
        return conn.execute(
            """SELECT u.id,u.username,u.role,u.student_id,s.student_no,s.name AS student_name
               FROM users u LEFT JOIN students s ON s.id=u.student_id WHERE u.id=?""",
            (uid,),
        ).fetchone()


def require_login(fn: Callable):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"ok": False, "error": "请先登录"}), 401
        return fn(*args, **kwargs)
    return wrapper


def require_role(*roles: str):
    def decorator(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            user = current_user()
            if not user:
                return jsonify({"ok": False, "error": "请先登录"}), 401
            if user["role"] not in roles:
                return jsonify({"ok": False, "error": "权限不足"}), 403
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def verify_user(username: str, password: str) -> dict | None:
    with db() as conn:
        user = conn.execute(
            "SELECT id,username,password_hash,role,student_id FROM users WHERE username=?",
            (username.strip(),),
        ).fetchone()
    if user and check_password_hash(user["password_hash"], password):
        return user
    return None
