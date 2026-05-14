from __future__ import annotations

import re
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def filename_basename(filename: str | Path) -> str:
    """兼容普通文件名、webkitdirectory 的 a/b.jpg 和 Windows 反斜杠路径。"""
    return str(filename or "").replace("\\", "/").split("/")[-1].strip()


def parse_student_image_filename(filename: str | Path) -> dict:
    """解析老师给的人脸图片命名：学号-姓名-专业-性别。

    同时兼容用户口头描述里可能出现的 “学号-姓名=专业-性别”、中文长横线、
    以及旧版无分隔符命名。
    """
    base = filename_basename(filename)
    stem = Path(base).stem.strip()
    normalized = re.sub(r"[=－—–]+", "-", stem)
    parts = [p.strip() for p in normalized.split("-") if p.strip()]
    if len(parts) >= 4:
        return {
            "student_no": parts[0],
            "name": parts[1],
            "class_name": "-".join(parts[2:-1]),
            "gender": parts[-1],
            "filename": base,
        }
    if len(parts) == 3:
        return {
            "student_no": parts[0],
            "name": parts[1],
            "class_name": parts[2],
            "gender": "",
            "filename": base,
        }
    if len(parts) == 2:
        return {
            "student_no": parts[0],
            "name": parts[1],
            "class_name": "",
            "gender": "",
            "filename": base,
        }

    # 兼容 “2023000000001张三网安男.jpg” 这类无分隔符命名。
    m = re.match(r"^(?P<no>\d{8,13})(?P<rest>.+)$", stem)
    if m:
        rest = m.group("rest").strip()
        gender = ""
        if rest.endswith("男") or rest.endswith("女"):
            gender = rest[-1]
            rest = rest[:-1]
        class_name = "网络空间安全" if "网安" in rest or "网络" in rest else ""
        name = (
            rest.replace("网络空间安全", "")
            .replace("网络安全", "")
            .replace("网安", "")
            .replace("试验班", "")
            .replace("实验班", "")
            .strip()
        )
        return {"student_no": m.group("no"), "name": name or stem, "class_name": class_name, "gender": gender, "filename": base}

    return {"student_no": stem, "name": stem, "class_name": "", "gender": "", "filename": base}


def is_supported_face_image(filename: str | Path) -> bool:
    return Path(filename_basename(filename)).suffix.lower() in IMAGE_EXTS
