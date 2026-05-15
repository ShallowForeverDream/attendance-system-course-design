from __future__ import annotations

import re
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def filename_basename(filename: str | Path) -> str:
    """兼容普通文件名、webkitdirectory 的 a/b.jpg 和 Windows 反斜杠路径。"""
    return str(filename or "").replace("\\", "/").split("/")[-1].strip()


def parse_student_image_filename(filename: str | Path) -> dict:
    """解析老师给的人脸图片命名：学号-姓名-专业-性别。

    仅支持标准格式 “学号-姓名-专业-性别”。
    """
    base = filename_basename(filename)
    stem = Path(base).stem.strip()
    normalized = re.sub(r"[－—–]+", "-", stem)
    parts = [p.strip() for p in normalized.split("-") if p.strip()]
    if len(parts) != 4:
        return {"student_no": "", "name": "", "class_name": "", "gender": "", "filename": base}
    return {
        "student_no": parts[0],
        "name": parts[1],
        "class_name": parts[2],
        "gender": parts[-1],
        "filename": base,
    }


def is_supported_face_image(filename: str | Path) -> bool:
    return Path(filename_basename(filename)).suffix.lower() in IMAGE_EXTS
