from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
FACES_DIR = STORAGE_DIR / "faces"
UPLOAD_DIR = STORAGE_DIR / "uploads"
ANNOTATED_DIR = STORAGE_DIR / "annotated"

for directory in (DATA_DIR, STORAGE_DIR, FACES_DIR, UPLOAD_DIR, ANNOTATED_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SECRET_KEY = os.getenv("SECRET_KEY", "dev-attendance-secret-change-me")
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(DATA_DIR / "app.db")))
if not DATABASE_PATH.is_absolute():
    DATABASE_PATH = BASE_DIR / DATABASE_PATH
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.78"))
UPLOAD_MAX_MB = int(os.getenv("UPLOAD_MAX_MB", "12"))
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
