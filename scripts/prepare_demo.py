from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

from core.config import ANNOTATED_DIR, FACES_DIR, UPLOAD_DIR  # noqa: E402
from scripts.evaluate_face_data import evaluate  # noqa: E402
from scripts.group_collage_selftest import main as group_selftest_main  # noqa: E402
from scripts.import_face_data import import_face_data  # noqa: E402
from scripts.make_demo_collage import make_collage  # noqa: E402
from scripts.seed_demo_records import seed_demo_records  # noqa: E402


def safe_clear_dir(path: Path) -> None:
    path = path.resolve()
    root = ROOT.resolve()
    if not str(path).startswith(str(root)):
        raise RuntimeError(f"拒绝清理项目外目录：{path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / ".gitkeep").write_text("", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="一键准备现场演示数据：清理存储、导入 face_data、生成稳定 10 人合照并自测")
    parser.add_argument("--source", default=str(PROJECT_ROOT / "face_data"), help="老师提供的 face_data 目录")
    parser.add_argument("--skip-clean-storage", action="store_true", help="不清理 storage 下旧图片")
    args = parser.parse_args()

    if not args.skip_clean_storage:
        for d in (FACES_DIR, UPLOAD_DIR, ANNOTATED_DIR):
            safe_clear_dir(d)

    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    import_report = import_face_data(Path(args.source), reset_demo=True)
    (docs / "face_data_import_report.json").write_text(json.dumps(import_report, ensure_ascii=False, indent=2), encoding="utf-8")
    eval_report = evaluate()
    (docs / "face_data_evaluation_report.json").write_text(json.dumps(eval_report, ensure_ascii=False, indent=2), encoding="utf-8")
    collage_report = make_collage()
    (docs / "demo_collage_report.json").write_text(json.dumps(collage_report, ensure_ascii=False, indent=2), encoding="utf-8")
    collage_50_report = make_collage(count=50, size=180, output_name="demo_collage_50_pressure.png")
    (docs / "demo_collage_50_report.json").write_text(json.dumps(collage_50_report, ensure_ascii=False, indent=2), encoding="utf-8")
    group_selftest_main()
    seed_report = seed_demo_records()
    (docs / "demo_seed_records_report.json").write_text(json.dumps(seed_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nDEMO READY")
    print(json.dumps({
        "face_images": import_report["total_images"],
        "samples_added": import_report["samples_added"],
        "failed": len(import_report["failed"]),
        "demo_collage": collage_report["image_path"],
        "demo_collage_50": collage_50_report["image_path"],
        "seed_records": seed_report,
        "reports": [
            "docs/face_data_import_report.json",
            "docs/face_data_evaluation_report.json",
            "docs/demo_collage_report.json",
            "docs/demo_collage_50_report.json",
            "docs/group_collage_selftest_report.json",
            "docs/demo_seed_records_report.json",
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
