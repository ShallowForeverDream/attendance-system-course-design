from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import BASE_DIR, UPLOAD_DIR  # noqa: E402
from core.db import db, init_db, now_iso, upsert_metric  # noqa: E402
from core.vision import read_image_path  # noqa: E402


def make_collage(count: int = 10, size: int = 260, output_name: str | None = None) -> dict:
    init_db(seed=True)
    with db() as conn:
        # 从已导入的人脸库中按质量分自动选择样本，避免在源码中硬编码真实学号/姓名。
        # 组员 clone 后只需把老师授权的 face_data 放到项目同级目录并运行 prepare_demo.py，
        # 即可生成稳定的 10 人/50 人演示合照和自测报告。
        raw_rows = conn.execute(
            """SELECT s.student_no,s.name,f.image_path,f.quality
               FROM face_samples f JOIN students s ON s.id=f.student_id
               ORDER BY f.quality DESC""",
        ).fetchall()
        # 合照压力图强调“人数规模”，因此同一学生只取质量最高的一张样本，
        # 避免多样本学生在 Top50 中重复出现，导致 50 张 tile 但唯一学生少于 50。
        seen = set()
        rows = []
        for row in raw_rows:
            if row["student_no"] in seen:
                continue
            seen.add(row["student_no"])
            rows.append(row)
            if len(rows) >= count:
                break
    if not rows:
        raise SystemExit("没有可用人脸样本，请先运行 scripts/import_face_data.py")
    cols = 10 if count > 10 else 5
    rows_n = int(np.ceil(len(rows) / cols))
    label_h = 64
    margin = 28
    tile_w = size + margin * 2
    tile_h = size + margin * 2 + label_h
    canvas = np.full((rows_n * tile_h, cols * tile_w, 3), 248, dtype=np.uint8)
    selected = []
    for idx, row in enumerate(rows):
        img = read_image_path(BASE_DIR / row["image_path"])
        # 使用已入库的人脸裁剪图直接拼接，保证演示合照与人脸库特征一致；
        # 这张图用于“多人合照流程演示/压力测试”，真实活动合照仍可在页面上传。
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        r, c = divmod(idx, cols)
        y, x = r * tile_h + margin, c * tile_w + margin
        cv2.rectangle(canvas, (x - 10, y - 10), (x + size + 10, y + size + 10), (255, 255, 255), -1)
        canvas[y:y + size, x:x + size] = img
        label = f"{row['student_no']} {row['name']}"
        label_y = y + size + 42
        selected.append({"student_no": row["student_no"], "name": row["name"], "quality": row["quality"]})
    # 用 PIL 画中文标签，避免 OpenCV putText 变成 ?????；同时写入纯数字学号供演示校验。
    pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    font_big = font_small = None
    for fp in font_paths:
        if Path(fp).exists():
            font_big = ImageFont.truetype(fp, 27)
            font_small = ImageFont.truetype(fp, 23)
            break
    if font_big is None:
        font_big = font_small = ImageFont.load_default()
    for idx, row in enumerate(rows):
        r, c = divmod(idx, cols)
        y, x = r * tile_h + margin, c * tile_w + margin
        label_y = y + size + 12
        draw.text((x + 4, label_y), str(row["student_no"]), fill=(20, 30, 45), font=font_big)
        draw.text((x + 4, label_y + 30), str(row["name"])[:8], fill=(20, 30, 45), font=font_small)
    canvas = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    out = UPLOAD_DIR / (output_name or f"demo_collage_{now_iso().replace(':','').replace(' ','_')}.png")
    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise SystemExit("collage encode failed")
    buf.tofile(str(out))
    report = {"count": len(selected), "image_path": str(out.relative_to(BASE_DIR)), "students": selected}
    with db() as conn:
        # 只记录正式 10 人演示指标，避免调试 count>10 时覆盖验收清单中的稳定合照指标。
        if count == 10:
            upsert_metric(conn, "demo_collage_students", len(selected), len(selected), report)
        else:
            upsert_metric(conn, f"demo_collage_students_{count}", len(selected), len(selected), report)
    return report


def main() -> None:
    report = make_collage()
    out_json = ROOT / "docs" / "demo_collage_report.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("report:", out_json)


if __name__ == "__main__":
    main()
