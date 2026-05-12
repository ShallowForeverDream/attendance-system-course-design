from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.db import db, init_db, upsert_metric  # noqa: E402
from core.vision import cosine_similarity  # noqa: E402


def evaluate(thresholds: list[float] | None = None) -> dict:
    thresholds = thresholds or [0.60, 0.70, 0.78, 0.80, 0.85, 0.90, 0.95]
    init_db(seed=True)
    with db() as conn:
        rows = conn.execute(
            """SELECT f.id,f.student_id,f.embedding,s.student_no,s.name
               FROM face_samples f JOIN students s ON s.id=f.student_id
               ORDER BY s.student_no,f.id"""
        ).fetchall()
    samples = []
    for r in rows:
        samples.append({**r, "vec": json.loads(r["embedding"])})
    n = len(samples)
    pair_scores = []
    for i in range(n):
        for j in range(i + 1, n):
            score = cosine_similarity(samples[i]["vec"], samples[j]["vec"])
            pair_scores.append({
                "a": samples[i]["id"],
                "b": samples[j]["id"],
                "same": samples[i]["student_id"] == samples[j]["student_id"],
                "score": score,
            })
    by_threshold = []
    for th in thresholds:
        tp = fp = tn = fn = 0
        for p in pair_scores:
            pred = p["score"] >= th
            if pred and p["same"]:
                tp += 1
            elif pred and not p["same"]:
                fp += 1
            elif not pred and p["same"]:
                fn += 1
            else:
                tn += 1
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        far = fp / max(fp + tn, 1)
        by_threshold.append({
            "threshold": th,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "false_accept_rate": round(far, 4),
        })
    student_counts = Counter(s["student_id"] for s in samples)
    report = {
        "samples": n,
        "students": len(student_counts),
        "students_with_multiple_samples": sum(1 for c in student_counts.values() if c >= 2),
        "pairs": len(pair_scores),
        "thresholds": by_threshold,
        "note": "当前 face_data 每人基本只有 1 张，因此同人正样本对较少；该评估主要用于证明阈值、误识别风险和样本覆盖率。",
    }
    # 用默认阈值附近的误接收率作为报告指标。
    default = min(by_threshold, key=lambda x: abs(x["threshold"] - 0.78))
    with db() as conn:
        upsert_metric(conn, "face_pair_false_accept_rate_at_0.78", default["false_accept_rate"], report["pairs"], default)
        upsert_metric(conn, "face_data_students_with_samples", report["students"], report["samples"], report)
    return report


def main() -> None:
    out = ROOT / "docs" / "face_data_evaluation_report.json"
    report = evaluate()
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("report:", out)


if __name__ == "__main__":
    main()
