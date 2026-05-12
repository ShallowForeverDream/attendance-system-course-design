from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app  # noqa: E402


def main() -> None:
    client = app.test_client()
    r = client.get("/api/me")
    assert r.status_code == 200
    r = client.post("/api/login", json={"username": "teacher", "password": "teacher123"})
    assert r.status_code == 200, r.get_data(as_text=True)
    assert r.get_json()["ok"]
    r = client.get("/api/summary")
    assert r.status_code == 200, r.get_data(as_text=True)
    print("summary:", json.dumps(r.get_json(), ensure_ascii=False))
    r = client.get("/api/students")
    assert r.status_code == 200, r.get_data(as_text=True)
    print("students:", len(r.get_json()["students"]))
    r = client.get("/api/attendance/challenge")
    assert r.status_code == 200, r.get_data(as_text=True)
    print("challenge:", r.get_json()["challenge"])
    r = client.post("/api/liveness/self-test-sample", json={})
    assert r.status_code == 200, r.get_data(as_text=True)
    liveness = r.get_json()["liveness"]
    assert liveness["pass"] is False, "静态样本攻击应被拒绝"
    print("sample_attack:", json.dumps({"pass": liveness["pass"], "reason": liveness["reason"]}, ensure_ascii=False))
    r = client.get("/api/security/challenge-randomness")
    assert r.status_code == 200, r.get_data(as_text=True)
    print("challenge_randomness:", r.get_json()["unique_pairs"])
    print("SMOKE TEST OK")


if __name__ == "__main__":
    main()
