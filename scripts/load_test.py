"""Send questions to a running API one at a time and print the latency summary the API itself keeps.

    uvicorn rag.service.api:app --port 8000 &
    python scripts/load_test.py 30
"""
import json
import sys
import time
from pathlib import Path

import requests

API = "http://localhost:8000"
ROOT = Path(__file__).resolve().parent.parent


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    questions = [json.loads(l)["question"] for l in open(ROOT / "eval" / "questions.jsonl")][:n]
    print(requests.get(f"{API}/health", timeout=10).json())
    wall = []
    for q in questions:
        t0 = time.perf_counter()
        r = requests.post(f"{API}/ask", json={"question": q}, timeout=300)
        r.raise_for_status()
        wall.append((time.perf_counter() - t0) * 1000)
    m = requests.get(f"{API}/metrics", timeout=10).json()
    print(json.dumps(m, indent=2))
    print(f"client-side: median {sorted(wall)[len(wall) // 2]:.0f} ms over {len(wall)} requests")


if __name__ == "__main__":
    main()
