"""In-process latency and error counters for the API."""
import threading
import time
from collections import deque

import numpy as np


class LatencyMonitor:
    def __init__(self, window=500):
        self.lock = threading.Lock()
        self.total = deque(maxlen=window)
        self.stages = {}
        self.counts = {"requests": 0, "errors": 0, "declined": 0}
        self.started = time.time()

    def record(self, total_ms, timings=None, declined=False, error=False):
        with self.lock:
            self.counts["requests"] += 1
            self.counts["errors"] += bool(error)
            self.counts["declined"] += bool(declined)
            if not error:
                self.total.append(total_ms)
                for k, v in (timings or {}).items():
                    if k.endswith("_ms") and v is not None:
                        self.stages.setdefault(k, deque(maxlen=self.total.maxlen)).append(v)

    @staticmethod
    def _summary(values):
        if not values:
            return None
        a = np.array(values)
        return {"count": len(a), "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
                "p95_ms": float(np.percentile(a, 95)), "max_ms": float(a.max())}

    def snapshot(self):
        with self.lock:
            return {"uptime_s": round(time.time() - self.started, 1), **self.counts,
                    "latency": self._summary(self.total),
                    "stages": {k: self._summary(v) for k, v in self.stages.items()}}

    def prometheus(self):
        s = self.snapshot()
        lines = [f"rag_requests_total {s['requests']}", f"rag_errors_total {s['errors']}",
                 f"rag_declined_total {s['declined']}"]
        if s["latency"]:
            for q in ("p50", "p95"):
                lines.append(f'rag_latency_ms{{quantile="{q}"}} {s["latency"][q + "_ms"]:.1f}')
        return "\n".join(lines) + "\n"
