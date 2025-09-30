from collections import deque
from statistics import median
import time


class Store:
    def __init__(self):
        self.threshold = 0.70
        self.active_version = "v1.0.0"
        self.threshold_last_changed = 0.0
        self.ref_locked = False
        self.ref_set_at = 0.0
        self.pred_feed = deque(
            maxlen=5000
        )  # [{ts, tx_id, score, is_anomaly, latency_ms, amount, merchant}]
        self.lat_hist = deque(maxlen=5000)
        self.metrics_cache = {
            "anomaly_rate": 0.0,
            "latency_p95_ms": 0.0,
            "updated_at": "",
        }

    def add_preds(self, items):
        for it in items:
            self.pred_feed.append(it)
            self.lat_hist.append(it["latency_ms"])

    def anomaly_rate_recent(self, window_sec=600):
        now = time.time()
        sub = [x for x in self.pred_feed if now - x["ts"] <= window_sec]
        if not sub:
            return 0.0
        return sum(1 for x in sub if x["is_anomaly"]) / len(sub)

    def latency_p95(self):
        arr = sorted(self.lat_hist)
        if not arr:
            return 0.0
        k = int(0.95 * (len(arr) - 1))
        return float(arr[k])


store = Store()
