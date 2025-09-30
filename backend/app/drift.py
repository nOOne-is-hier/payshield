import numpy as np


def psi_bucketed(ref, cur, bins=10):
    # 간략 PSI; 분모 0 방지
    ref_hist, _ = np.histogram(ref, bins=bins)
    cur_hist, _ = np.histogram(cur, bins=bins)
    ref_p = np.clip(ref_hist / max(1, ref_hist.sum()), 1e-6, 1)
    cur_p = np.clip(cur_hist / max(1, cur_hist.sum()), 1e-6, 1)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


class DriftMonitor:
    def __init__(self):
        self.ref_amount = None
        self.ref_hour = None
        self.ref_score = None  # 추가

    def set_reference(self, amount_series, hour_series, score_series=None):
        self.ref_amount = None if amount_series is None else np.asarray(amount_series)
        self.ref_hour = None if hour_series is None else np.asarray(hour_series)
        self.ref_score = None if score_series is None else np.asarray(score_series)

    def compute(self, recent_amount, recent_hour, recent_score=None):
        items = []
        if self.ref_amount is not None and len(recent_amount) > 20:
            psi_a = psi_bucketed(self.ref_amount, np.asarray(recent_amount))
            items.append(
                {
                    "feature": "amount",
                    "psi": round(psi_a, 2),
                    "level": (
                        "high"
                        if psi_a > 0.4
                        else ("warn" if psi_a > 0.25 else "normal")
                    ),
                }
            )
        if self.ref_hour is not None and len(recent_hour) > 20:
            psi_h = psi_bucketed(self.ref_hour, np.asarray(recent_hour), bins=6)
            items.append(
                {
                    "feature": "hour",
                    "psi": round(psi_h, 2),
                    "level": (
                        "high"
                        if psi_h > 0.4
                        else ("warn" if psi_h > 0.25 else "normal")
                    ),
                }
            )
        if (
            self.ref_score is not None
            and recent_score is not None
            and len(recent_score) > 20
        ):
            psi_s = psi_bucketed(self.ref_score, np.asarray(recent_score), bins=10)
            items.append(
                {
                    "feature": "score",
                    "psi": round(psi_s, 2),
                    "level": (
                        "high"
                        if psi_s > 0.4
                        else ("warn" if psi_s > 0.25 else "normal")
                    ),
                }
            )
        return items


drift_monitor = DriftMonitor()
