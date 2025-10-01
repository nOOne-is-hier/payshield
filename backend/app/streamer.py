# app/streamer.py
import threading, time, csv, os, requests
import numpy as np
from typing import Optional
from .store import store
from .model_loader import model_svc


class Streamer:
    def __init__(self):
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.batch = 64
        self.sleep_sec = 0.5

    def _pump_csv(self, path, deadline_epoch):
        # CSV: tx_id, tx_time, merchant, amount, V1..V30
        if not os.path.exists(path):
            return
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            buf = []
            for row in reader:
                if not self.running or time.time() >= deadline_epoch:
                    break
                V = [float(row.get(f"V{i}", 0) or 0) for i in range(1, 31)]
                X = np.array([V], dtype=np.float32)
                scores, flags, latency_ms = model_svc.predict_batch(X, store.threshold)
                store.add_preds(
                    [
                        {
                            "ts": time.time(),
                            "tx_id": row.get("tx_id") or f"tx_{int(time.time()*1000)}",
                            "score": float(scores[0]),
                            "is_anomaly": bool(flags[0]),
                            "latency_ms": latency_ms,
                            "amount": float(row.get("amount") or 0),
                            "merchant": row.get("merchant"),
                            "tx_time": row.get("tx_time"),
                        }
                    ]
                )
                buf.append(1)
                if len(buf) >= self.batch:
                    time.sleep(self.sleep_sec)
                    buf.clear()

    def stream_for(
        self, path: str, seconds: float, batch: int = 64, sleep_sec: float = 0.5
    ):
        self.batch = batch
        self.sleep_sec = sleep_sec
        deadline = time.time() + seconds
        self._pump_csv(path, deadline)


streamer = Streamer()


# ====== 시나리오 런너 ======
class ScenarioRunner:
    def __init__(self, base_url="http://127.0.0.1:8000/api"):
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.base_url = base_url

    # 내부 유틸: /metrics 폴링로 ref_set_at 확인
    def _wait_ref(self, timeout_s=5.0, interval_s=0.5):
        deadline = time.time() + timeout_s
        last = {}
        while time.time() < deadline:
            try:
                r = requests.get(f"{self.base_url}/metrics", timeout=3).json()
                last = r
                if r.get("ref_set_at", 0) != 0:
                    return True, r
            except Exception:
                pass
            time.sleep(interval_s)
        return False, last

    def start(
        self, normal_path: str, drift_path: str, batch: int = 64, sleep_sec: float = 0.5
    ):
        if self.running:
            return False
        self.running = True
        self.thread = threading.Thread(
            target=self._run,
            args=(normal_path, drift_path, batch, sleep_sec),
            daemon=True,
        )
        self.thread.start()
        return True

    def _run(self, normal_path, drift_path, batch, sleep_sec):
        try:
            # 0) 기준 리셋
            try:
                requests.post(f"{self.base_url}/drift/reference/reset", timeout=3)
            except Exception:
                pass

            # 1) 정상 30초 → 기준 후보 데이터 쌓기
            streamer.running = True
            streamer.stream_for(
                normal_path, seconds=30, batch=batch, sleep_sec=sleep_sec
            )

            # 1-1) /metrics 폴링으로 ref_set_at 생길 때까지 대기
            ok_ref, r = self._wait_ref(timeout_s=6.0, interval_s=0.5)
            print("[scenario] metrics(after-normal) =", r)

            # 1-2) 기준 '잠금'은 레퍼런스가 잡힌 뒤에만
            if ok_ref:
                try:
                    requests.post(f"{self.base_url}/drift/reference/lock", timeout=3)
                except Exception:
                    pass
            else:
                print("[scenario] WARN: reference not set; continuing without lock")

            # 2) 드리프트 40초 (길이 늘려 희석 방지)
            streamer.stream_for(
                drift_path, seconds=40, batch=batch, sleep_sec=sleep_sec
            )

            # 2-1) 드리프트 직후 /metrics 한번 더 호출 (PSI 반영/hi_count 누적)
            try:
                r_drift = requests.get(f"{self.base_url}/metrics", timeout=3).json()
                print("[scenario] metrics(after-drift-1) =", r_drift)
            except Exception as e:
                print("[scenario] metrics(after-drift-1) error:", e)

            # 2-2) 에이전트 1차: THRESHOLD 조정 유도
            try:
                r1 = requests.post(f"{self.base_url}/agent/run", timeout=5).json()
                print("[agent-1]", r1)
            except Exception as e:
                print("[agent-1] error:", e)
            time.sleep(2)

            # 2-3) 에이전트 2차: RETRAIN 트리거 시도(최대 5회, 매회 전 metrics로 상태 업데이트)
            for i in range(5):
                try:
                    _ = requests.get(f"{self.base_url}/metrics", timeout=3).json()
                except Exception:
                    pass
                try:
                    r2 = requests.post(f"{self.base_url}/agent/run", timeout=5).json()
                    print(f"[agent-2.{i+1}]", r2)
                    acts = r2.get("actions", [])
                    if any(a.get("type") == "RETRAIN" for a in acts):
                        break
                except Exception as e:
                    print(f"[agent-2.{i+1}] error:", e)
                time.sleep(2)

            # 3) 재학습 후 효과 관찰: 드리프트 20초 더 + 상태 확인
            streamer.stream_for(
                drift_path, seconds=20, batch=batch, sleep_sec=sleep_sec
            )
            try:
                r_drift2 = requests.get(f"{self.base_url}/metrics", timeout=3).json()
                print("[scenario] metrics(after-drift-2) =", r_drift2)
            except Exception as e:
                print("[scenario] metrics(after-drift-2) error:", e)

        finally:
            streamer.running = False
            self.running = False


scenario = ScenarioRunner()
