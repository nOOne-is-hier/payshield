# app/feeder.py
import threading, time, csv, os, requests
from typing import Optional


class Feeder:
    def __init__(self, base_url="http://127.0.0.1:8000"):
        self.base_url = base_url
        self._normal_thread: Optional[threading.Thread] = None
        self._drift_thread: Optional[threading.Thread] = None
        self._normal_running = False
        self._drift_running = False
        self._lock = threading.Lock()

        # 기본 경로(환경변수로 바꿀 수 있음)
        self.normal_path = os.getenv("AIOPS_NORMAL_CSV", "data/transactions_normal.csv")
        self.drift_path = os.getenv("AIOPS_DRIFT_CSV", "data/transactions_drift.csv")
        self.batch = int(os.getenv("AIOPS_FEED_BATCH", "64"))
        self.sleep_sec = float(os.getenv("AIOPS_FEED_SLEEP", "0.5"))

    def _post_predict(self, batch):
        try:
            requests.post(
                f"{self.base_url}/predict",
                json={"records": batch},
                timeout=5,
            )
        except Exception as e:
            print("[feeder] predict post error:", repr(e))

    def _pump_csv_http(self, path, stop_predicate):
        if not os.path.exists(path):
            print("[feeder] CSV not found:", path)
            return
        batch = []
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if stop_predicate():
                    break
                V = [float(row.get(f"V{i}", 0) or 0) for i in range(1, 31)]
                batch.append(
                    {
                        "tx_id": row.get("tx_id") or f"tx_{int(time.time()*1000)}",
                        "V": V,
                        "amount": float(row.get("amount") or 0),
                        "merchant": row.get("merchant"),
                        "tx_time": row.get("tx_time"),
                    }
                )
                if len(batch) >= self.batch:
                    self._post_predict(batch)
                    batch.clear()
                    time.sleep(self.sleep_sec)
        if batch and not stop_predicate():
            self._post_predict(batch)

    # ── 정상 루프: 파일 끝나면 다시 처음부터(무한 루프)
    def _normal_loop(self):
        while self._normal_running:
            self._pump_csv_http(self.normal_path, lambda: not self._normal_running)

    def start_normal(self):
        with self._lock:
            if self._normal_running:
                return False
            self._normal_running = True
            self._normal_thread = threading.Thread(
                target=self._normal_loop, daemon=True
            )
            self._normal_thread.start()
            print("[feeder] normal started")
            return True

    def stop_normal(self):
        with self._lock:
            self._normal_running = False
            return True

    # ── 드리프트 주입: N초 동안 드리프트 CSV만 밀고 종료
    def inject_drift(self, seconds: float = 30.0):
        def _run():
            end = time.time() + seconds
            while time.time() < end and self._drift_running:
                self._pump_csv_http(
                    self.drift_path,
                    lambda: not self._drift_running or time.time() >= end,
                )
            print("[feeder] drift injection finished")

        with self._lock:
            if self._drift_running:
                return False
            # 드리프트 중엔 정상 멈춤
            normal_was_running = self._normal_running
            self._normal_running = False

            self._drift_running = True
            self._drift_thread = threading.Thread(target=_run, daemon=True)
            self._drift_thread.start()
            print(f"[feeder] drift injecting for {seconds}s")

            # 드리프트 끝나면 정상 재개(백그라운드에서 확인)
            def _wait_and_resume():
                if self._drift_thread:
                    self._drift_thread.join()
                with self._lock:
                    self._drift_running = False
                    if normal_was_running:
                        self._normal_running = True
                        if (
                            self._normal_thread is None
                            or not self._normal_thread.is_alive()
                        ):
                            self._normal_thread = threading.Thread(
                                target=self._normal_loop, daemon=True
                            )
                            self._normal_thread.start()
                        print("[feeder] normal resumed")

            threading.Thread(target=_wait_and_resume, daemon=True).start()
            return True

    def status(self):
        with self._lock:
            return {
                "normal_running": self._normal_running,
                "drift_running": self._drift_running,
                "normal_path": self.normal_path,
                "drift_path": self.drift_path,
                "batch": self.batch,
                "sleep_sec": self.sleep_sec,
            }


feeder = Feeder()
