from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time, os
import numpy as np

from app.store import store
from app.model_loader import model_svc
from app.drift import drift_monitor
from app.agent_rules import agent_decide
from .streamer import scenario

app = FastAPI(title="Fraud Mini AIOps", version="0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== 초기 모델 로드 ======
MODEL_DIR = os.environ.get("MODEL_DIR", "models/v1.0")
model_svc.load_from_dir(MODEL_DIR)
store.active_version = model_svc.version


# ====== 요청/응답 모델 ======
class Record(BaseModel):
    tx_id: str
    V: List[float] = Field(
        ..., description="길이 30의 리스트"
    )  # V1..V30을 리스트로 받자 (프런트 단순화)
    amount: Optional[float] = None
    merchant: Optional[str] = None
    tx_time: Optional[str] = None  # ISO


class PredictRequest(BaseModel):
    records: List[Record]


class PredictItem(BaseModel):
    tx_id: str
    score: float
    is_anomaly: bool
    model_version: str
    latency_ms: int


class PredictResponse(BaseModel):
    results: List[PredictItem]


# ====== /predict ======
@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest):
    X = np.array([r.V for r in body.records], dtype=np.float32)
    scores, flags, latency_ms = model_svc.predict_batch(X, store.threshold)
    ts = time.time()
    out = []
    for r, s, f in zip(body.records, scores, flags):
        item = {
            "ts": ts,
            "tx_id": r.tx_id,
            "score": float(s),
            "is_anomaly": bool(f),
            "latency_ms": latency_ms,
            "amount": r.amount,
            "merchant": r.merchant,
            "tx_time": r.tx_time,  # ★ 추가
        }
        store.add_preds([item])
        out.append(
            PredictItem(
                tx_id=r.tx_id,
                score=float(s),
                is_anomaly=bool(f),
                model_version=model_svc.version,
                latency_ms=latency_ms,
            )
        )
    return PredictResponse(results=out)


# ====== /metrics ======
@app.get("/metrics")
def metrics(cur_win: int = 300):
    """
    - 최근 cur_win 표본(기본 300)만 사용 (희석 방지)
    - amount / hour / score 모두 PSI 대상 (drift_monitor가 score 지원해야 함)
    - 레퍼런스 자동 세팅(단 1회): ref가 없고 표본 충분 & 최근 AR 매우 낮을 때
      * 여기서는 '세팅'만 한다. 잠금은 /drift/reference/lock 에서만 수행.
    - 부수효과 최소화: 여기서 store.ref_locked는 건드리지 않는다.
    """
    # ── 0) 기본 지표
    ar = store.anomaly_rate_recent(600)  # 최근 10분 이상치율
    p95 = store.latency_p95()  # P95 latency
    model_ver = model_svc.version
    thr = float(store.threshold)

    # ── 1) 최근 표본 슬라이스
    win = max(50, int(cur_win))  # 최소 50 보장
    recent = list(store.pred_feed)[-win:]
    # 표본이 너무 적으면 바로 리턴 (참조 세팅/드리프트 계산 불필요)
    if len(recent) == 0:
        resp = {
            "anomaly_rate": ar,
            "latency_p95_ms": p95,
            "model_version": model_ver,
            "threshold": thr,
            "window_size": 0,
            "ref_set_at": (
                int(store.ref_set_at) if getattr(store, "ref_set_at", 0) else 0
            ),
            "drift": [],
            "score_stats": {
                "p90": 0.0,
                "p95": 0.0,
                "mean": 0.0,
                "near_rate": 0.0,
                "tail_rate": 0.0,
                "delta_p95": 0.0,
                "delta_near": 0.0,
                "delta_tail": 0.0,
            },
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        store.metrics_cache = resp
        return resp

    amounts = [float(x.get("amount") or 0.0) for x in recent]

    # HH 파싱
    hours = []
    for x in recent:
        tt = x.get("tx_time")
        if tt and len(tt) >= 13:
            try:
                hours.append(int(tt[11:13]))
            except Exception:
                pass

    scores = [float(x.get("score") or 0.0) for x in recent]

    # ── 2) 점수 요약(LLM/룰 판단용) ─ 분위/근처/꼬리
    def _q(arr, q):
        if not arr:
            return 0.0
        a = np.asarray(arr, dtype=float)
        a.sort()
        k = int(np.clip(q * (len(a) - 1), 0, len(a) - 1))
        return float(a[k])

    score_p90 = _q(scores, 0.90)
    score_p95 = _q(scores, 0.95)
    score_mean = float(np.mean(scores)) if scores else 0.0
    rate_near = (
        float(np.mean([(thr - 0.02) <= s < (thr + 0.02) for s in scores]))
        if scores
        else 0.0
    )
    rate_tail = float(np.mean([s >= 0.90 for s in scores])) if scores else 0.0

    # ── 3) 레퍼런스 자동 세팅(1회) — 부수효과 최소화
    MIN_REF = max(150, min(300, win))  # 창 크기에 맞춰 유연하게(150~300)
    AR_REF_MAX = 0.002  # 0.2% 미만이면 정상 구간으로 간주
    if (
        getattr(drift_monitor, "ref_amount", None) is None
        and len(amounts) >= MIN_REF
        and ar < AR_REF_MAX
    ):
        # score 분포 요약 델타 계산용 저장(없으면 생성)
        if not hasattr(store, "ref_stats") or store.ref_stats is None:
            store.ref_stats = {}
        store.ref_stats.update(
            {
                "p95": score_p95,
                "near": rate_near,
                "tail": rate_tail,
            }
        )
        # amount / hour / score 참조 세팅
        # (drift_monitor가 set_reference(amounts, hours, scores) 시그니처를 지원해야 함)
        drift_monitor.set_reference(amounts, hours, scores)
        store.ref_set_at = time.time()

    # ── 4) 델타 계산(레퍼런스가 있을 때만)
    ref_stats = getattr(store, "ref_stats", None)
    delta_p95 = (
        float(score_p95 - ref_stats["p95"]) if ref_stats and "p95" in ref_stats else 0.0
    )
    delta_near = (
        float(rate_near - ref_stats["near"])
        if ref_stats and "near" in ref_stats
        else 0.0
    )
    delta_tail = (
        float(rate_tail - ref_stats["tail"])
        if ref_stats and "tail" in ref_stats
        else 0.0
    )

    # ── 5) 드리프트 계산 (amount / hour / score)
    # (drift_monitor.compute(amounts, hours, scores) 시그니처 가정)
    drift = drift_monitor.compute(amounts, hours, scores)

    # ── 6) 캐시 & 반환
    resp = {
        "anomaly_rate": ar,
        "latency_p95_ms": p95,
        "model_version": model_ver,
        "threshold": thr,
        "window_size": len(recent),
        "ref_set_at": int(store.ref_set_at) if getattr(store, "ref_set_at", 0) else 0,
        "drift": drift,
        "score_stats": {
            "p90": round(score_p90, 4),
            "p95": round(score_p95, 4),
            "mean": round(score_mean, 4),
            "near_rate": round(rate_near, 4),
            "tail_rate": round(rate_tail, 4),
            "delta_p95": round(delta_p95, 4),
            "delta_near": round(delta_near, 4),
            "delta_tail": round(delta_tail, 4),
        },
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    store.metrics_cache = resp
    return resp


@app.post("/drift/reference/reset")
def drift_ref_reset():
    drift_monitor.ref_amount = None
    drift_monitor.ref_hour = None
    store.ref_locked = False
    store.ref_set_at = 0.0
    return {"ok": True}


@app.post("/drift/reference/lock")
def drift_ref_lock():
    store.ref_locked = True
    return {"ok": True}


# ====== /alerts ======
@app.get("/alerts")
def alerts(limit: int = 50):
    items = []
    for x in reversed(store.pred_feed):
        if x["is_anomaly"]:
            items.append(
                {
                    "tx_id": x["tx_id"],
                    "time": int(x["ts"]),
                    "amount": x.get("amount"),
                    "merchant": x.get("merchant"),
                    "score": x["score"],
                    "level": "critical" if x["score"] > 0.95 else "warn",
                }
            )
        if len(items) >= limit:
            break
    return {"items": items}


# ====== /model/info & /model/threshold ======
@app.get("/model/info")
def model_info():
    return {
        "active_version": model_svc.version,
        "threshold": store.threshold,
        "range": [0.55, 0.85],
        "last_changed_at": store.threshold_last_changed,
    }


class ThrBody(BaseModel):
    threshold: float


@app.post("/model/threshold")
def model_threshold(body: ThrBody):
    now = time.time()
    if not (0.55 <= body.threshold <= 0.85):
        return {"ok": False, "error": "out_of_range"}
    if now - store.threshold_last_changed < 300:
        return {"ok": False, "error": "cooldown"}
    store.threshold = float(body.threshold)
    store.threshold_last_changed = now
    return {"ok": True, "threshold": store.threshold}


# ====== /alerts/sms (stub) ======
class SmsBody(BaseModel):
    to: str
    message: str


@app.post("/alerts/sms")
def sms_stub(body: SmsBody):
    print("[SMS]", body.to, body.message[:120])
    return {"status": "sent"}


# ====== /agent/run ======
_guard = {
    "base_ar": 0.001,
    "thr_changed_at": 0.0,
    "retrain_at": 0.0,
    "threshold": 0.70,
    "sms_to": "+82-10-0000-0000",
    "hi_count": 0,  # ★ 추가
}


@app.post("/drift/reset")
def drift_reset():
    drift_monitor.set_reference(None, None)  # 구현에 맞게 초기화
    drift_monitor.ref_amount = None
    drift_monitor.ref_hour = None
    return {"ok": True}


@app.post("/agent/run")
def agent_run():
    m = metrics()  # 최신 메트릭 계산
    recent = alerts(20)["items"]
    _guard["threshold"] = store.threshold
    decision = agent_decide(m, recent, _guard)
    # 액션 실행
    for act in decision["actions"]:
        if act["type"] == "THRESHOLD":
            store.threshold = act["value"]
            store.threshold_last_changed = time.time()
            _guard["thr_changed_at"] = store.threshold_last_changed
        elif act["type"] == "SMS":
            print("[SMS]", act.get("to"), act.get("message", "")[:120])
        elif act["type"] == "RETRAIN":
            # 초간단 동기 재학습 자리: 이번 스코프에선 생략/프린트
            _guard["retrain_at"] = time.time()
            print("[RETRAIN] (demo) retrain triggered")
    return {"summary": decision["summary"], "actions": decision["actions"]}


class ScenarioBody(BaseModel):
    normal_path: str = "data/transactions_normal.csv"
    drift_path: str = "data/transactions_drift.csv"
    batch: int = 64
    sleep_sec: float = 0.5


@app.post("/scenario/start")
def scenario_start(b: ScenarioBody):
    ok = scenario.start(b.normal_path, b.drift_path, b.batch, b.sleep_sec)
    return {"ok": ok}


@app.get("/scenario/status")
def scenario_status():
    from .streamer import streamer

    return {
        "scenario_running": scenario.running,
        "feed_len": len(store.pred_feed),
        "threshold": store.threshold,
        "active_version": model_svc.version,
    }
