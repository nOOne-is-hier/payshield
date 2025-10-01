# agent/tools.py
from langchain_core.tools import tool
import httpx, os, time

BASE_URL = os.getenv("AIOPS_BASE_URL", "http://127.0.0.1:8000/api")


@tool
def get_metrics(cur_win: int = 300) -> dict:
    """백엔드 /metrics에서 최신 지표 가져오기."""
    r = httpx.get(f"{BASE_URL}/metrics", params={"cur_win": cur_win}, timeout=5)
    r.raise_for_status()
    return r.json()


@tool
def set_threshold(new_threshold: float) -> dict:
    """모델 점수 임곗값 조정 (/model/threshold). 쿨다운/범위 검증은 백엔드가 한다."""
    r = httpx.post(
        f"{BASE_URL}/model/threshold", json={"threshold": new_threshold}, timeout=5
    )
    r.raise_for_status()
    return r.json()


@tool
def trigger_retrain(reason: str = "agent_decision") -> dict:
    """재학습 트리거. 데모에선 /agent/run과 별도로, 재학습 API를 하나 두거나 내부 로깅."""
    # 데모용: /alerts/sms와 동일하게 로그 찍는 stub 또는 별도 /retrain API
    print("[AGENT-RETRAIN]", reason, time.time())
    return {"ok": True, "reason": reason}


@tool
def send_sms(to: str, message: str) -> dict:
    """SMS 발송 스텁(/alerts/sms)"""
    r = httpx.post(
        f"{BASE_URL}/alerts/sms", json={"to": to, "message": message}, timeout=5
    )
    r.raise_for_status()
    return r.json()


@tool
def post_dashboard_summary(text: str, payload: dict = None) -> dict:
    """대시보드 요약을 FastAPI로 전송해서 피드에 적재합니다."""
    try:
        r = httpx.post(
            f"{BASE_URL}/dashboard/summary",
            json={"text": text, "payload": payload or {}},
            timeout=5,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        # 최소한 콘솔에도 남겨 두기
        print("[DASHBOARD-LOCAL]", text)
        print("[DASHBOARD-ERR]", repr(e))
        return {"ok": False, "error": str(e)}


@tool
def feeder_start() -> dict:
    """정상 거래 데이터를 /predict 엔드포인트로 지속적으로 전송하기 시작합니다."""
    r = httpx.post(f"{BASE_URL}/feeder/start", timeout=5)
    r.raise_for_status()
    return r.json()


@tool
def feeder_inject_drift(seconds: float = 30.0) -> dict:
    """지정한 시간 동안 드리프트(이상 거래) 데이터를 /predict 엔드포인트로 주입합니다."""
    r = httpx.post(
        f"{BASE_URL}/feeder/inject_drift", json={"seconds": seconds}, timeout=5
    )
    r.raise_for_status()
    return r.json()


@tool
def feeder_stop() -> dict:
    """정상 feeder 루프를 중지합니다."""
    r = httpx.post(f"{BASE_URL}/feeder/stop", timeout=5)
    r.raise_for_status()
    return r.json()
