from __future__ import annotations
import os
import time
from typing import Any, Dict, Optional, List

from pydantic import BaseModel
from langgraph.graph import StateGraph, END

# 툴: httpx로 FastAPI 호출 (이미 구현해둔 agent/tools.py 사용)
from agent.tools import (
    get_metrics,
    set_threshold,
    trigger_retrain,
    send_sms,
    post_dashboard_summary,
    feeder_start as tl_feeder_start,
    feeder_inject_drift as tl_feeder_inject,
    feeder_stop,
)


DEMO = os.getenv("AIOPS_DEMO", "1") == "1"
AIOPS_DEMO = os.getenv("AIOPS_DEMO", "0") == "1"  # 데모에서만 드리프트 주입 허용
_FEEDER_STARTED = False  # 프로세스 내 1회 보장
_LAST_DRIFT_AT = 0.0
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY, "OPENAI_API_KEY missing"


# (NEW) OpenAI LLM - 요약 강화용
# 환경변수 OPENAI_API_KEY 필요
LLM_ENABLED = os.getenv("AIOPS_LLM_ENABLED", "1") == "1"
LLM_MODEL = os.getenv("AIOPS_LLM_MODEL", "gpt-4o-mini")
RICH_SUMMARY = os.getenv("AIOPS_SUMMARY_RICH", "1") == "1"
if LLM_ENABLED:
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage

        _llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    except Exception as _e:
        # LLM 로딩 실패 시 자동 비활성화
        print("[graph] LLM init failed -> fallback to rule-only:", repr(_e))
        LLM_ENABLED = False


# ────────────────────────────────────────────────────────
# 상태 정의
# ────────────────────────────────────────────────────────
class AgentState(BaseModel):
    metrics: Optional[Dict[str, Any]] = None
    summary: Optional[str] = None
    actions: List[Dict[str, Any]] = []
    hi_count: int = 0
    last_threshold_change_at: float = 0.0
    last_retrain_at: float = 0.0


# ────────────────────────────────────────────────────────
# 노드 1) 수집: /metrics 호출 + hi_count 갱신
# ────────────────────────────────────────────────────────
def ingest(state: AgentState) -> AgentState:
    m = get_metrics.invoke({"cur_win": 300})  # dict 반환
    drift = m.get("drift", []) if isinstance(m, dict) else []
    high = any(d.get("level") == "high" for d in drift)
    state.hi_count = state.hi_count + 1 if high else 0
    state.metrics = m
    return state


# ────────────────────────────────────────────────────────
# 노드 2) 요약: 운영자가 보기 쉬운 한 줄 요약 생성
# ────────────────────────────────────────────────────────
def summarize(state: AgentState) -> AgentState:
    m = state.metrics or {}
    drift = m.get("drift", [])
    amount_psi = next(
        (d.get("psi", 0.0) for d in drift if d.get("feature") == "amount"), 0.0
    )
    score_psi = next(
        (d.get("psi", 0.0) for d in drift if d.get("feature") == "score"), 0.0
    )

    sstats = m.get("score_stats", {}) or {}
    score_p95 = sstats.get("p95", 0.0)
    d_score_p95 = sstats.get("delta_p95", 0.0)
    lat_p95_ms = m.get("latency_p95_ms", 0.0)

    near = sstats.get("near_rate", 0.0)
    tail = sstats.get("tail_rate", 0.0)
    ar = m.get("anomaly_rate", 0.0)
    state.summary = (
        f"win={m.get('window_size')} thr={m.get('threshold')} "
        f"AR={ar*100:.2f}% "
        f"amount.psi={amount_psi:.2f} score.psi={score_psi:.2f} "
        f"score.p95={score_p95:.3f} Δscore.p95={d_score_p95:.3f} near={near:.3f} tail={tail:.3f} "
        f"lat.p95_ms={lat_p95_ms:.0f} hi_count={state.hi_count}"
    )
    post_dashboard_summary.invoke({"text": state.summary, "payload": m})
    return state


# ────────────────────────────────────────────────────────
# (NEW) 노드 2.5) LLM 요약 강화: 운영자 친화 2~3줄 자연어 보강
#  - 액션은 수행하지 않음(툴콜 없음)
#  - 실패해도 파이프라인은 계속 진행
# ────────────────────────────────────────────────────────
def llm_summarize(state: AgentState) -> AgentState:
    if not LLM_ENABLED:
        return state

    try:
        m = state.metrics or {}
        if RICH_SUMMARY:
            sys = (
                "너는 AIOps 드리프트 요약 작성자다. 입력(/metrics JSON, hi_count)을 바탕으로 "
                "**불릿 4~6줄의 리치 요약**을 한국어로 출력하라. 마크다운/이모지 허용. 액션은 수행하지 말고 요약만 출력.\n"
                "- 1줄: 상태 뱃지(✅ 정상/⚠️ 경고/🚨 고위험) + 한줄 총평\n"
                "- 2줄: 핵심 수치: threshold, window_size, anomaly_rate(%), latency_p95_ms(ms)\n"
                "- 3줄: PSI 표(이모지 레벨): amount.psi, score.psi (예: amount.psi=0.68 ⚠️, score.psi=2.26 🚨)\n"
                "- 4줄: 점수 분포: score.p95, Δscore.p95, near_rate, tail_rate (증감 화살표 ↑↓→)\n"
                "- 5줄: 변화 원인 가설 1줄 (예: 상위 꼬리 비대/스파이크/주기성 등)\n"
                "- 6줄: 권고 1줄 (예: 임계 보정 유지/레퍼런스 재설정 검토/재학습 완료 확인 등)\n"
                "숫자는 소수 2자리, 퍼센트는 ×100 후 소수 2자리. 너무 길면 12줄 이내."
            )
        else:
            sys = (
                "너는 AIOps 드리프트 요약 작성자다. 2~3줄의 간결 요약만 출력하라. "
                "amount.psi, score.psi, score_stats.p95/Δp95/near_rate, threshold, window_size, latency_p95_ms 포함."
            )
        user = "state.hi_count=" + str(state.hi_count) + "\n" "metrics JSON:\n" + str(m)
        msgs = [SystemMessage(content=sys), HumanMessage(content=user)]
        ai = _llm.invoke(msgs)
        if ai and getattr(ai, "content", None):
            llm_txt = ai.content.strip()
            # 리치 모드면 줄바꿈 유지, 아니면 한 줄로 축약
            if not RICH_SUMMARY:
                llm_txt = llm_txt.replace("\n", " ")
            state.summary = (state.summary or "") + (
                f"\n{llm_txt}" if RICH_SUMMARY else f" | LLM: {llm_txt}"
            )
            post_dashboard_summary.invoke({"text": state.summary})
    except Exception as e:
        # LLM 실패는 무시하고 룰 계속
        print("[graph] llm_summarize error:", repr(e))
    return state


# ────────────────────────────────────────────────────────
# 노드 3) 결정 & 실행: 임곗값 조정 / 재학습 / SMS
# ────────────────────────────────────────────────────────
def decide_and_act(state: AgentState) -> AgentState:
    state.actions = []  # 이번 턴의 실행 내역 기록

    m = state.metrics or {}
    drift = m.get("drift", [])
    thr = float(m.get("threshold", 0.70))
    now = time.time()

    amount_psi = next(
        (d.get("psi", 0.0) for d in drift if d.get("feature") == "amount"), 0.0
    )
    score_psi = next(
        (d.get("psi", 0.0) for d in drift if d.get("feature") == "score"), 0.0
    )
    psi_high = (amount_psi >= 0.4) or (score_psi >= 0.4)

    # 1) PSI high면 임곗값 +0.02 (쿨다운 300s)
    thr_cool_ok = (now - state.last_threshold_change_at) >= 300.0
    if psi_high and thr_cool_ok:
        new_thr = min(thr + 0.02, 0.85)
        if new_thr > thr:
            set_threshold.invoke({"new_threshold": new_thr})
            state.last_threshold_change_at = now
            state.actions.append(
                {"type": "THRESHOLD", "value": new_thr, "reason": "psi_high"}
            )
            post_dashboard_summary.invoke(
                {"text": f"Threshold {thr:.2f} → {new_thr:.2f} (PSI high)"}
            )

    # 2) PSI high 3회 연속이면 재학습 + SMS (쿨다운 600s)
    rt_cool_ok = (now - state.last_retrain_at) >= 600.0
    if psi_high and state.hi_count >= 3 and rt_cool_ok:
        trigger_retrain.invoke({"reason": "psi_high_persistent"})
        send_sms.invoke(
            {
                "to": "+82-10-0000-0000",
                "message": "[Fraud] PSI high persists; retrain triggered.",
            }
        )
        # 🔻 시연용: 재학습과 동시에 feeder 정지
        if DEMO:
            try:
                feeder_stop.invoke({})
                state.actions.append({"type": "FEEDER_STOP", "reason": "retrain"})
                post_dashboard_summary.invoke(
                    {"text": "[FEEDER] stopped (due to retrain)"}
                )
            except Exception as e:
                post_dashboard_summary.invoke(
                    {"text": f"[FEEDER] stop failed: {repr(e)}"}
                )

        state.last_retrain_at = now
        state.actions.append({"type": "RETRAIN", "reason": "psi_high_persistent"})

    return state


# ────────────────────────────────────────────────────────
# (NEW) 노드 0) feeder 제어: ref 없을 때 정상 피드 1회, 데모일 때만 드리프트 주입
# ────────────────────────────────────────────────────────
def feeder_control(state: AgentState) -> AgentState:
    global _FEEDER_STARTED, _LAST_DRIFT_AT
    m = state.metrics or {}

    # 0) ref 없고 표본 부족/AR 낮으면 정상 피드 1회 시작
    if not _FEEDER_STARTED:
        ref_set = int(m.get("ref_set_at") or 0)
        if (ref_set == 0) and (m.get("window_size", 0) < 150):
            try:
                tl_feeder_start.invoke({})
                _FEEDER_STARTED = True
                post_dashboard_summary.invoke(
                    {"text": "[FEEDER] normal feed started (auto)"}
                )
            except Exception as e:
                print("[feeder_control] start_normal failed:", repr(e))

    # 1) (옵션) 데모 모드에서만 드리프트 간헐 주입 (과도 호출 방지)
    if AIOPS_DEMO:
        now = time.time()
        psi_high = False
        for d in m.get("drift", []):
            if d.get("level") == "high":
                psi_high = True
                break
        # 데모 체험용: 한동안 잠잠하면 40초 주입
        if (not psi_high) and (now - _LAST_DRIFT_AT) > 180:
            try:
                tl_feeder_inject.invoke({"seconds": 40.0})
                _LAST_DRIFT_AT = now
                post_dashboard_summary.invoke(
                    {"text": "[FEEDER] drift injected 40s (demo)"}
                )
            except Exception as e:
                print("[feeder_control] inject_drift failed:", repr(e))

    return state


# ────────────────────────────────────────────────────────
# 그래프 컴파일
# ────────────────────────────────────────────────────────
def build_graph():
    sg = StateGraph(AgentState)
    sg.add_node("ingest", ingest)
    sg.add_node("feeder_control", feeder_control)  # ← 추가
    sg.add_node("summarize", summarize)
    sg.add_node("llm_summarize", llm_summarize)
    sg.add_node("decide_and_act", decide_and_act)

    sg.set_entry_point("ingest")
    sg.add_edge("ingest", "feeder_control")  # ingest 다음 feeder 제어
    sg.add_edge("feeder_control", "summarize")
    sg.add_edge("summarize", "llm_summarize")
    sg.add_edge("llm_summarize", "decide_and_act")
    sg.add_edge("decide_and_act", END)
    return sg.compile()


graph = build_graph()
