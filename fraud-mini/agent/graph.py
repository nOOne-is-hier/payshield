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
)

# (NEW) OpenAI LLM - 요약 강화용
# 환경변수 OPENAI_API_KEY 필요
LLM_ENABLED = os.getenv("AIOPS_LLM_ENABLED", "1") == "1"
LLM_MODEL = os.getenv("AIOPS_LLM_MODEL", "gpt-4o-mini")
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
    p95 = sstats.get("p95", 0.0)
    dp95 = sstats.get("delta_p95", 0.0)

    state.summary = (
        f"win={m.get('window_size')} thr={m.get('threshold')} "
        f"amount.psi={amount_psi:.2f} score.psi={score_psi:.2f} "
        f"p95={p95:.3f} Δp95={dp95:.3f} hi_count={state.hi_count}"
    )
    # 대시보드 요약 전송(원하면 주석 해제)
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
        sys = (
            "너는 AIOps 드리프트 감지/조치 에이전트다. "
            "입력으로 /metrics JSON과 현재 hi_count를 받아 운영자가 보기 좋은 2~3줄 요약을 생성하라. "
            "핵심 수치(amount.psi, score.psi, score_stats.p95/Δp95/near_rate, threshold, window_size)를 포함하고 "
            "변화 원인을 간략히 추정하라(예: 금액 상위 꼬리 비대, p95 +0.12). "
            "액션(임곗값 조정/재학습/SMS)은 네가 수행하지 않는다. 요약만 출력하라."
        )
        user = "state.hi_count=" + str(state.hi_count) + "\n" "metrics JSON:\n" + str(m)
        msgs = [SystemMessage(content=sys), HumanMessage(content=user)]
        ai = _llm.invoke(msgs)
        if ai and getattr(ai, "content", None):
            # 본문에 LLM 요약 1~2줄만 첨부 (너무 길면 240자 제한)
            llm_txt = ai.content.strip().replace("\n", " ")
            if len(llm_txt) > 240:
                llm_txt = llm_txt[:240] + "..."
            state.summary = (state.summary or "") + f" | LLM: {llm_txt}"
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
        state.last_retrain_at = now
        state.actions.append({"type": "RETRAIN", "reason": "psi_high_persistent"})

    return state


# ────────────────────────────────────────────────────────
# 그래프 컴파일
# ────────────────────────────────────────────────────────
def build_graph():
    sg = StateGraph(AgentState)
    sg.add_node("ingest", ingest)
    sg.add_node("summarize", summarize)
    # (NEW) LLM 요약 강화 노드
    sg.add_node("llm_summarize", llm_summarize)
    sg.add_node("decide_and_act", decide_and_act)
    sg.set_entry_point("ingest")
    sg.add_edge("ingest", "summarize")
    sg.add_edge("summarize", "llm_summarize")  # ← 요약 다음에 LLM
    sg.add_edge("llm_summarize", "decide_and_act")
    sg.add_edge("decide_and_act", END)
    return sg.compile()


graph = build_graph()
