# app/aiops_agent.py
import asyncio, threading
from agent.graph import graph, AgentState

_lock = threading.Lock()
_state = AgentState()


def tick() -> AgentState:
    global _state
    with _lock:
        out = graph.invoke(_state)
        _state = AgentState(**out) if isinstance(out, dict) else out
        return _state


async def agent_worker(
    interval_sec: int = 10,
    stop_event: asyncio.Event | None = None,
    initial_delay: float = 3.0,
):
    await asyncio.sleep(initial_delay)
    if stop_event is None:
        stop_event = asyncio.Event()

    backoff = 1.0
    while not stop_event.is_set():
        try:
            # ⬇️ 동기 tick을 이벤트 루프 밖 스레드로 이동
            await asyncio.to_thread(tick)
            backoff = 1.0
        except Exception as e:
            print("[agent-worker] warn:", repr(e))
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=backoff)
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, interval_sec)
            continue

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_sec)
        except asyncio.TimeoutError:
            pass
