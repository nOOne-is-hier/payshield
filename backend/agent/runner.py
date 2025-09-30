# agent/runner.py
import asyncio
import time
import json
from agent.graph import graph, AgentState


async def run_agent(interval_sec: int = 10):
    state = AgentState()
    while True:
        try:
            # graph 실행
            state = graph.invoke(state)

            # state 안의 정보 로깅 (필요한 속성에 맞게 조정)
            print("=" * 60)
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Agent tick")
            if hasattr(state, "metrics"):
                print(
                    "[metrics]", json.dumps(state.metrics, indent=2, ensure_ascii=False)
                )
            if hasattr(state, "summary"):
                print("[summary]", state.summary)
            if hasattr(state, "actions") and state.actions:
                print(
                    "[actions]", json.dumps(state.actions, indent=2, ensure_ascii=False)
                )
            print("=" * 60)

        except Exception as e:
            print("[runner-error]", e)

        await asyncio.sleep(interval_sec)


if __name__ == "__main__":
    asyncio.run(run_agent(10))
