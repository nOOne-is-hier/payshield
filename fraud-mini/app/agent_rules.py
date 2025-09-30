# app/agent_rules.py
import time


def agent_decide(metrics, recent_alerts, guard):
    actions, summary = [], []
    ar = float(metrics.get("anomaly_rate", 0.0))
    drift = metrics.get("drift", [])
    hi = [d for d in drift if d.get("level") == "high"]

    now = time.time()
    thr = float(guard.get("threshold", 0.70))
    thr_ok = (now - float(guard.get("thr_changed_at", 0.0))) >= 300.0
    rt_ok = (now - float(guard.get("retrain_at", 0.0))) >= 600.0

    # 지속성 카운트
    guard["hi_count"] = int(guard.get("hi_count", 0)) + 1 if hi else 0

    summary.append(
        f"AR={ar*100:.2f}%, DriftHigh={','.join([d['feature'] for d in hi]) or 'None'}"
    )
    summary.append(f"hi_count={guard['hi_count']}")

    # 1) PSI high → 임곗값 1스텝 상향
    if hi and thr_ok:
        new_thr = min(thr + 0.02, 0.85)
        if new_thr > thr:
            actions.append(
                {"type": "THRESHOLD", "value": new_thr, "reason": "psi_high"}
            )
            summary.append(f"Threshold -> {new_thr:.2f}")

    # 2) PSI high 3회 연속 → 재학습
    if hi and guard["hi_count"] >= 3 and rt_ok:
        actions.append({"type": "RETRAIN", "reason": "psi_high_persistent"})
        summary.append("Retrain triggered (PSI persistent)")
        actions.append(
            {
                "type": "SMS",
                "to": guard.get("sms_to", "+82-10-0000-0000"),
                "message": "[Fraud] PSI high persists; retrain triggered.",
            }
        )

    return {"summary": " | ".join(summary), "actions": actions}
