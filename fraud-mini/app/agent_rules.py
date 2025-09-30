import time


def agent_decide(metrics, recent_alerts, guard):
    actions, summary_lines = [], []
    ar = metrics.get("anomaly_rate", 0.0)
    drift = metrics.get("drift", [])
    hi = [d for d in drift if d["level"] == "high"]
    summary_lines.append(
        f"AR={ar*100:.2f}%, DriftHigh={','.join([d['feature'] for d in hi]) or 'None'}"
    )

    now = time.time()
    can_change_thr = (now - guard.get("thr_changed_at", 0)) >= 300
    can_retrain = (now - guard.get("retrain_at", 0)) >= 600

    # (추가) PSI high가 점수/시간/금액 중 하나라도 있으면 카운팅
    if hi:
        guard["hi_count"] = guard.get("hi_count", 0) + 1
    else:
        guard["hi_count"] = 0

    # 기존 조건: hi & AR 상승 → 우선 threshold 조정
    if hi and ar >= guard.get("base_ar", 0) + 0.0015:
        if can_change_thr:
            new_thr = min(guard.get("threshold", 0.7) + 0.02, 0.85)
            actions.append(
                {"type": "THRESHOLD", "value": new_thr, "reason": "drift_high"}
            )
            summary_lines.append(f"Threshold -> {new_thr:.2f}")
        if can_retrain and not can_change_thr:
            actions.append({"type": "RETRAIN", "reason": "drift_persistent"})
            summary_lines.append("Retrain triggered")
            actions.append(
                {
                    "type": "SMS",
                    "to": guard.get("sms_to", "+82-10-0000-0000"),
                    "message": "[Fraud] Drift high; retrain triggered.",
                }
            )

    # (신규 분기) AR이 낮아도 PSI high가 2~3회 연속이면 재학습
    if can_retrain and guard.get("hi_count", 0) >= 3 and not can_change_thr:
        actions.append({"type": "RETRAIN", "reason": "psi_high_persistent"})
        summary_lines.append("Retrain triggered (PSI persistent)")
        actions.append(
            {
                "type": "SMS",
                "to": guard.get("sms_to", "+82-10-0000-0000"),
                "message": "[Fraud] PSI high persists; retrain triggered.",
            }
        )

    summary = " | ".join(summary_lines)
    return {"summary": summary, "actions": actions}
