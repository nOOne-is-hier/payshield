import os, time, json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)

# --- 경로/결과 폴더 ---
os.makedirs("./result", exist_ok=True)

# --- 1) 데이터 로딩 (그대로) ---
train_df = pd.read_csv("./uploaded_files/train.csv")
val_df = pd.read_csv("./uploaded_files/val.csv")

features = [f"V{i}" for i in range(1, 31)]
X_train = train_df[features]
X_val = val_df[features]
y_val_true = val_df["Class"].astype(int)

print(f"[DATA] train: {X_train.shape}, val: {X_val.shape}")
print(
    f"[DATA] 실제 이상 거래 (val): {int(y_val_true.sum())} / {len(y_val_true)} "
    f"({y_val_true.mean():.4f})"
)
print(
    "[DATA] 경로] train.csv -> ./uploaded_files/train.csv, val.csv -> ./uploaded_files/val.csv"
)

# --- 2) 모델 학습 + 시간 측정 ---
contamination_rate = y_val_true.mean() if y_val_true.mean() > 0 else "auto"
model = IsolationForest(
    n_estimators=100, contamination=contamination_rate, random_state=42, n_jobs=-1
)

print("\n[TRAIN] IsolationForest 학습 시작...")
t0 = time.time()
model.fit(X_train)
fit_sec = time.time() - t0
print(
    f"[TRAIN] 완료. 소요: {fit_sec:.3f}s (n_estimators=100, contamination={contamination_rate})"
)

# --- 3) 검증 예측 + 시간 측정 ---
print("\n[PRED] 예측 시작...")
t1 = time.time()
# predict(): 1=정상, -1=이상  → y_pred: 0=정상, 1=이상으로 변환
raw_pred = model.predict(X_val)
y_pred_default = np.where(raw_pred == -1, 1, 0)
pred_sec = time.time() - t1
print(f"[PRED] 완료. 소요: {pred_sec:.3f}s")


# --- 4) 기본 성능 요약 (모델 기본 결정규칙) ---
def summarize(y_true, y_pred, tag="default"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = (tp + tn) / cm.sum()
    print(
        f"\n[METRICS::{tag}] acc={acc:.4f} | precision={prec:.4f} | recall={rec:.4f} | f1={f1:.4f}"
    )
    print("[CM]\n[[TN, FP],\n [FN, TP]]\n", cm)
    print(
        "\n[REPORT]\n",
        classification_report(
            y_true, y_pred, target_names=["정상(0)", "이상(1)"], zero_division=0
        ),
    )
    return {
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


metrics_default = summarize(y_val_true, y_pred_default, tag="default_rule")

# --- 5) 점수 기반 임계값 스윕으로 최적 임계값 제안 ---
# IsolationForest의 decision_function: 값이 클수록 정상, 작을수록 이상
# anomaly_score는 "이상일수록 큰 값"이 되도록 부호 반전하여 사용
decision = model.decision_function(X_val)  # higher=more normal
anomaly_score = -decision  # higher=more anomalous


def evaluate_at_threshold(y_true, scores, thr):
    y_hat = (scores >= thr).astype(int)  # 이상=1
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_hat, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    return prec, rec, f1, tn, fp, fn, tp


# 후보 임계값: 상위 백분위 구간을 넓게 스캔 (데이터 스케일에 무관)
percentiles = np.concatenate(
    [np.linspace(80, 99, 40, endpoint=True), np.linspace(99.1, 99.9, 17, endpoint=True)]
)
thresholds = [np.percentile(anomaly_score, p) for p in percentiles]

curve_rows = []
best_f1 = (-1, None)  # (f1, thr)
best_r90 = (-1, None)  # 최대 재현율 @ precision≥0.90
for thr in thresholds:
    prec, rec, f1, tn, fp, fn, tp = evaluate_at_threshold(
        y_val_true, anomaly_score, thr
    )
    curve_rows.append(
        {
            "threshold": thr,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
        }
    )
    if f1 > best_f1[0]:
        best_f1 = (f1, thr)
    if prec >= 0.90 and rec > best_r90[0]:
        best_r90 = (rec, thr)

curve_df = pd.DataFrame(curve_rows)
curve_df.to_csv("./result/threshold_curve.csv", index=False)

print("\n[THRESHOLD SWEEP] 완료 → ./result/threshold_curve.csv 저장")
print(f"[THRESHOLD] best F1: {best_f1[0]:.4f} @ thr={best_f1[1]:.6f}")
if best_r90[1] is not None:
    print(
        f"[THRESHOLD] max recall @ precision≥0.90: recall={best_r90[0]:.4f} @ thr={best_r90[1]:.6f}"
    )
else:
    print("[THRESHOLD] precision≥0.90을 만족하는 임계값을 찾지 못함")

# --- 6) 제안 임계값들로 재평가 및 산출물 저장 ---
suggestions = {}

# (a) best F1 임계값
thr_f1 = best_f1[1]
y_pred_f1 = (anomaly_score >= thr_f1).astype(int)
metrics_f1 = summarize(y_val_true, y_pred_f1, tag="bestF1")
suggestions["bestF1"] = {"threshold": float(thr_f1), **metrics_f1}

# (b) precision≥0.90에서 최대 재현율
if best_r90[1] is not None:
    thr_r90 = best_r90[1]
    y_pred_r90 = (anomaly_score >= thr_r90).astype(int)
    metrics_r90 = summarize(y_val_true, y_pred_r90, tag="recall_at_p90")
    suggestions["recall@p90"] = {"threshold": float(thr_r90), **metrics_r90}


# --- 7) 잘못 분류된 샘플 내보내기 (개선 분석용) ---
def dump_misclassified(tag, y_true, y_pred, scores):
    df = val_df.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["anomaly_score"] = scores
    df["miscls_type"] = np.where(
        (y_true == 1) & (y_pred == 0),
        "FN",
        np.where((y_true == 0) & (y_pred == 1), "FP", "OK"),
    )
    out = df[df["miscls_type"].isin(["FN", "FP"])].copy()
    out.sort_values(by="anomaly_score", ascending=False, inplace=True)
    out.to_csv(f"./result/misclassified_{tag}.csv", index=False)
    return len(out)


mis_default = dump_misclassified("default", y_val_true, y_pred_default, anomaly_score)
mis_f1 = dump_misclassified("bestF1", y_val_true, y_pred_f1, anomaly_score)
if "recall@p90" in suggestions:
    _ = dump_misclassified(
        "recall_at_p90",
        y_val_true,
        (anomaly_score >= suggestions["recall@p90"]["threshold"]).astype(int),
        anomaly_score,
    )

# --- 8) 메타/요약 저장 ---
summary = {
    "data": {
        "train_shape": X_train.shape,
        "val_shape": X_val.shape,
        "val_anomaly_count": int(y_val_true.sum()),
        "val_anomaly_rate": float(y_val_true.mean()),
        "paths": {
            "train": "./uploaded_files/train.csv",
            "val": "./uploaded_files/val.csv",
        },
    },
    "timing_sec": {"fit": fit_sec, "predict": pred_sec},
    "isoforest": {"n_estimators": 100, "contamination": str(contamination_rate)},
    "default_rule": metrics_default,
    "threshold_suggestions": suggestions,
    "files": {
        "threshold_curve_csv": "./result/threshold_curve.csv",
        "misclassified_default_csv": "./result/misclassified_default.csv",
        "misclassified_bestF1_csv": "./result/misclassified_bestF1.csv",
    },
}
with open("./result/metrics.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n[SAVE] 핵심 지표 → ./result/metrics.json")
print("[SAVE] 오탐/누락 목록 → ./result/misclassified_*.csv")
print("[DONE] 이미지 없이 수치/파일로 결과 정리 완료.")
