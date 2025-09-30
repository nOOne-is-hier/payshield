import os, time, json
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)

import torch
import torch.nn as nn

# ------------------------
# 0) 결과 폴더
# ------------------------
os.makedirs("./result", exist_ok=True)

# ------------------------
# 1) 데이터 로딩
# ------------------------
train_df = pd.read_csv("./uploaded_files/train.csv")
val_df = pd.read_csv("./uploaded_files/val.csv")

features = [f"V{i}" for i in range(1, 31)]
X_train = train_df[features].astype(np.float32)
X_val = val_df[features].astype(np.float32)
y_val_true = val_df["Class"].astype(int).values

print(f"[DATA] train: {X_train.shape}, val: {X_val.shape}")
print(
    f"[DATA] 실제 이상 거래 (val): {int(y_val_true.sum())} / {len(y_val_true)} ({y_val_true.mean():.4f})"
)
print(
    "[DATA] 경로] train.csv -> ./uploaded_files/train.csv, val.csv -> ./uploaded_files/val.csv"
)

# 표준화(권장): BN이 있어도 입력 스케일 안정화에 도움
scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train.values)
X_val_np = scaler.transform(X_val.values)


# ------------------------
# 2) AE (BN + LeakyReLU, L1Loss)
# ------------------------
class CosAE(nn.Module):
    def __init__(self, d=30):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, d),  # 출력층에는 활성/BN 없음
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = int(os.getenv("AE_EPOCHS", 40))  # 기본 40 (빠르게는 20)
batch_size = int(os.getenv("AE_BS", 16384))  # 글의 설정 따라 큼직하게
lr = float(os.getenv("AE_LR", 1e-2))

model = CosAE(d=30).to(device)
opt = torch.optim.Adam(model.parameters(), lr=lr)
crit = nn.L1Loss()  # MAE

Xtr = torch.tensor(X_train_np, dtype=torch.float32).to(device)
Xva = torch.tensor(X_val_np, dtype=torch.float32).to(device)

# ------------------------
# 3) 학습
# ------------------------
print("\n[TRAIN] Cosine-AE 학습 시작...")
t0 = time.time()
model.train()
N = len(Xtr)
for ep in range(epochs):
    ep_losses = []
    for i in range(0, N, batch_size):
        xb = Xtr[i : i + batch_size]
        opt.zero_grad()
        recon = model(xb)
        loss = crit(recon, xb)
        loss.backward()
        opt.step()
        ep_losses.append(loss.item())
    if (ep + 1) % 5 == 0:
        print(f"[EPOCH {ep+1:>3}] loss={np.mean(ep_losses):.6f}")
fit_sec = time.time() - t0
print(f"[TRAIN] 완료. 소요: {fit_sec:.3f}s (epochs={epochs}, bs={batch_size}, lr={lr})")

# ------------------------
# 4) 추론 & 이상점수(1 - cosine)
# ------------------------
print("\n[PRED] 예측 시작...")
t1 = time.time()
model.eval()
with torch.no_grad():
    recon = model(Xva)
    # cosine_sim in [-1,1], 정상은 1에 가까움 → anomaly_score = 1 - cos
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    cosine_sim = cos(Xva, recon).detach().cpu().numpy()
    anomaly_score = 1.0 - cosine_sim  # ↑일수록 이상
pred_sec = time.time() - t1
print(f"[PRED] 완료. 소요: {pred_sec:.3f}s")


# ------------------------
# 5) 요약/리포팅(기존 형식 유지)
# ------------------------
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
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# 기본 임계값: 검증셋 이상비율 기반(상위 p%)
contamination_rate = y_val_true.mean() if y_val_true.mean() > 0 else 0.01
pct = 100.0 * (1.0 - contamination_rate)
thr_default = float(np.percentile(anomaly_score, pct))
y_pred_default = (anomaly_score >= thr_default).astype(int)
metrics_default = summarize(y_val_true, y_pred_default, tag="default_rule")
print(
    f"[DEFAULT THR] contam={contamination_rate:.6f} → pct={pct:.3f} → thr={thr_default:.6f}"
)

# 임계값 스윕 (코사인 기반 점수라 상위 백분위 스캔이 합리적)
percentiles = np.r_[np.linspace(80, 99, 40), np.linspace(99.1, 99.9, 17)]
thresholds = [np.percentile(anomaly_score, p) for p in percentiles]


def evaluate_at_threshold(y_true, scores, thr):
    y_hat = (scores >= thr).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_hat, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    return prec, rec, f1, tn, fp, fn, tp


curve_rows, best_f1 = [], (-1, None)
best_r90 = (-1, None)  # max recall @ precision≥0.90
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

pd.DataFrame(curve_rows).to_csv("./result/threshold_curve.csv", index=False)
print("\n[THRESHOLD SWEEP] 완료 → ./result/threshold_curve.csv")
print(f"[THRESHOLD] best F1: {best_f1[0]:.4f} @ thr={best_f1[1]:.6f}")
if best_r90[1] is not None:
    print(
        f"[THRESHOLD] max recall @ p≥0.90: recall={best_r90[0]:.4f} @ thr={best_r90[1]:.6f}"
    )
else:
    print("[THRESHOLD] precision≥0.90 만족 지점 없음")


# 잘못 분류 덤프
def dump_misclassified(tag, y_true, y_pred, scores):
    df = val_df.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    df["anomaly_score"] = scores
    df["miscls_type"] = np.where(
        (df["y_true"] == 1) & (df["y_pred"] == 0),
        "FN",
        np.where((df["y_true"] == 0) & (df["y_pred"] == 1), "FP", "OK"),
    )
    out = df[df["miscls_type"].isin(["FN", "FP"])].copy()
    out.sort_values(by="anomaly_score", ascending=False, inplace=True)
    out.to_csv(f"./result/misclassified_{tag}.csv", index=False)
    return len(out)


suggestions = {}
thr_f1 = float(best_f1[1])
y_pred_f1 = (anomaly_score >= thr_f1).astype(int)
metrics_f1 = summarize(y_val_true, y_pred_f1, tag="bestF1")
suggestions["bestF1"] = {"threshold": thr_f1, **metrics_f1}
_ = dump_misclassified("bestF1", y_val_true, y_pred_f1, anomaly_score)

if best_r90[1] is not None:
    thr_r90 = float(best_r90[1])
    y_pred_r90 = (anomaly_score >= thr_r90).astype(int)
    metrics_r90 = summarize(y_val_true, y_pred_r90, tag="recall_at_p90")
    suggestions["recall@p90"] = {"threshold": thr_r90, **metrics_r90}
    _ = dump_misclassified("recall_at_p90", y_val_true, y_pred_r90, anomaly_score)

_ = dump_misclassified("default", y_val_true, y_pred_default, anomaly_score)

# ------------------------
# 6) metrics.json 저장 (형식 호환)
# ------------------------
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
    "timing_sec": {"fit": float(fit_sec), "predict": float(pred_sec)},
    "model": {
        "type": "autoencoder_cosine",
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "arch": "30-64-128-64-30 (BN+LeakyReLU), L1Loss",
        "device": str(device),
        "scaler": "StandardScaler(train-fit)",
    },
    "isoforest": None,
    "default_rule": metrics_default,
    "threshold_suggestions": suggestions,
    "files": {
        "threshold_curve_csv": "./result/threshold_curve.csv",
        "misclassified_default_csv": "./result/misclassified_default.csv",
        "misclassified_bestF1_csv": "./result/misclassified_bestF1.csv",
        "misclassified_recall_at_p90_csv": (
            "./result/misclassified_recall_at_p90.csv"
            if "recall@p90" in suggestions
            else None
        ),
    },
}
with open("./result/metrics.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n[SAVE] 핵심 지표 → ./result/metrics.json")
print("[SAVE] 오탐/누락 목록 → ./result/misclassified_*.csv")
print("[DONE] Cosine-AE 파이프라인 완료.")

# ========================
# 7) 모델 아티팩트 내보내기
# ========================
import joblib, datetime, os, json, torch

export_dir = os.getenv("EXPORT_DIR", "models/v1.0")
os.makedirs(export_dir, exist_ok=True)

# 7-1) StandardScaler 저장
joblib.dump(scaler, os.path.join(export_dir, "scaler.pkl"))

# 7-2) AE 가중치 저장 (state_dict)
torch.save(model.state_dict(), os.path.join(export_dir, "ae_state.pt"))

# 7-3) model_card.json 저장 (버전/권고 threshold/지표 요약)
trained_at = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

# 운영 임계값 권고 선택 로직
thr_default_suggest = float(thr_default)
thr_bestF1_suggest = float(
    suggestions.get("bestF1", {}).get("threshold", thr_default_suggest)
)
thr_recall_p90_suggest = suggestions.get("recall@p90", {}).get("threshold")
if thr_recall_p90_suggest is not None:
    thr_recall_p90_suggest = float(thr_recall_p90_suggest)

model_card = {
    "version": os.getenv("MODEL_VERSION", "v1.0.0"),
    "trained_at": trained_at,
    "data": {
        "train_rows": int(X_train.shape[0]),
        "val_rows": int(X_val.shape[0]),
        "val_anomaly_rate": float(y_val_true.mean()),
    },
    "model": summary["model"],  # 네가 위에서 만든 summary의 model 블록 그대로 활용
    "metrics_offline": {
        "default_rule": summary["default_rule"],
        "bestF1": suggestions.get("bestF1"),
        "recall@p90": suggestions.get("recall@p90"),
    },
    "threshold_suggestions": {
        "default_rule": thr_default_suggest,
        "bestF1": thr_bestF1_suggest,
        "recall@p90": thr_recall_p90_suggest,
    },
    "files": {"threshold_curve_csv": summary["files"]["threshold_curve_csv"]},
}
with open(os.path.join(export_dir, "model_card.json"), "w", encoding="utf-8") as f:
    json.dump(model_card, f, ensure_ascii=False, indent=2)

# 7-4) signature.json (서빙 I/O 스키마)
signature = {
    "inputs": {
        "dtype": "float32",
        "shape": ["N", 30],
        "columns": [f"V{i}" for i in range(1, 31)],
    },
    "outputs": {"anomaly_score": "float32", "cosine_sim": "float32"},
    "score_logic": "anomaly_score = 1 - cosine(X, AE(X))",
}
with open(os.path.join(export_dir, "signature.json"), "w", encoding="utf-8") as f:
    json.dump(signature, f, ensure_ascii=False, indent=2)

print(f"\n[EXPORT] Artifacts saved to: {export_dir}")
print("[EXPORT] scaler.pkl, ae_state.pt, model_card.json, signature.json")
