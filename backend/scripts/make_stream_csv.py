# scripts/make_stream_csv.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# --------- 설정 ---------
VAL = "uploaded_files/val.csv"
NORMAL_OUT = "data/transactions_normal.csv"
DRIFT_OUT = "data/transactions_drift.csv"
SEED = 42
rng = np.random.default_rng(SEED)

# --------- 입력 ---------
df = pd.read_csv(VAL)

# 필요한 컬럼만 보존
features = [f"V{i}" for i in range(1, 31)]
assert all(c in df.columns for c in features), "V1..V30 필요"

# --------- 공통 합성 필드(정상 스트림) ---------
base = datetime(2025, 1, 1, 12, 0, 0)
df = df.reset_index(drop=True)

# tx_id
df["tx_id"] = [f"tx_{i:07d}" for i in range(len(df))]

# 정상 금액: V1 기반 (현행 로직 유지)
amt_norm = (df["V1"].abs() * 100000).clip(1, 2_000_000).round(0)
df["amount"] = amt_norm

# 정상 merchant: 균등 분포
merchants_norm = ["A-Mart", "B-Cafe", "C-Online", "D-Store"]
df["merchant"] = rng.choice(merchants_norm, size=len(df))

# 정상 시간: 등간격 초 단위 증가
df["tx_time"] = [(base + timedelta(seconds=i)).isoformat() for i in range(len(df))]

# 정상 스트림 저장
cols = ["tx_id", "tx_time", "merchant", "amount"] + features
df[cols].to_csv(NORMAL_OUT, index=False)

# --------- 드리프트 스트림(강하게 흔듦) ---------
df_d = df.copy()

# 1) 범주 드리프트: 신규 카테고리 + 강한 쏠림
merchants_drift = ["A-Mart", "B-Cafe", "C-Online", "D-Store", "E-Market"]  # 신규 포함
probs_drift = np.array([0.05, 0.10, 0.75, 0.08, 0.02])  # 온라인 쏠림
df_d["merchant"] = rng.choice(merchants_drift, p=probs_drift, size=len(df_d))

# 2) 금액 분포 드리프트: 로그정규 + 상향 시프트 + 헤비테일
median_amt = float(np.median(amt_norm))
log_mean = np.log(max(median_amt, 10)) + 0.3  # 상향 시프트
log_sigma = 0.9
amt_drift = rng.lognormal(mean=log_mean, sigma=log_sigma, size=len(df_d))
df_d["amount"] = np.clip(amt_drift, 1, 5_000_000).round(0)

# 3) 시간 패턴 드리프트: 심야 집중 + 버스트(비등간 간격)
night_base = base.replace(hour=1, minute=0, second=0)
gaps = rng.exponential(scale=0.6, size=len(df_d))  # 초 단위 간격(버스트/스파이크)
cum = np.cumsum(gaps)
# numpy datetime64로 변환 후 ISO 스트링화
t0 = np.datetime64(night_base).astype("datetime64[s]")
times = (t0 + cum.astype("timedelta64[s]")).astype("datetime64[s]").astype(str)
df_d["tx_time"] = times

# 4) 피처 전역 변형: 스케일업 + 평균 시프트 + 상관 붕괴(셔플)
scale_up = ["V2", "V3", "V7", "V10", "V13", "V18"]
for k in scale_up:
    df_d[k] = df_d[k] * 1.6

shift_up = ["V4", "V12", "V19"]
for k in shift_up:
    df_d[k] = df_d[k] + 0.8

# 상관 붕괴: 한 피처를 무작위 셔플
df_d["V14"] = rng.permutation(df_d["V14"].values)

# 5) 결측 증가(센서/로그 누락 가정)
for k in ["V5", "V16", "V21"]:
    mask = rng.random(len(df_d)) < 0.12  # 12% 결측
    df_d.loc[mask, k] = np.nan

# 6) 분절 드리프트(리짐 체인지): 중간 이후 급변
cut = len(df_d) // 2
for k in ["V1", "V8", "V20", "V25"]:
    df_d.loc[cut:, k] = df_d.loc[cut:, k] * 1.9 + 0.5

# (선택) 약간의 이상치 주입: 꼬리 강화
# 상위 1%를 강한 스파이크로
n_out = max(1, len(df_d) // 100)
idx_out = rng.choice(len(df_d), size=n_out, replace=False)
df_d.loc[idx_out, "amount"] = np.clip(df_d.loc[idx_out, "amount"] * 8, 1, 10_000_000)

# 드리프트 스트림 저장
df_d[cols].to_csv(DRIFT_OUT, index=False)

print("Wrote:", NORMAL_OUT, DRIFT_OUT)
print(
    f"[SUMMARY] normal: n={len(df)}  | merchants={merchants_norm} "
    f"\n          drift : n={len(df_d)} | merchants={merchants_drift} (skewed)"
)
