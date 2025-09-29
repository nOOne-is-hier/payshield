import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
import platform

# Matplotlib 스타일 설정
plt.style.use("fivethirtyeight")
# 운영체제에 맞는 한글 폰트 설정
if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
elif platform.system() == "Darwin":  # Mac OS
    plt.rc("font", family="AppleGothic")
else:  # Linux
    # 코랩과 같은 리눅스 환경에서는 아래 코드를 실행하여 나눔 폰트를 설치하고,
    # 런타임 다시 시작을 해야 한글 폰트가 적용됩니다.
    # !sudo apt-get install -y fonts-nanum
    # !sudo fc-cache -fv
    # !rm ~/.cache/matplotlib -rf
    plt.rc("font", family="NanumGothic")
# 마이너스 기호 깨짐 방지
plt.rc("axes", unicode_minus=False)

# --- 1. 데이터 로딩 및 준비 ---

try:
    # 훈련 데이터는 정상 거래만 있다고 가정
    train_df = pd.read_csv("./uploaded_files/train.csv")
    # 검증 데이터에는 정상(0)과 이상(1) 거래가 섞여 있음
    val_df = pd.read_csv("./uploaded_files/val.csv")
    print("파일을 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print("오류: train.csv 또는 val.csv 파일을 찾을 수 없습니다.")
    print("스크립트 파일과 CSV 파일들이 같은 폴더에 있는지 확인하세요.")
    exit()

# 특성(feature) 정의 (ID 제외)
features = [f"V{i}" for i in range(1, 31)]

# 훈련 데이터 (X_train)와 검증 데이터 (X_val) 준비
X_train = train_df[features]
X_val = val_df[features]
# 실제 이상 거래 레이블
y_val_true = val_df["Class"]

print(f"훈련 데이터 형태: {X_train.shape}")
print(f"검증 데이터 형태: {X_val.shape}")
print(f"검증 데이터 내 실제 이상 거래 건수: {y_val_true.sum()}")

# --- 2. Isolation Forest 모델 학습 ---

# Isolation Forest 모델 생성 및 학습
# contamination: 데이터셋에서 예상되는 이상치 비율. val_df의 비율을 참고하여 설정
contamination_rate = y_val_true.mean() if y_val_true.mean() > 0 else "auto"
model = IsolationForest(
    n_estimators=100, contamination=contamination_rate, random_state=42, n_jobs=-1
)

print("\nIsolation Forest 모델 학습 중...")
model.fit(X_train)
print("모델 학습 완료.")


# --- 3. 검증 데이터에 대한 이상탐지 예측 ---

print("\n검증 데이터로 이상 거래 예측 중...")
# 예측 결과: 정상(1), 이상(-1)
predictions = model.predict(X_val)

# Scikit-learn의 평가 함수(y_true)와 형식을 맞추기 위해 변환: 정상(0), 이상(1)
y_val_pred = [1 if p == -1 else 0 for p in predictions]
print("예측 완료.")


# --- 4. 성능 평가 ---

print("\n--- 이상탐지 모델 성능 ---")
print(f"모델이 탐지한 이상 거래 건수: {sum(y_val_pred)}")
print(f"실제 이상 거래 건수: {y_val_true.sum()}")

print("\nConfusion Matrix (혼동 행렬):")
# TN, FP
# FN, TP
# [[실제 Normal-예측 Normal, 실제 Normal-예측 Anomaly],
#  [실제 Anomaly-예측 Normal, 실제 Anomaly-예측 Anomaly]]
print(confusion_matrix(y_val_true, y_val_pred))

print("\nClassification Report (분류 리포트):")
print(
    classification_report(
        y_val_true, y_val_pred, target_names=["정상 (Class 0)", "이상 (Class 1)"]
    )
)


# --- 5. 결과 시각화 (PCA 사용) ---

print("\nPCA를 이용한 결과 시각화 중...")
# PCA를 사용하여 30차원 데이터를 2차원으로 축소
pca = PCA(n_components=2)
X_val_pca = pca.fit_transform(X_val)

results_df = pd.DataFrame(
    {
        "pca1": X_val_pca[:, 0],
        "pca2": X_val_pca[:, 1],
        "is_anomaly_pred": y_val_pred,
        "is_anomaly_true": y_val_true,
    }
)

plt.figure(figsize=(12, 8))

# 정상 거래 시각화
normal_df = results_df[results_df["is_anomaly_true"] == 0]
plt.scatter(
    normal_df["pca1"], normal_df["pca2"], c="blue", alpha=0.5, label="정상 거래"
)

# 실제 이상 거래 시각화
anomaly_df = results_df[results_df["is_anomaly_true"] == 1]
plt.scatter(
    anomaly_df["pca1"],
    anomaly_df["pca2"],
    c="gold",
    marker="*",
    s=200,
    edgecolor="black",
    label="실제 이상 거래",
)

# 모델이 탐지한 이상 거래에 테두리 표시
detected_anomalies_df = results_df[results_df["is_anomaly_pred"] == 1]
plt.scatter(
    detected_anomalies_df["pca1"],
    detected_anomalies_df["pca2"],
    facecolors="none",
    edgecolors="red",
    s=150,
    linewidth=2,
    label="모델이 탐지한 이상 거래",
)

plt.title("이상탐지 결과 시각화 (PCA 차원 축소)")
plt.xlabel("주성분 1 (Principal Component 1)")
plt.ylabel("주성분 2 (Principal Component 2)")
plt.legend()
plt.tight_layout()
plt.savefig("anomaly_detection_pca.png")
plt.show()
print("\n결과 그래프가 anomaly_detection_pca.png 파일로 저장되었습니다.")
