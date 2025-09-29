#### 다음 실습 코드는 학습 목적으로만 사용 바랍니다. 문의 : audit@korea.ac.kr 임성열 Ph.D.
#### 제공되는 실습 코드는 완성된 버전이 아니며, 일부 이스터 에그 (개선이 필요한 발견 사항)을 포함하고 있습니다.

# pip install fastapi uvicorn[standard] pandas pytz numpy matplotlib scikit-learn keras tensorflow pydot graphviz

'''설치 패키지 설명 : 
# scikit-learn → sklearn.preprocessing.MinMaxScaler, sklearn.metrics.mean_squared_error
# keras, tensorflow → Sequential, LSTM, Dropout, Dense, load_model, plot_model
# pydot, graphviz → keras.utils.plot_model 사용 시 필요
# 참고: 이미지 내보내기 실패 시 OS에 Graphviz 시스템 패키지(예: brew install graphviz, apt-get install graphviz)도 설치해야 합니다.'''

# model.py
import os
import math
from pathlib import Path

import numpy as np
import pandas as pd

# 화면 없이 저장만 하도록 (서버/윈도우에서 권장)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras import Input
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from keras.utils import plot_model

from config import (
    DATA_PATH,
    MODEL_SAVE_PATH,
    MODEL_PLOT_PATH,
    MODEL_SHAPES_PLOT_PATH,
    PREDICTION_PLOT_PATH,
)


# 유틸
def _ensure_paths():
    for p in [MODEL_SAVE_PATH, MODEL_PLOT_PATH, MODEL_SHAPES_PLOT_PATH, PREDICTION_PLOT_PATH]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    """CSV 로딩 (Date 인덱스)"""
    return pd.read_csv(
        DATA_PATH, index_col="Date", parse_dates=["Date"], encoding="utf-8"
    )


def prepare_training_data(dataset: pd.DataFrame):
    """학습 데이터(X_train, y_train)와 스케일러 반환"""
    training_set = dataset.loc[: "2016", ["High"]].values  # (N, 1)

    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    X_train, y_train = [], []
    for i in range(60, len(training_set_scaled)):
        X_train.append(training_set_scaled[i - 60 : i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)  # (N-60, 60), (N-60,)

    # LSTM 입력 형태: (samples, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    return X_train, y_train, sc


def build_lstm_model(window: int, n_features: int = 1) -> Sequential:
    """
    Keras 3 권고에 따라 input_shape를 레이어에 직접 전달하지 않고
    첫 레이어로 Input(...)을 사용.
    """
    model = Sequential(
        [
            Input(shape=(window, n_features)),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1),
        ]
    )
    model.compile(optimizer="rmsprop", loss="mean_squared_error")
    return model


def train_and_save(dataset: pd.DataFrame, epochs: int = 2, batch_size: int = 32):
    """모델 학습, 저장 및 구조 이미지 내보내기"""
    _ensure_paths()

    X_train, y_train, _ = prepare_training_data(dataset)
    window = X_train.shape[1]

    model = build_lstm_model(window=window, n_features=1)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    # 저장
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to '{MODEL_SAVE_PATH}'")

    # 모델 구조 이미지 (Graphviz 미설치 시 예외 무시)
    try:
        plot_model(model, to_file=str(MODEL_PLOT_PATH))
        plot_model(model, to_file=str(MODEL_SHAPES_PLOT_PATH), show_shapes=True)
        print(f"Model structure saved to '{MODEL_PLOT_PATH}' and '{MODEL_SHAPES_PLOT_PATH}'")
    except Exception as e:
        print(f"[warn] plot_model failed: {e}")

    return model


# 예측/평가
def process(dataset: pd.DataFrame):
    """
    저장된 모델을 로드하여 예측 및 평가 수행.
    반환: (예측 이미지 경로, RMSE 메시지)
    """
    _ensure_paths()
    model = load_model(MODEL_SAVE_PATH)

    # 'High' 열 선택
    training_set = dataset.loc[: "2016", ["High"]].values
    test_set = dataset.loc["2017":, ["High"]].values  # (M, 1)

    # 스케일러는 학습 구간 기준으로 적합
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    # 테스트 입력 구간 구성
    dataset_total = pd.concat(
        [dataset.loc[: "2016", "High"], dataset.loc["2017":, "High"]], axis=0
    )
    inputs = dataset_total[len(dataset_total) - len(test_set) - 60 :].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)

    X_test = []
    for i in range(60, len(inputs)):
        X_test.append(inputs[i - 60 : i, 0])
    X_test = np.array(X_test).reshape((-1, 60, 1))

    # 예측
    predicted = model.predict(X_test)
    predicted = sc.inverse_transform(predicted)  # (M, 1)

    # 시각화 및 평가
    img_path = plot_predictions(test_set, predicted)
    msg = return_rmse(test_set, predicted)
    return img_path, msg


def plot_predictions(test, predicted):
    plt.clf()
    plt.figure(dpi=120)
    plt.plot(test, label="Real IBM Stock Price")
    plt.plot(predicted, label="Predicted IBM Stock Price")
    plt.title("IBM Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("IBM Stock Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PREDICTION_PLOT_PATH)
    return str(PREDICTION_PLOT_PATH)


def return_rmse(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    result_msg = f"The root mean squared error is {rmse}."
    print(result_msg)
    return result_msg


# 스크립트 실행 시에만 학습(서버 import 시 학습 방지)
if __name__ == "__main__":
    df = load_dataset()
    train_and_save(df, epochs=2, batch_size=32)
