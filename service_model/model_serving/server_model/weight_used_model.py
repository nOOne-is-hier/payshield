#### 다음 실습 코드는 학습 목적으로만 사용 바랍니다. 문의 : audit@korea.ac.kr 임성열 Ph.D.
#### 제공되는 실습 코드는 완성된 버전이 아니며, 일부 이스터 에그 (개선이 필요한 발견 사항)을 포함하고 있습니다.

# pip install pandas numpy matplotlib scikit-learn keras tensorflow pydot graphviz

'''설치 패키지 설명 :
# 데이터 처리 : pandas, numpy
# 시각화 : matplotlib
# 머신러닝/딥러닝 : scikit-learn (MinMaxScaler, mean_squared_error), keras, tensorflow (모델 불러오기, 학습, 레이어 구성)
# 모델 시각화 : pydot, graphviz (keras.utils.plot_model)
# graphviz는 OS 패키지 설치 필요 (apt-get install graphviz 또는 brew install graphviz)
# 표준 라이브러리 : math, os → 파이썬 기본 내장 모듈, 설치 불필요'''

import pandas as pd
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
from keras.utils import plot_model
import os
from config import MODEL_DIR, IMAGE_DIR, MODEL_SAVE_PATH, DATA_PATH, MODEL_SHAPES_PLOT_PATH, PREDICTION_PLOT_PATH

# 모델 로딩
model = load_model(MODEL_SAVE_PATH)

# 데이터 로딩
dataset = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=['Date'], encoding='utf-8')

# 모델 아키텍처 이미지 생성
plot_model(model, to_file=os.path.join(IMAGE_DIR, "model.png"))
plot_model(model, to_file=MODEL_SHAPES_PLOT_PATH, show_shapes=True)

# RMSE 계산 함수
def return_rmse(test, predicted):
    rmse = math.sqrt(mean_squared_error(test, predicted))
    result_msg = f"The root mean squared error is {rmse}."
    print(result_msg)
    return result_msg

# 예측 결과 그래프 저장 함수
def plot_predictions(test, predicted):
    plt.clf()  # 이전 그래프 초기화
    plt.plot(test, color='red', label='Real IBM Stock Price')
    plt.plot(predicted, color='blue', label='Predicted IBM Stock Price')
    plt.title('IBM Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('IBM Stock Price')
    plt.legend()
    plt.savefig(PREDICTION_PLOT_PATH)
    return PREDICTION_PLOT_PATH

# 데이터 전처리 및 모델 예측 실행 함수
def process(dataset):
    model = load_model(MODEL_SAVE_PATH)

    # 'High' 열 선택
    training_set = dataset.loc[:'2016', ["High"]].values
    test_set = dataset.loc['2017':, ["High"]].values

    # 데이터 스케일링
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    # 테스트 데이터 준비
    dataset_total = pd.concat([dataset.loc[:'2016', "High"], dataset.loc['2017':, "High"]], axis=0)
    inputs = dataset_total[len(dataset_total) - len(test_set) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)

    X_test = []
    for i in range(60, len(inputs)):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # 모델 예측
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    # 결과 시각화 및 평가
    result_visualizing = plot_predictions(test_set, predicted_stock_price)
    result_evaluating = return_rmse(test_set, predicted_stock_price)

    return result_visualizing, result_evaluating

# 추가된 함수: 모델 아키텍처 이미지 경로 반환
def get_model_shapes_png():
    """모델 구조 이미지의 경로 반환"""
    return MODEL_SHAPES_PLOT_PATH

# 추가된 함수: 예측 이미지 경로 반환
def get_stock_png():
    """주식 예측 결과 이미지의 경로 반환"""
    return PREDICTION_PLOT_PATH
