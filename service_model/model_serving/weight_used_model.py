# weight_used_model.py

# -*- coding: utf-8 -*-
import pandas as pd
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
from keras.utils import plot_model
from config import MODEL_DIR, IMAGE_DIR, MODEL_SAVE_PATH, DATA_PATH

# 모델 로딩
model = load_model(MODEL_SAVE_PATH)

# 데이터 로딩
dataset = pd.read_csv(DATA_PATH, index_col='Date', parse_dates=['Date'], encoding='utf-8')

# 모델 아키텍처 이미지 생성
plot_model(model, to_file=f'{IMAGE_DIR}/model.png')
plot_model(model, to_file=f'{IMAGE_DIR}/model_shapes.png', show_shapes=True)

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
    plt.savefig(f'{IMAGE_DIR}/stock.png')
    return f'{IMAGE_DIR}/stock.png'

# 데이터 전처리 및 모델 예측 실행 함수
def process(dataset):
    model = load_model(MODEL_SAVE_PATH)

    # 'High' 열 선택
    training_set = dataset[:'2016'].iloc[:, 1:2].values
    test_set = dataset['2017':].iloc[:, 1:2].values

    # 데이터 스케일링
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    # 테스트 데이터 준비
    dataset_total = pd.concat((dataset["High"][:'2016'], dataset["High"]['2017':]), axis=0)
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
