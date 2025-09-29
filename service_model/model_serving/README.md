## 신용카드 사기 탐지용 이상탐지 모델 파이프라인 구축

코드를 직접 실행하시면 다음과 같은 결과물을 얻게 됩니다:

1. **성능 평가 지표 출력**:

    - **Confusion Matrix**: 모델이 정상 거래와 이상 거래를 얼마나 정확하게 분류했는지 보여주는 행렬입니다.

    - **Classification Report**: precision (정밀도), recall (재현율), f1-score 등 이상 거래(Anomaly) 탐지 성능에 대한 핵심 지표들을 보여줍니다. 특히 Anomaly 클래스의 recall 값이 중요한데, 이는 실제 이상 거래 중 얼마나 많이 잡아냈는가를 의미합니다.

2. **anomaly_detection_pca.png 이미지 파일 생성**:

    - 파란 점 (정상 거래): 대부분의 정상 거래 데이터가 밀집해 있는 것을 볼 수 있습니다.

    - 노란 별 (실제 이상 거래): 파란 점들의 군집에서 벗어나 있는 경우가 많습니다.

    - 빨간 원 (모델이 탐지한 이상 거래): 모델이 이상하다고 판단한 거래들입니다. 이 빨간 원이 노란 별을 잘 감싸고 있다면, 모델이 성공적으로 이상 거래를 탐지해낸 것입니다.

- lstm, mlops 폴더: 가상환경
- 최종 코드: Isolation Forest 이상탐지 모델
- 경로: ./service_model/model_serving/server/model/anomaly_detector.py

- 모델 확인 경로:
./service_model/model_serving/server/model/result/anomaly_detection_pca.png