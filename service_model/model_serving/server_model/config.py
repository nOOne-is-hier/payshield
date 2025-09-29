# # config.py
# import os

# # 기본 경로 설정
# BASE_DIR = os.getenv("BASE_DIR", "/Users/phoenix/Eagle/2025_FastAPI/service_model/model_serving")
# UPLOAD_DIR = os.path.join(BASE_DIR, "/server/uploaded_files")
# MODEL_DIR = os.path.join(BASE_DIR, "/server/model")
# IMAGE_DIR = os.path.join(BASE_DIR, "/server/model-images")
# MODEL_IMG_DIR = os.path.join(BASE_DIR, "/server/model-images")

# # 파일 경로
# DATA_PATH = os.path.join(UPLOAD_DIR, "IBM_2006-01-01_to_2018-01-01.csv")
# MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "stock_lstm_model_nogpu.keras")
# MODEL_PLOT_PATH = os.path.join(IMAGE_DIR, "model_service.png")
# MODEL_SHAPES_PLOT_PATH = os.path.join(IMAGE_DIR, "model_shapes.png")
# PREDICTION_PLOT_PATH = os.path.join(IMAGE_DIR, "stock.png")

import os

# 기본 경로 설정 (환경 변수에서 가져오거나 기본값 사용)
BASE_DIR = os.getenv("BASE_DIR", "/Users/phoenix/Eagle/2025_MLOps/service_model/model_serving/server")

# 상대 경로를 연결할 때 슬래시(/) 제거
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_files")
MODEL_DIR = os.path.join(BASE_DIR, "model")
IMAGE_DIR = os.path.join(BASE_DIR, "view-model-architecture")
MODEL_IMG_DIR = os.path.join(BASE_DIR, "model-images")

# 파일 경로 설정
DATA_PATH = os.path.join(UPLOAD_DIR, "IBM_2006-01-01_to_2018-01-01.csv")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "stock_lstm_model_nogpu.keras")
MODEL_PLOT_PATH = os.path.join(IMAGE_DIR, "model.png")
MODEL_SHAPES_PLOT_PATH = os.path.join(IMAGE_DIR, "shapes/model_shapes.png")
PREDICTION_PLOT_PATH = os.path.join(IMAGE_DIR, "stock.png")
