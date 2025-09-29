# config.py

import os
from pathlib import Path

# 기본 경로 (끝에 '\' 금지). 환경변수 BASE_DIR이 있으면 우선 사용.
BASE_DIR = Path(os.getenv("BASE_DIR", r"C:\Users\SKAX\Desktop\Creditcard_Project\creditcard\service_model\model_serving\server")).resolve()

# 디렉터리
UPLOAD_DIR    = BASE_DIR / "uploaded_files"
MODEL_DIR     = BASE_DIR / "model"
IMAGE_DIR     = BASE_DIR / "view-model-architecture"
MODEL_IMG_DIR = BASE_DIR / "model-images"

# 파일 경로
DATA_PATH               = UPLOAD_DIR / "IBM_2006-01-01_to_2018-01-01.csv"
MODEL_SAVE_PATH         = MODEL_DIR / "result" / "stock_lstm_model.keras"
MODEL_PLOT_PATH         = IMAGE_DIR / "model.png"
MODEL_SHAPES_PLOT_PATH  = IMAGE_DIR / "shapes" / "model_shapes.png"
PREDICTION_PLOT_PATH    = IMAGE_DIR / "stock.png"
