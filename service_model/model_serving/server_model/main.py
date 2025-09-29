#### 다음 실습 코드는 학습 목적으로만 사용 바랍니다. 문의 : audit@korea.ac.kr 임성열 Ph.D.
#### 제공되는 실습 코드는 완성된 버전이 아니며, 일부 이스터 에그 (개선이 필요한 발견 사항)을 포함하고 있습니다.

# pip install fastapi "uvicorn[standard]" pandas pytz python-multipart
# pip install -U pip wheel
# pip install matplotlib

'''설치 패키지 설명 :
# fastapi, uvicorn[standard] : FastAPI를 통한 모델 서빙에 필요, uvicorn[standard]는 의존성 패키지까지 추가 설치
# pandas: pd (데이터프레임 처리)
# pytz: 시간대(timezone) 처리
# python-multipart: 파일 업로드 처리'''

# main.py
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from contextlib import asynccontextmanager
import asyncio
import base64
from datetime import datetime
import importlib
import os
from pathlib import Path

import pandas as pd
import pytz
from fastapi import FastAPI, APIRouter, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from . import model
from . import weight_used_model
# 상대 경로 사용, 현재 폴더인 server_model 상위 폴더에서 현 위치 인식
from fastapi.staticfiles import StaticFiles
from fastapi import Response

# 정적 마운트 아래에 추가
from config import UPLOAD_DIR, IMAGE_DIR, MODEL_IMG_DIR

# -------------------------------------------------
# 경로/디렉터리 및 프리픽스(root_path)
# -------------------------------------------------
STD_DIR = Path(__file__).resolve().parent.parent  # .../model_serving
PUBLIC_DIR = STD_DIR / "public"

# 프록시 하위 경로에서 서비스할 경우 설정 (예: /api/v2)
APP_ROOT_PATH = os.getenv("APP_ROOT_PATH", "").rstrip("/")  # 빈 문자열 또는 "/api/v2"

# 타임존
timezone = pytz.timezone("Asia/Seoul")

router = APIRouter()

# -------------------------------------------------
# Lifespan: 스타트업을 가볍게 (블로킹 작업 금지)
# -------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 정적/결과 디렉터리 보장
    for d in (PUBLIC_DIR, UPLOAD_DIR, IMAGE_DIR, MODEL_IMG_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)
    yield
    # 종료 시 별도 정리 없음

app = FastAPI(
    lifespan=lifespan,
    root_path=APP_ROOT_PATH,           # ✅ 프리픽스 반영
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 필요 시 도메인 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /static 경로에 정적 리소스 제공
app.mount("/static", StaticFiles(directory=str(PUBLIC_DIR)), name="static")

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    ico = PUBLIC_DIR / "favicon.ico"
    if ico.exists():
        return FileResponse(str(ico), media_type="image/x-icon")
    png = PUBLIC_DIR / "favicon.png"
    if png.exists():
        return FileResponse(str(png), media_type="image/png")
    return Response(status_code=204)  # 404 대신 조용히 처리

# 간단한 요청 로그 (디버그용)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    resp: Response
    try:
        resp = await call_next(request)
    finally:
        # 필요한 경우 상세 로깅 추가
        pass
    return resp

# -------------------------------------------------
# 유틸
# -------------------------------------------------
def _b64_png(path: Path) -> str:
    """PNG 파일을 data URI(base64)로 변환"""
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {path}")
    try:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return "data:image/png;base64," + encoded
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading image: {e}")

async def _read_csv_async(file_path: Path) -> pd.DataFrame:
    """CSV를 스레드에서 읽기 (이벤트 루프 비블로킹)"""
    def _read():
        return pd.read_csv(file_path, index_col="Date", parse_dates=["Date"]).fillna("NaN")
    return await asyncio.to_thread(_read)

# -------------------------------------------------
# 헬스체크 / 루트
# -------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "root_path": APP_ROOT_PATH or "/"}

@app.get("/")
def root():
    """
    index.html을 반환하되, 프리픽스가 있을 경우 <base href=".../">를 주입해
    정적 자원 경로 문제를 완화합니다.
    """
    index_html = PUBLIC_DIR / "index.html"
    if not index_html.exists():
        return {"message": "public/index.html not found. Place your frontend under /public or use /static."}

    html = index_html.read_text(encoding="utf-8", errors="ignore")

    rp = APP_ROOT_PATH or "/"
    # 이미 base가 없다면 <head> 바로 뒤에 주입
    if "<base" not in html.lower():
        html = html.replace("<head>", f'<head><base href="{rp if rp.endswith("/") else rp + "/"}">', 1)

    return HTMLResponse(content=html)

# -------------------------------------------------
# 업로드/예측
# -------------------------------------------------
@router.post("/upload")
async def post_data_set(file: UploadFile = File(...)):
    """
    CSV 업로드 → 두 LSTM 모델(weight_used_model, model)로 예측 수행
    - 무거운 연산은 모두 스레드로 오프로드하여 서버 반응성 유지
    - 모델 모듈은 요청 시 동적 임포트(스타트업 블로킹 방지)
    """
    try:
        # 1) 저장 경로 구성
        current_time = datetime.now(timezone).strftime("%Y%m%d_%H%M%S")
        new_filename = f"{current_time}_{file.filename}"
        file_location = Path(UPLOAD_DIR) / new_filename

        # 2) 업로드 파일 저장
        contents = await file.read()
        await asyncio.to_thread(file_location.write_bytes, contents)

        # 3) CSV 로드
        dataset = await _read_csv_async(file_location)

        # 4) 모듈 지연 임포트
        weight_mod = importlib.import_module(".weight_used_model", package=__package__)
        model_mod = importlib.import_module(".model", package=__package__)

        # 5) 예측 실행 (스레드 오프로드)
        result_visualizing_LSTM, result_evaluating_LSTM = await asyncio.to_thread(weight_mod.process, dataset)
        result_visualizing_LSTM_v2, result_evaluating_LSTM_v2 = await asyncio.to_thread(model_mod.process, dataset)

        # 6) 결과 이미지 존재 확인
        img1 = Path(result_visualizing_LSTM)
        img2 = Path(result_visualizing_LSTM_v2)
        if not img1.exists():
            raise HTTPException(status_code=500, detail=f"File not found: {img1}")
        if not img2.exists():
            raise HTTPException(status_code=500, detail=f"File not found: {img2}")

        return {
            "result_visualizing_LSTM": _b64_png(img1),
            "result_evaluating_LSTM": result_evaluating_LSTM,
            "result_visualizing_LSTM_v2": _b64_png(img2),
            "result_evaluating_LSTM_v2": result_evaluating_LSTM_v2,
            "saved_filename": new_filename,
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------
# 다운로드/뷰
# -------------------------------------------------
@router.get("/download")
async def download():
    """weight_used_model이 생성한 stock 예측 이미지를 다운로드"""
    try:
        weight_mod = importlib.import_module(".weight_used_model", package=__package__)
        img_name = Path(IMAGE_DIR) / weight_mod.get_stock_png()
        if not img_name.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {img_name}")
        return FileResponse(path=str(img_name), media_type="application/octet-stream", filename="stock.png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download_shapes")
async def download_model_architecture_shapes():
    """weight_used_model이 생성한 모델 구조(shapes) 이미지를 다운로드"""
    try:
        weight_mod = importlib.import_module(".weight_used_model", package=__package__)
        img_name = Path(IMAGE_DIR) / weight_mod.get_model_shapes_png()
        if not img_name.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {img_name}")
        return FileResponse(path=str(img_name), media_type="application/octet-stream", filename="model_shapes.png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/view-download")
async def view_downloaded_image():
    """weight_used_model이 생성한 stock 예측 이미지를 HTML로 보기"""
    try:
        weight_mod = importlib.import_module(".weight_used_model", package=__package__)
        img_name = Path(IMAGE_DIR) / weight_mod.get_stock_png()
        img_base64 = _b64_png(img_name)
        return HTMLResponse(
            content=f"""
            <html>
                <body>
                    <h1>Downloaded Stock Prediction Image</h1>
                    <img src="{img_base64}" alt="Stock Prediction Image" />
                </body>
            </html>
            """
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(router)

# 실행 명령어 예시: 순서대로 백엔드 띄운 후, 프론트엔드 띄우기
# python -m uvicorn main:app --port 8001 --reload
# http://localhost:8001/static/index.html
