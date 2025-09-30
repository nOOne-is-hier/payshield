# Credit Card Fraud Detection

## 🚀 개발 환경 세팅

본 프로젝트는 **Docker + NVIDIA GPU** 환경을 권장합니다.  
로컬 코드 수정 → 컨테이너 내부에서 즉시 반영되도록 **volume mount**를 사용합니다.

### 1. Docker 컨테이너 실행

터미널 환경(OS)에 맞는 명령어를 복사하여 실행합니다.

#### 1-1. NVIDIA GPU 환경 (권장)

* **Linux / macOS (bash/zsh 터미널)**
    ```bash
    docker run --rm -it --gpus all \
      -v "$(pwd):/workspace" \
      -w /workspace \
      pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime bash
    ```

* **Windows (VS Code 등 PowerShell 터미널)**
    ```bash
    docker run --rm -it --gpus all \
      -v "${PWD}:/workspace" \
      -w /workspace \
      pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime bash
    ```

#### 1-2. CPU 전용 환경

NVIDIA GPU가 없거나, GPU 설정이 어려운 경우 `--gpus all` 옵션을 제외하고 실행합니다. (PyTorch가 CPU 모드로 동작합니다.)

* **Linux / macOS (bash/zsh 터미널)**
    ```bash
    docker run --rm -it \
      -v "$(pwd):/workspace" \
      -w /workspace \
      pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime bash
    ```

* **Windows (VS Code 등 PowerShell 터미널)**
    ```bash
    docker run --rm -it \
      -v "${PWD}:/workspace" \
      -w /workspace \
      pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime bash
    ```

* `-v "$(pwd):/workspace"` : 현재 로컬 프로젝트 경로를 컨테이너의 `/workspace`로 마운트
    * (Windows PowerShell에서는 `"${PWD}"` 사용)
* `-w /workspace` : 컨테이너 실행 시 작업 디렉토리 지정

---

### 2. Python 패키지 설치

```bash
pip install -r requirements.txt
```

* `requirements.txt`는 `pip freeze > requirements.txt`로 생성
* 필요 시 새 패키지 설치 후 `pip freeze` 갱신

---

### 3. 실행 방법

예시 (Isolation Forest 파이프라인):

```bash
python main.py
```

AutoEncoder 파이프라인:

```bash
python run_ae_pipeline.py
```

PacMAP + Isolation Forest (실험적):

```bash
python run_pacmap_iso_pipeline.py
```

---

## 📂 프로젝트 구조

```
.
├── uploaded_files/         # train.csv, val.csv, test.csv
├── result/                 # 실행 후 저장되는 결과물(metrics.json, misclassified.csv 등)
├── main.py                 # IsolationForest 파이프라인
├── run_ae_pipeline.py      # AutoEncoder 파이프라인
├── run_pacmap_iso_pipeline.py # PacMAP+IF 파이프라인 (실험적)
├── anomaly_detector.py     # 공통 유틸
├── requirements.txt        # Python 환경 의존성
└── README.md
```

---

## 💡 참고

* 로컬에서 작업하면 컨테이너에 실시간 반영됨 (별도의 `docker cp` 불필요).
* GPU 사용 확인:

```bash
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```
