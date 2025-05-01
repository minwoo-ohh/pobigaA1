# 🦯 POBIGA-A1

시각 장애인을 위한 실시간 보행 보조 및 상황 설명 AI 시스템입니다.  
YOLOv8, DPT, MobileVLM, GPT API, 카카오네비 API 등을 통합하여 **장애물 감지**, **길 안내**, **장면 설명**, **감성 일기 생성** 기능을 제공합니다.

---

## 🛠️ 설치 방법 (Installation)

### 1. 가상환경 생성

```bash
conda create -n pobigaA1 python=3.10
conda activate pobigaA1
```

### 2. PyTorch 설치 (CUDA 환경에 맞게 선택)

[👉 설치 가이드 확인하기](https://pytorch.org/get-started/locally/)

예: CUDA 11.8 환경에서는 다음과 같이 설치합니다:

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### 3. 기타 필수 라이브러리 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

※ `ffmpeg` 설치 필요:  
```bash
sudo apt install ffmpeg
```

---

## 🚀 실행 방법 (Usage)

### 메인 실행 파일

```bash
python main.py
```

### 개별 테스트 실행

아래 모듈들로 각각 테스트가 가능합니다:

- `vlm_short.py`: 짧은 지시문 기반 장면 설명
- `vlm_long.py`: 하루 단위 요약을 위한 이미지 캡셔닝 및 감성 일기 생성
- `yolo_engine.py`, `depth_engine.py`: 개별 감지 엔진 확인

---

## 📁 프로젝트 구조 (Project Structure)

```
pobigaA1/
├── models/             # 모델 실행 및 추론 엔진 (YOLO, DPT, VLM 등)
├── services/           # 수신 서버, TTS, GPS 처리 등
├── logs/               # 일기 텍스트 및 JSONL 기록
├── test/               # 테스트 이미지
├── utils/              # 공통 유틸리티 (예정)
├── mobilevlm_official/ # MobileVLM GitHub 복사본
├── main.py             # 전체 통합 실행 파일
├── requirements.txt    # 설치해야 할 Python 패키지
└── README.md           # 현재 문서
```

---

## 📋 사용 기술 요약

| 구성요소      | 기술/모델명                                      |
|---------------|--------------------------------------------------|
| 객체 감지     | YOLOv8 (Ultralytics)                             |
| 깊이 추정     | DPT (Intel-ISL MiDaS 기반)                       |
| 장면 설명     | MobileVLM (SCUT-DLVCLab, Lightweight VLM)       |
| 감성 일기 생성 | OpenAI GPT API                                  |
| 번역 및 음성  | deep-translator, gTTS                            |

---

## ⚠️ 주의사항

- GPU (CUDA 11.8 이상) 환경에서의 실행을 권장합니다.
- 메모리는 최소 8GB 이상 필요하며, 16GB 이상 권장됩니다.
- `.env` 파일에 OpenAI API 키가 설정되어 있어야 GPT 기능이 작동합니다.

```
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

---

## 📄 라이선스 (License)

본 프로젝트는 **연구 및 학습 목적**에 한해 자유롭게 사용할 수 있습니다.  
상업적 사용 및 재배포 시에는 별도 문의 바랍니다.
