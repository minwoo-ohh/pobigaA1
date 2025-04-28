# POBIGA-A1

보행 보조 및 상황 설명을 위한 실시간 AI 시스템입니다.  
YOLOv8, DPT, MobileVLM, GPT API, 카카오네비 API 등을 통합하여 장애물 감지, 길 안내, 일기 생성 기능을 제공합니다.

---

## 🛠️ 설치 방법 (Installation)

가상환경을 생성하고 필요한 라이브러리를 설치합니다.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

※ PyTorch 설치는 CUDA 환경에 맞춰 설정되어 있습니다.  
※ 시스템에 `ffmpeg`이 필요합니다. (`sudo apt install ffmpeg`)

---

## 🚀 실행 방법 (Usage)

**메인 실행 파일**:

```bash
python main.py
```

**또는** 메뉴에서 개별 테스트를 진
포함할 수 있습니다. (예: `vlm_short`, `yolo_engine` 등)

---

## 📁 프로젝트 구조 (Project Structure)

```
project/
├── models/             # 모델 및 엔진 코드 (YOLO, DPT, VLM 등)
├── services/           # 서버/수신 모듈
├── storage/            # 임시 데이터 저장
├── utils/              # 공통 유틸리티 (예정)
├── config.yaml         # 설정 파일
├── main.py             # 메인 실행 파일
├── requirements.txt    # 필수 라이브러리 목록
└── README.md           # 프로젝트 설명서
```

---

## 📋 추가 정보

- YOLOv8: Ultralytics 오픈소스 사용
- MobileVLM: SCUT-DLVCLab 공개 모델 사용
- DPT: Intel-ISL MiDaS 기반
- GPT API, 카카오네비 API 연동 예정

---

## ⚡ 주의사항

- GPU 사용을 권장합니다. (NVIDIA CUDA 11.8 이상)
- 메모리 요구사항이 높을 수 있습니다 (8GB 이상 추천).

---

## 📄 라이센스

본 프로젝트는 연구 및 학습 목적에 한하여 자유롭게 사용할 수 있습니다.
