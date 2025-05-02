# config.py

from dotenv import load_dotenv
import os
import sys

# .env 파일 로드
load_dotenv(dotenv_path='/home/piai/ai_p/pobigaA1/.env')

# 환경 변수 읽기
openai_api_key = os.getenv("OPENAI_API_KEY")
python_path = os.getenv("PYTHONPATH")

# PYTHONPATH가 있으면 sys.path에 추가
if python_path:
    sys.path.append(python_path)