import sys
import os

# config.py가 있는 루트 디렉토리를 sys.path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from mobilevlm_runtime import MobileVLMRuntime
from transformers import pipeline
from deep_translator import GoogleTranslator
import time

# 번역기 초기화 (1회만)
translator_ko2en = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en")  # Hugging Face 파이프라인
translator_en2ko = GoogleTranslator(source='en', target='ko')  # 구글 번역기

# VLM 모델 초기화 (1회만)
runtime = MobileVLMRuntime()

def run_once(image_path: str, kor_command: str):
    start_total = time.time()

    # Step 1: 한→영 번역
    translated = translator_ko2en(kor_command, max_length=60)[0]["translation_text"]
    prompt = f""" '{translated}' """

    # Step 2: 모델 추론 실행
    result, duration = runtime.run(image_path, prompt)

    # Step 3: 영→한 번역
    translated_back = translator_en2ko.translate(result)

    # # Step 4: TTS 저장 및 재생
    # tts = gTTS(translated_back, lang='ko')
    # tts.save("output.mp3")
    # os.system("mpg123 output.mp3")

    end_total = time.time()
    total_time = end_total - start_total

    # Step 5: 출력
    print(f"\n Image: {image_path}")
    print(f"Korean Prompt: {kor_command}")
    print(f"English Prompt: {prompt}")
    print(f"Output (EN): {result}")
    print(f"Output (KO): {translated_back}")
    print(f"Inference Time: {duration:.2f} seconds (Model only)")
    print(f"Total Time: {total_time:.2f} seconds (Translation + Model)")

if __name__ == "__main__":
    # 예시 실행
    run_once("test/test_images/test2.jpg", "앞에 뭐 있어?")