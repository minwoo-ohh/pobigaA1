# models/vlm_short.py
import os
import time
from transformers import pipeline
from deep_translator import GoogleTranslator
from models.mobilevlm_runtime import MobileVLMRuntime

# 번역기 및 모델은 전역 객체로 한 번만 초기화
translator_ko2en = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en")
translator_en2ko = GoogleTranslator(source='en', target='ko')
runtime = MobileVLMRuntime()

def run_short_vlm(image_path: str, kor_command: str) -> str:
    start_total = time.time()

    # 번역
    translated = translator_ko2en(kor_command, max_length=60)[0]["translation_text"]
    prompt = f"'{translated}'"

    # 추론
    result, duration = runtime.run(image_path, prompt)

    # 다시 번역
    translated_back = translator_en2ko.translate(result)

    end_total = time.time()
    total_time = end_total - start_total

    print(f"\nImage: {image_path}")
    print(f"Korean Prompt: {kor_command}")
    print(f"English Prompt: {prompt}")
    print(f"Output (EN): {result}")
    print(f"Output (KO): {translated_back}")
    print(f"Inference Time: {duration:.2f} sec")
    print(f"Total Time: {total_time:.2f} sec")

    return translated_back
