# models/vlm_long.py

import sys
import os
import json
import time
import threading
from PIL import Image
import openai
import numpy as np
import cv2

from models.mobilevlm_runtime import MobileVLMRuntime
import models.shared_state as shared_state

# ========== 환경 설정 ==========
openai.api_key = os.getenv("OPENAI_API_KEY")

# VLM 모델 초기화
runtime = MobileVLMRuntime()

# 결과 저장 경로
jsonl_path = "logs/vlm_long_log.jsonl"
diary_txt_path = "logs/diary_entry.txt"
os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
os.makedirs(os.path.dirname(diary_txt_path), exist_ok=True)

# ========== 이미지 한 장 처리 함수 ==========
def run_vlm(frame):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    prompt = "Describe this image in one complete sentence."
    start_total = time.time()

    result, duration = runtime.run(pil_image, prompt)

    end_total = time.time()
    total_time = end_total - start_total

    print(f"[VLM] {prompt} → {result} ({round(total_time,2)}s)")

    json_data = {
        "output_en": result,
        "inference_time": round(duration, 2),
        "total_time": round(total_time, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(json_data, ensure_ascii=False) + "\n")

# ========== 일기 생성 + TTS 저장 함수 ==========
def generate_diary_and_tts():
    descriptions = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            descriptions.append(data["output_en"])

    if not descriptions:
        print("설명이 존재하지 않습니다. 일기를 생성할 수 없습니다.")
        return

    joined_desc = " ".join(descriptions)
    prompt = f"""
The following are brief English descriptions generated from images. 
Please write a warm and emotional **Korean diary** that reflects a day for a visually impaired person, based on these scenes.

 Instructions:
- Write exactly **7 sentences**, each on a new line.
- Use **first-person expressions** such as "나는", "오늘은", "산책을 하다가", etc.
- Ensure the diary flows **in natural timeline order**, as if reflecting on a full day from morning to night.
- Use **clear and simple Korean**, easy for a visually impaired listener to understand when heard aloud.
- Emphasize **sensory experiences** such as sounds, feelings, textures, or smells over visual details.
- Avoid repeating words or describing similar things more than once.
- In the **last (7th) sentence**, clearly finish the diary by **summarizing the day emotionally** or **sharing a personal reflection**.

 Scene descriptions (in English):
{joined_desc}

 Korean Diary:
"""

    print("\n Requesting GPT for diary generation...")

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "너는 감성적인 한국어 일기 작가야."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500,
    )

    diary_text = response.choices[0].message.content.strip()
    print("\n 생성된 일기:\n", diary_text)

    with open(diary_txt_path, "w", encoding="utf-8") as f:
        f.write(diary_text)

# ========== VLM 루프 ==========
def start_vlm_long_loop(interval=5.0):
    print(f"[VLM_LONG] 시작됨: {interval}초부터 실시간 프레임 추론")
    
    if overwrite:
        print("[VLM_LONG] 기존 로그 파일 덮어쓰기 중...")
        open("logs/vlm_long_log.jsonl", "w", encoding="utf-8").close()
        open("logs/diary_entry.txt", "w", encoding="utf-8").close()

    last_infer_time = 0

    while not shared_state.stop_event.is_set():
        current_time = time.time()
        elapsed = current_time - last_infer_time

        with shared_state.latest_frame_lock:
            frame = shared_state.latest_frame.copy() if shared_state.latest_frame is not None else None

        if frame is not None and elapsed >= interval:
            print(f"[VLM] 프레임 수신됨, 추론 시작 ({round(elapsed, 2)}s 이후)")
            last_infer_time = time.time()
            frame_copy = np.array(frame)

            def run_and_log():
                try:
                    run_vlm(frame_copy)
                except Exception as e:
                    print(f"[VLM][ERROR] {e}")

            threading.Thread(target=run_and_log, daemon=True).start()

        time.sleep(0.05)
