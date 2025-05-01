# models/vlm_long.py

import sys
import os
import json
import time
from PIL import Image
import openai

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# 현재 파일 기준으로 상위 폴더 → test/test_images 경로 계산
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
image_dir = os.path.join(base_dir, "test", "test_images")

from mobilevlm_runtime import MobileVLMRuntime

# ========== 환경 설정 ==========
openai.api_key = os.getenv("OPENAI_API_KEY")

# VLM 모델 초기화
runtime = MobileVLMRuntime()

# 결과 저장 경로
jsonl_path = "../logs/vlm_long_log.jsonl"
diary_txt_path = "../logs/diary_entry.txt"
# diary_mp3_path = "/home/piai/AI_project/diary/diary_entry.mp3"
os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
os.makedirs(os.path.dirname(diary_txt_path), exist_ok=True)

# ========== 이미지 한 장 처리 함수 ==========
def run_once(image_path: str, kor_command: str):
    start_total = time.time()

    prompt = f"Describe this image in one complete sentence."

    # Step 2: VLM 추론
    result, duration = runtime.run(image_path, prompt)

    # # Step 4: TTS 저장 및 재생
    # tts = gTTS(translated_back, lang='ko')
    # tts.save("output.mp3")
    # os.system("mpg123 output.mp3")

    end_total = time.time()
    total_time = end_total - start_total

    # Step 5: 출력
    print(f"\n Image: {image_path}")
    print(f" English Prompt: {prompt}")
    print(f" Output (EN): {result}")
    print(f" Inference Time: {duration:.2f} seconds (Model only)")
    print(f" Total Time: {total_time:.2f} seconds (Translation + Model)")

    # Step 6: JSONL 저장
    json_data = {
        "image_path": image_path,
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

    print("\n📨 Requesting GPT for diary generation...")

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

    # print("\n Converting diary to MP3...")
    # tts = gTTS(diary_text, lang='ko')
    # tts.save(diary_mp3_path)
    # print(f"\n MP3 저장 완료: {diary_mp3_path}")

# ========== 실행 ==========
if __name__ == "__main__":
    # image_dir = "../test/test_images"
    for fname in sorted(os.listdir(image_dir)):
        if fname.lower().endswith(".jpg"):
            image_path = os.path.join(image_dir, fname)
            run_once(image_path, "앞에 뭐 있어?")

    generate_diary_and_tts()