# models/diary_generator.py

import os
import json
import openai
from dotenv import load_dotenv


# ========== 환경 설정 ==========
load_dotenv(dotenv_path=".env")
openai.api_key = os.getenv("OPENAI_API_KEY")

jsonl_path = "logs/vlm_long_log.jsonl"
diary_txt_path = "logs/diary_entry.txt"
os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
os.makedirs(os.path.dirname(diary_txt_path), exist_ok=True)

# ========== 일기 생성 함수 ==========
def generate_diary():
    descriptions = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            descriptions.append(data["output_en"])

    if not descriptions:
        print("[DIARY] 설명이 존재하지 않습니다. 일기를 생성할 수 없습니다.")
        return

    joined_desc = " ".join(descriptions)
    prompt = f"""
The following are brief English descriptions generated from images. 
Please write a warm and emotional **Korean diary** that reflects a day for a visually impaired person, based on these scenes.

 Instructions:
- Write exactly **7 sentences**, each on a new line.
- Use **first-person expressions** such as \"아니\", \"오늘은\", \"산책을 하다가\", etc.
- Ensure the diary flows **in natural timeline order**, as if reflecting on a full day from morning to night.
- Use **clear and simple Korean**, easy for a visually impaired listener to understand when heard aloud.
- Emphasize **sensory experiences** such as sounds, feelings, textures, or smells over visual details.
- Avoid repeating words or describing similar things more than once.
- In the **last (7th) sentence**, clearly finish the diary by **summarizing the day emotionally** or **sharing a personal reflection**.

 Scene descriptions (in English):
{joined_desc}

 Korean Diary:
"""

    print("\n[DIARY] GPT에게 일기 생성을 요청 중...")

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
    print("\n[DIARY] 생성 완료:\n")
    print(diary_text)

    with open(diary_txt_path, "w", encoding="utf-8") as f:
        f.write(diary_text)

    print(f"\n[DIARY] 파일 저장완료: {diary_txt_path}")

if __name__ == "__main__":
    generate_diary()
