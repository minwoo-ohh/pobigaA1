# models/vlm_long.py

import os
import json
import time
import threading
from PIL import Image
import numpy as np
import cv2

from models.mobilevlm_runtime import MobileVLMRuntime
import models.shared_state as shared_state

# VLM 모델 초기화
runtime = MobileVLMRuntime()

# 결과 저장 경로
jsonl_path = "logs/vlm_long_log.jsonl"
os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

# 이미지 한 장 처리 함수
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

# 프레임 추론 루프
def start_vlm_long_loop(interval=5.0, overwrite=False):
    print(f"[VLM_LONG] 시작됨: {interval}초부터 실시간 프레임 추론")

    if overwrite:
        print("[VLM_LONG] 기존 로그 파일 덮어쓰기 중...")
        open(jsonl_path, "w", encoding="utf-8").close()

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
