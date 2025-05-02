# services/frame_saver.py

import os
import cv2
import time
from datetime import datetime

FRAME_DIR = "storage/frames"
FRAME_KEEP_SEC = 20  # 20초 유지
os.makedirs(FRAME_DIR, exist_ok=True)

frame_log = []  # (timestamp, filepath)

def clean_old_frames():
    """
    20초 이상된 오래된 프레임 삭제
    """
    now = time.time()
    while frame_log and now - frame_log[0][0] > FRAME_KEEP_SEC:
        _, old_path = frame_log.pop(0)
        if os.path.exists(old_path):
            os.remove(old_path)
            
last_saved_time = 0

def save_frame(frame):
    global last_saved_time
    timestamp = time.time()

    # 1초마다만 저장
    if timestamp - last_saved_time >= 1:
        filename = f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        path = os.path.join(FRAME_DIR, filename)
        cv2.imwrite(path, frame)

        frame_log.append((timestamp, path))
        last_saved_time = timestamp
        clean_old_frames()  # 저장할 때 바로 정리

    return path



def start_frame_capture():
    """
    주기적으로 오래된 프레임 삭제하는 루프
    """
    print("[FRAME SAVER] 프레임 자동 삭제 루프 시작")
    while True:
        clean_old_frames()
        time.sleep(5)