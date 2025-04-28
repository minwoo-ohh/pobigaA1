import os
import cv2
import time
from datetime import datetime

FRAME_DIR = "./storage/frames"
FRAME_KEEP_SEC = 1  # 1분 유지
os.makedirs(FRAME_DIR, exist_ok=True)

frame_log = []  # (timestamp, filepath)

def save_frame(frame):
    """
    프레임을 저장하고 경로를 반환
    """
    timestamp = time.time()
    filename = f"frame_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
    path = os.path.join(FRAME_DIR, filename)
    cv2.imwrite(path, frame)

    frame_log.append((timestamp, path))
    return path

def clean_old_frames():
    """
    60초 이상된 오래된 프레임 삭제
    """
    now = time.time()
    while frame_log and now - frame_log[0][0] > FRAME_KEEP_SEC:
        _, old_path = frame_log.pop(0)
        if os.path.exists(old_path):
            os.remove(old_path)
            print(f"[CLEANUP] 삭제됨: {old_path}")

def start_frame_capture():
    """
    주기적으로 오래된 프레임 삭제하는 루프
    """
    print("[FRAME SAVER] 프레임 자동 삭제 루프 시작")
    while True:
        clean_old_frames()
        time.sleep(5)
