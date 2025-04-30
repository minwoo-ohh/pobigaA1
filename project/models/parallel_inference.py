import cv2
from concurrent.futures import ThreadPoolExecutor
from models.yolo_engine import run_yolo
from models.depth_engine import run_depth
# from models.vlm_long import run_vlm
from services.frame_saver import save_frame
# from logs.log_vlm_output import log_vlm_output  # VLM 로그 저장
from models.obstacle_reasoner import process_obstacles
from models.curb_detector import detect_curbs
# from services.gps_state import get_latest_gps
import threading

stop_event = threading.Event()
import numpy as np
import cv2

def letterbox_image(image, target_size=(640, 640), color=(0,0,0)):
    """
    입력 이미지를 target_size로 비율 유지하면서 resize하고, 부족한 영역은 padding(색상 114)으로 채워준다.

    Args:
        image (np.ndarray): 입력 이미지 (OpenCV BGR)
        target_size (tuple): 원하는 출력 사이즈 (width, height)
        color (tuple): padding 색상 (기본 114,114,114)

    Returns:
        np.ndarray: letterbox 처리된 이미지
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # 스케일 비율 계산
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 원본 비율 유지하며 resize
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 배경(canvas) 만들기
    canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)

    # 가운데에 resized_image를 놓기
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w, :] = resized_image

    return canvas

def process_video_stream():
    video_path = '/home/piai/ai/remove.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] 영상 파일 열기 실패")
        exit()

    print("[VIDEO] 영상 처리 시작")
    
    executor = ThreadPoolExecutor(max_workers=2)

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] 프레임 수신 실패, 영상 종료")
                break  # 여기서 깨끗하게 루프 종료시킴
            frame_ori = frame.copy()
            frame_yolo = letterbox_image(frame, (640, 640))
            frame_depth = cv2.resize(frame,(640,360))
            # ROI 정의 추가
            h, w, _ = frame_ori.shape
            roi_y_start = int(h * 2 / 3)
            roi_y_end = h
            roi_x_start = int(w * 2 / 5) 
            roi_x_end = int(w * 4 / 5) 
            ROI = frame_ori[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

            future_yolo = executor.submit(run_yolo, frame_yolo)
            future_depth = executor.submit(run_depth, frame_depth)
            ROI_depth = executor.submit(run_depth, ROI)
            # future_vlm = executor.submit(run_vlm, frame_ori)
            future_save = executor.submit(save_frame, frame_ori)

            yolo_result = future_yolo.result()
            depth_result = future_depth.result()
            ROI_depth_result = ROI_depth.result()

            process_obstacles(yolo_result,depth_result,frame_yolo)
            detect_curbs(ROI_depth_result,frame_depth)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] 사용자 종료")
                stop_event.set()
                break

    except KeyboardInterrupt:
        print("[STREAM] 키보드 인터럽트 감지, 수신 중단됨")
        stop_event.set()

    finally:
        cap.release()
        executor.shutdown(wait=True)  # 여기 wait=True 중요
        cv2.destroyAllWindows()