import cv2
import numpy as np
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

#객체 탐지용 ROI
def get_roi_polygon(frame_width, frame_height):
    top_w_ratio = 0.15
    bottom_w_ratio = 0.8
    top_y_ratio = 0.65
    bottom_y_ratio = 1

    top_y = int(frame_height * top_y_ratio)
    bottom_y = int(frame_height * bottom_y_ratio)
    top_left = (int(frame_width * (1 - top_w_ratio) / 2), top_y)
    top_right = (int(frame_width * (1 + top_w_ratio) / 2), top_y)
    bottom_left = (int(frame_width * (1 - bottom_w_ratio) / 2), bottom_y)
    bottom_right = (int(frame_width * (1 + bottom_w_ratio) / 2), bottom_y)

    return np.array([bottom_left, top_left, top_right, bottom_right], dtype=np.int32)

#턱 탐지용 ROI
def curb_roi_polygon(frame_width, frame_height):
    top_w_ratio = 0.1
    bottom_w_ratio = 0.1
    top_y_ratio = 0.75
    bottom_y_ratio = 0.95

    top_y = int(frame_height * top_y_ratio)
    bottom_y = int(frame_height * bottom_y_ratio)
    top_left = (int(frame_width * (1 - top_w_ratio) / 2), top_y)
    top_right = (int(frame_width * (1 + top_w_ratio) / 2), top_y)
    bottom_left = (int(frame_width * (1 - bottom_w_ratio) / 2), bottom_y)
    bottom_right = (int(frame_width * (1 + bottom_w_ratio) / 2), bottom_y)

    return np.array([bottom_left, top_left, top_right, bottom_right], dtype=np.int32)

#비디오 스트리밍
def process_video_stream():
    video_path = '/home/piai/ai_p/remove.mp4'
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
            
            # 👇 ROI 정의 추가
            h, w, _ = frame_ori.shape
            roi_polygon = get_roi_polygon(w, h)
            cv2.polylines(frame_ori, [roi_polygon], isClosed=True, color=(0, 255, 255), thickness=2)

            curb_polygon = curb_roi_polygon(w, h)
            curb_mask = np.zeros_like(frame_ori[:, :, 0], dtype=np.uint8)
            cv2.fillPoly(curb_mask, [curb_polygon], 255)
            curb_frame = cv2.bitwise_and(frame_ori, frame_ori, mask=curb_mask)
            x, y, w_box, h_box = cv2.boundingRect(curb_polygon)
            ROI = curb_frame[y:y+h_box, x:x+w_box]

            cv2.rectangle(frame_ori, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
            
            future_yolo = executor.submit(run_yolo, frame_ori)
            ROI_depth = executor.submit(run_depth, ROI)
            # future_vlm = executor.submit(run_vlm, frame_ori)
            future_save = executor.submit(save_frame, frame_ori)

            yolo_result = future_yolo.result()
            # depth_result = future_depth.result()
            ROI_depth_result = ROI_depth.result()

            process_obstacles(yolo_result,frame_ori)
            detect_curbs(ROI_depth_result,ROI)
            
            resize_scale = 0.3  # (50% 크기로 축소), 0.3으로 하면 더 작게
            resized_frame = cv2.resize(frame_ori, None, fx=resize_scale, fy=resize_scale)
            
            cv2.imshow("Detection", resized_frame)
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