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
            frame_vlm = frame
            frame=cv2.resize(frame, (640, 640))

            # 👇 ROI 정의 추가
            h, w, _ = frame.shape
            roi_y_start = int(h * 2 / 3)
            roi_y_end = h
            roi_x_start = 0
            roi_x_end = w
            ROI = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

            future_yolo = executor.submit(run_yolo, frame)
            future_depth = executor.submit(run_depth, frame)
            ROI_depth = executor.submit(run_depth, ROI)
            # future_vlm = executor.submit(run_vlm, frame_vlm)
            future_save = executor.submit(save_frame, frame_vlm)

            yolo_result = future_yolo.result()
            depth_result = future_depth.result()
            ROI_depth_result = ROI_depth.result()

            process_obstacles(yolo_result,depth_result,frame)
            detect_curbs(ROI_depth_result,frame)
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