import cv2
from concurrent.futures import ThreadPoolExecutor
from models.yolo_engine import run_yolo
from models.depth_engine import run_depth
from models.vlm_long import run_vlm
from services.frame_saver import save_frame
from logs.log_vlm_output import log_vlm_output  # VLM 로그 저장
from models.obstacle_reasoner import process_obstacles
from models.curb_detector import detect_curbs
from services.gps_state import get_latest_gps

def process_video_stream():
    # 스트림 열기
    cap = cv2.VideoCapture("udp://127.0.0.1:5000", cv2.CAP_FFMPEG)  # @ 대신 127.0.0.1로 수정
    if not cap.isOpened():
        print("[ERROR] 스트림 연결 실패")
        return

    print("[STREAM] 영상 수신 시작")

    # 쓰레드 풀 생성
    executor = ThreadPoolExecutor(max_workers=4)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] 프레임 수신 실패")
                continue

            # 각 작업을 병렬로 실행
            future_yolo = executor.submit(run_yolo, frame)
            future_depth = executor.submit(run_depth, frame)
            future_vlm = executor.submit(run_vlm, frame)
            future_save = executor.submit(save_frame, frame)

            # 결과 수집
            yolo_result = future_yolo.result()
            depth_result = future_depth.result()
            vlm_text = future_vlm.result()
            frame_path = future_save.result()

            # 후처리 연결
            process_obstacles(yolo_result, depth_result)
            detect_curbs(depth_result)

            # VLM 결과 로그 저장
            gps = get_latest_gps()
            log_vlm_output(vlm_text, frame_path, gps)

    except KeyboardInterrupt:
        print("[STREAM] 수신 중단됨")
    finally:
        cap.release()
        executor.shutdown()
        cv2.destroyAllWindows()
