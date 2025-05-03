# video_output.py

import cv2
import numpy as np
from ultralytics import YOLO

# 모델 로드 (처음 1회만)
WEIGHTS_PATH = "yolov8m.pt"
model = YOLO(WEIGHTS_PATH).to("cuda")
model.fuse()
model.model = model.model.half()

def run_yolo(frame):
    """
    YOLO를 이용해 객체 감지 수행
    ByteTrack을 활용한 객체 감지 + 추적 ID 포함
    Args:
        frame (np.ndarray): 입력 이미지 (OpenCV BGR 이미지)

    Returns:
        dict: {"objects": [{"label": str, "bbox": [x1, y1, x2, y2]}, ...]}
    """

    results = model.track(
        source=frame,
        persist=True,
        stream=False,
        tracker="bytetrack.yaml",  # 중요!
        conf=0.3,
        iou=0.6,
        verbose=False
    )

    objects = []
    for res in results:
        for box in res.boxes:
            label = res.names[int(box.cls)]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            obj_id = int(box.id) if box.id is not None else None
            objects.append({
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "id": obj_id  # 반드시 추가해야 함!!
            })

    return {"objects": objects}

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

# 객체 이동 방향 화살표 그리기
prev_positions = {}
def draw_arrow(frame, obj_id, cx, cy):
    if obj_id in prev_positions:
        prev_cx, prev_cy = prev_positions[obj_id]
        dx, dy = cx - prev_cx, cy - prev_cy
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 5:  # 일정 거리 이상 이동 시 화살표 그리기
            norm = np.sqrt(dx**2 + dy**2)
            dx_n, dy_n = dx / norm, dy / norm
            arrow_len = min(dist, 50)  # 화살표 최대 길이 설정
            new_prev = (int(cx - dx_n * arrow_len), int(cy - dy_n * arrow_len))
            cv2.arrowedLine(frame, new_prev, (cx, cy), (0, 0, 255), 5, tipLength=0.3)
    prev_positions[obj_id] = (cx, cy)


def process_video_stream():
    # 비디오 캡처 열기
    video_path = 'test/video2.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERROR] 영상 파일 열기 실패")
        exit()

    # 입력 영상의 크기 (h, w)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 저장할 비디오 설정 (입력 영상의 해상도와 동일하게 설정)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' 또는 'XVID' 등
    out = cv2.VideoWriter('output1.mp4', fourcc, 30.0, (frame_width, frame_height))  # fps, 크기 주의

    print("[VIDEO] 영상 처리 시작")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] 프레임 수신 실패, 영상 종료")
            break

        # YOLO 결과 처리
        yolo_result = run_yolo(frame)

        # ROI 영역 시각화
        h, w, _ = frame.shape
        roi_polygon = get_roi_polygon(w, h)
        cv2.polylines(frame, [roi_polygon], isClosed=True, color=(0, 255, 255), thickness=2)

        # ROI 내 객체 이동 방향 화살표 표시
        for obj in yolo_result["objects"]:
            label = obj["label"]
            bbox = obj["bbox"]
            obj_id = obj["id"]
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            draw_arrow(frame, obj_id, cx, cy)

            # 바운딩 박스 및 라벨 표시
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {obj_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # 화면에 프레임을 표시
        cv2.imshow('Detection', frame)

        # 비디오로 저장
        out.write(frame)

        # 'q' 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 사용자 종료")
            break

    # 자원 해제
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# 실행
process_video_stream()
