# models/obstacle_reasoner.py

import cv2
import numpy as np
from models.tts_engine import enqueue_tts

prev_positions = {}
prev_x_history = {}
warned_ids = set()
prev_widths = {}
LABEL_KOR = {
    "person": "사람",
    "car": "차",
    "truck": "트럭",
    "bus": "버스",
    "bicycle": "자전거",
    "motorcycle": "오토바이",
    "bench": "벤치",
    # 필요한 만큼 추가 가능
}

def is_bbox_in_roi(bbox, roi_polygon):
    x1, y1, x2, y2 = bbox
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    for pt in corners:
        if cv2.pointPolygonTest(roi_polygon, pt, False) >= 0:
            return True
    return False

def compute_slope(history):
    n = len(history)
    x_indices = list(range(n))
    mean_x = sum(x_indices) / n
    mean_y = sum(history) / n

    numerator = sum((x_indices[i] - mean_x) * (history[i] - mean_y) for i in range(n))
    denominator = sum((x_indices[i] - mean_x) ** 2 for i in range(n))

    return numerator / denominator if denominator != 0 else 0

def process_obstacles(yolo_result: dict, frame: np.ndarray):
    global prev_positions, prev_x_history, warned_ids

    h, w = frame.shape[:2]
    from models.parallel_inference import get_roi_polygon
    roi_polygon = get_roi_polygon(w, h)
    
    top_left, top_right = roi_polygon[1], roi_polygon[2]
    top_left_x = top_left[0]
    top_right_x = top_right[0]
    center_x = w // 2

    current_ids = set()

    for obj in yolo_result["objects"]:
        label = obj["label"]
        bbox = obj["bbox"]
        obj_id = obj.get("id", None)

        if obj_id is None:
            continue

        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        center = (cx, cy)
        current_ids.add(obj_id)

        # ROI 시각화
        in_roi = is_bbox_in_roi(bbox, roi_polygon)
        if in_roi:
            prev = prev_positions.get(obj_id)
            if prev:
                dx, dy = cx - prev[0], cy - prev[1]
                dist = np.sqrt(dx**2 + dy**2)

                if dist > 2:
                    # 기본 설정값
                    scale = 3.0
                    min_len = 25  # 최소 화살표 길이
                    max_len = 100  # 최대 화살표 길이

                    # 방향 벡터 정규화
                    norm = np.sqrt(dx**2 + dy**2)
                    dx_n, dy_n = dx / norm, dy / norm

                    # 최종 화살표 길이 결정
                    arrow_len = min(max(dist * scale, min_len), max_len)

                    new_prev = (
                        int(cx - dx_n * arrow_len),
                        int(cy - dy_n * arrow_len)
                    )

                    cv2.arrowedLine(frame, new_prev, center, (0, 0, 255), 10, tipLength=0.3)

            prev_positions[obj_id] = center

        # 박스 + 라벨
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"{label} {obj_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # 중심 x 히스토리 저장
        bbox_width = x2 - x1

        # 너비 히스토리 저장 (새 방식)
        width_history = prev_widths.get(obj_id, [])
        width_history.append(bbox_width)
        if len(width_history) > 10:
            width_history.pop(0)
        prev_widths[obj_id] = width_history

        avg_width = sum(width_history) / len(width_history)
        if bbox_width < avg_width * 0.8:
            continue

        # 중심 x좌표 히스토리 저장 (기존 유지)
        history = prev_x_history.get(obj_id, [])
        history.append(cx)
        if len(history) > 3:
            history.pop(0)
        prev_x_history[obj_id] = history

        # 경고 판단
        if in_roi and obj_id not in warned_ids and len(history) >= 3:
            if label not in ["person", "car", "truck","bicycle","motorcycle","bus","bench"]:
                continue
            direction = None
            slope = compute_slope(history)

            if top_left_x <= cx <= top_right_x:
                direction = "정면"
            elif cx < top_left_x and slope > 5:  # 왼쪽 → 중앙으로 접근 중
                direction = "왼쪽"
            elif cx > top_right_x and slope < -5:  # 오른쪽 → 중앙으로 접근 중
                direction = "오른쪽"

            if direction:
                label_kor = LABEL_KOR.get(label, label)  # 못 찾으면 원래 이름 유지
                print(f"{label_kor},{obj_id},{direction},{slope},true")
                warned_ids.add(obj_id)
                enqueue_tts(f"{direction}에 {label_kor} 있어", priority=0)

        # ROI 밖이면 추적 제거
        if not in_roi:
            prev_positions.pop(obj_id, None)

    # ID 정리
    prev_positions = {k: v for k, v in prev_positions.items() if k in current_ids}
    prev_x_history = {k: v for k, v in prev_x_history.items() if k in current_ids}
    warned_ids = {k for k in warned_ids if k in current_ids}
