import cv2
import numpy as np

# 이전 중심 좌표, 히스토리, 경고 기록
prev_positions = {}
prev_x_history = {}
warned_ids = set()
prev_widths = {}
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
                    cv2.arrowedLine(frame, prev, center, (0, 0, 255), 10, tipLength=0.5)
            prev_positions[obj_id] = center

        # 박스 + 라벨
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, f"{label} {obj_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # 중심 x 히스토리 저장
        bbox_width = x2 - x1
        prev_w = prev_widths.get(obj_id, bbox_width)
        width_change = bbox_width - prev_w
        prev_widths[obj_id] = bbox_width
        history = prev_x_history.get(obj_id, [])
        history.append(cx)
        if len(history) > 3:
            history.pop(0)
        prev_x_history[obj_id] = history
        if width_change < -5:
            continue
        # 경고 판단
        if in_roi and obj_id not in warned_ids and len(history) >= 3:
            direction = None
            slope = compute_slope(history)

            if top_left_x <= cx <= top_right_x:
                direction = "정면"
            elif cx < top_left_x and slope > 5:  # 왼쪽 → 중앙으로 접근 중
                direction = "왼쪽 방향"
            elif cx > top_right_x and slope < -5:  # 오른쪽 → 중앙으로 접근 중
                direction = "오른쪽 방향"

            if direction:
                print(f"{label},{obj_id},{direction},{slope},true")
                warned_ids.add(obj_id)
                # 추후: generate_warning(f"{direction}에 {label} 있음")

        # ROI 밖이면 추적 제거
        if not in_roi:
            prev_positions.pop(obj_id, None)

    # ID 정리
    prev_positions = {k: v for k, v in prev_positions.items() if k in current_ids}
    prev_x_history = {k: v for k, v in prev_x_history.items() if k in current_ids}
    warned_ids = {k for k in warned_ids if k in current_ids}
