import numpy as np
# from services.output_queue import add_to_queue

# def process_obstacles(yolo_result: dict, depth_result: dict):
#     """
#     YOLO + DepthMap을 기반으로 장애물 위험 판단

#     """
import cv2
def process_obstacles(yolo_result: dict,depth_result: dict, frame: np.ndarray):
    """#
    YOLO 결과를 받아 프레임에 박스를 그리고 화면에 출력

    Args:
        yolo_result (dict): {"objects": [{"label": str, "bbox": [x1, y1, x2, y2]}, ...]}
        frame (np.ndarray): 원본 이미지 프레임 (OpenCV BGR)
    """
    for obj in yolo_result["objects"]:
        label = obj["label"]
        x1, y1, x2, y2 = obj["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 여기서 화면에 보여준다
    resize_scale = 1  # (50% 크기로 축소), 0.3으로 하면 더 작게
    resized_frame = cv2.resize(frame, None, fx=resize_scale, fy=resize_scale)

    # 여기서 축소된 frame을 보여준다
    cv2.imshow('YOLO Detection', resized_frame)
    if depth_result is not None:
        norm_depth = cv2.normalize(depth_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        color_depth = cv2.applyColorMap(norm_depth, cv2.COLORMAP_MAGMA)
        resized_depth = cv2.resize(color_depth, None, fx=resize_scale, fy=resize_scale)

    # 화면에 Depth map 출력
        cv2.imshow('Depth Map', resized_depth)

