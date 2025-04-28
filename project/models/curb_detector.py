import numpy as np
# from services.output_queue import add_to_queue

# def process_obstacles(yolo_result: dict, depth_result: dict):
#     """
#     YOLO + DepthMap을 기반으로 장애물 위험 판단

#     """
import cv2
def detect_curbs(roi_depth_result: dict, frame: np.ndarray):
    """#
    YOLO 결과를 받아 프레임에 박스를 그리고 화면에 출력

    Args:
        yolo_result (dict): {"objects": [{"label": str, "bbox": [x1, y1, x2, y2]}, ...]}
        frame (np.ndarray): 원본 이미지 프레임 (OpenCV BGR)
    """
  # 여기서 화면에 보여준다
    resize_scale = 1  # (50% 크기로 축소), 0.3으로 하면 더 작게
    resized_frame = cv2.resize(frame, None, fx=resize_scale, fy=resize_scale)

    # ROI Depth Map 출력
    if roi_depth_result is not None:
        norm_roi_depth = cv2.normalize(roi_depth_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        color_roi_depth = cv2.applyColorMap(norm_roi_depth, cv2.COLORMAP_MAGMA)
        resized_roi_depth = cv2.resize(color_roi_depth, None, fx=resize_scale, fy=resize_scale)
        cv2.imshow('ROI Depth Map', resized_roi_depth)
