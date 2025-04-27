# # models/curb_detector.py

# import numpy as np
# import cv2

# def detect_curbs(depth_result: dict):
#     """
#     뎁스맵을 이용해서 턱(curbs)이 있는지 감지하는 함수

#     Args:
#         depth_result (dict): run_depth에서 반환된 결과. {"depth_map": np.array}

#     Returns:
#         bool 또는 dict: 턱이 감지되었는지 여부, 또는 턱 위치 정보
#     """
#     depth_map = depth_result.get("depth_map")
#     if depth_map is None:
#         return

#     # ROI 영역 선택 (예: 하단 중앙 영역만 분석)
#     h, w = depth_map.shape
#     roi = depth_map[int(h*0.75):, int(w*0.4):int(w*0.6)]  # 하단 중앙

#     # 깊이 변화가 급격한 부분 찾기
#     edges = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=5)  # x축 방향 변화
#     variation = np.std(edges)

#     if variation > 20:  # 임계값은 실험적으로 조정
#         print("[CURB] 턱 감지됨!")
#         return True
#     else:
#         return False
