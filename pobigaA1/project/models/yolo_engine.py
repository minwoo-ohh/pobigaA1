# import torch
# import cv2

# # 모델 로드 (YOLOv5s 기준, 처음 한 번만)
# yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
# yolo_model.eval()

# def run_yolo(frame):
#     """
#     YOLOv5를 이용해 객체 감지 수행

#     Args:
#         frame (np.ndarray): 입력 이미지 (OpenCV BGR 이미지)

#     Returns:
#         dict: {"objects": [{"label": str, "bbox": [x1, y1, x2, y2]}, ...]}
#     """
#     # BGR → RGB 변환
#     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # 추론
#     results = yolo_model(img_rgb)

#     detected_objects = []
#     for *box, conf, cls in results.xyxy[0].tolist():
#         x1, y1, x2, y2 = map(int, box)
#         label = yolo_model.names[int(cls)]
#         detected_objects.append({"label": label, "bbox": [x1, y1, x2, y2]})

#     return {"objects": detected_objects}
