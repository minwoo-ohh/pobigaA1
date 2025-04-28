from ultralytics import YOLO

# 모델 로드 (처음 1회만)
WEIGHTS_PATH = "/home/piai/ai/runs/detect/train3/weights/best.pt"
model = YOLO(WEIGHTS_PATH).to("cuda")
model.fuse()
model.model = model.model.half()

def run_yolo(frame):
    """
    YOLO를 이용해 객체 감지 수행

    Args:
        frame (np.ndarray): 입력 이미지 (OpenCV BGR 이미지)

    Returns:
        dict: {"objects": [{"label": str, "bbox": [x1, y1, x2, y2]}, ...]}
    """

    results = model.predict(
        source=frame,
        conf=0.3,
        iou=0.6,
        stream=False
    )

    objects = []
    for res in results:
        for box in res.boxes:
            label = res.names[int(box.cls)]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            objects.append({
                "label": label,
                "bbox": [x1, y1, x2, y2]
            })

    return {"objects": objects}
