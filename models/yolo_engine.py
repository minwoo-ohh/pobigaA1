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
