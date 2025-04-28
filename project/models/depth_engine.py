import torch
import cv2
import numpy as np
from PIL import Image
from transformers import DPTFeatureExtractor, DPTForDepthEstimation

# Depth 모델 로드 (처음 1회만)
DEPTH_MODEL_PATH = "Intel/dpt-hybrid-midas"

depth_extractor = DPTFeatureExtractor.from_pretrained(DEPTH_MODEL_PATH)
depth_model = DPTForDepthEstimation.from_pretrained(DEPTH_MODEL_PATH)
depth_model.to("cuda")
depth_model.eval()

def run_depth(frame):
    """
    Depth 모델을 이용해 Depth map 예측 수행

    Args:
        frame (np.ndarray): 입력 이미지 (OpenCV BGR 이미지)

    Returns:
        np.ndarray: 예측된 Depth map (2D float32 array)
    """
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = depth_extractor(images=image, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = depth_model(**inputs)
        depth = outputs.predicted_depth
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze().cpu().numpy()

    return depth
