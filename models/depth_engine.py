import sys
sys.path.append('./MiDaS')

import torch
import cv2
import numpy as np
from collections import deque
from torchvision import transforms
from midas.dpt_depth import DPTDepthModel  # MiDaS repo에서 import

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
model_path = "./MiDaS/weights/dpt_hybrid_384.pt"
model = DPTDepthModel(path=None, backbone="vitb_rn50_384", non_negative=True)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# 변환기 정의
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

# 최근 뎁스 저장 큐
depth_queue = deque(maxlen=1)

# 추론 함수
def run_depth(frame: np.ndarray) -> np.ndarray:
    try:
        input_tensor = transform(frame).unsqueeze(0).to(device)

        with torch.no_grad():
            depth_map = model(input_tensor)

        depth_map = depth_map.squeeze().cpu().numpy()
        depth_resized = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)

        # 큐에 추가
        depth_queue.append(depth_resized)
        avg_depth = np.mean(depth_queue, axis=0)

        if np.isnan(avg_depth).any() or np.isinf(avg_depth).any():
            print("[DEPTH] 평균 뎁스맵에 NaN/INF 포함, 0맵 반환")
            return np.zeros_like(depth_resized)

        return avg_depth.astype(np.float32)

    except Exception as e:
        print(f"[DEPTH] DPT-Hybrid 추론 실패: {e}")
        return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
