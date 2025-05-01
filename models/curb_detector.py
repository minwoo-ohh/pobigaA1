import numpy as np
import cv2
import time
from collections import deque
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from models.tts_engine import enqueue_tts
# 상태 저장용 deque
gradient_history = deque(maxlen=5)
feature_queue = deque(maxlen=30)
grad_deviation_history = deque(maxlen=20)

# 최근 턱 감지 시각
last_curb_time = 0
curb_cooldown = 3  # 3초

# CNN 모델 설정 (ResNet18 특징 추출기)
resnet = models.resnet18(pretrained=True)
resnet.fc = torch.nn.Identity()  # 마지막 분류기 제거
resnet.eval()

# CNN 입력 전처리
feature_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# CNN 특징 추출
def extract_feature(image: np.ndarray) -> torch.Tensor:
    tensor = feature_transform(image).unsqueeze(0)
    with torch.no_grad():
        return resnet(tensor).squeeze()

# 턱 감지 함수
def detect_curbs(
    depth_map: np.ndarray, 
    rgb_patch: np.ndarray, 
    frame_idx=0, 
    threshold_std=1.3, 
    feature_thresh=0.85
):
    global last_curb_time
    now = time.time()

    # 쿨다운 시간 이내면 감지 스킵
    if now - last_curb_time < curb_cooldown:
        # print(f"[CURB] 감지 쿨다운 중... ({now - last_curb_time:.1f}s 경과)")
        return {
            "curb_detected": False,
            "gradient_deviation": 0.0,
            "cosine_similarity": 1.0
        }

    # 수직 평균 깊이 → gradient
    vertical_profile = np.mean(depth_map, axis=1)
    current_gradient = np.gradient(vertical_profile)
    gradient_history.append(current_gradient)

    # 평균 그래디언트의 이상도
    avg_gradient = np.mean(gradient_history, axis=0)
    trimmed_grad = avg_gradient[1:-1]  # 경계 제외
    grad_deviation = np.abs(trimmed_grad - np.mean(trimmed_grad))
    max_deviation = np.max(grad_deviation)

    grad_deviation_history.append(max_deviation)
    recent_avg = np.mean(list(grad_deviation_history)[-10:]) if len(grad_deviation_history) >= 10 else np.mean(grad_deviation_history)
    is_grad_curb = max_deviation > threshold_std * recent_avg

    # CNN 피처 유사도
    current_feat = extract_feature(rgb_patch)
    feature_queue.append(current_feat)
    is_feat_curb, cosine_sim = False, 1.0
    if len(feature_queue) >= 30:
        past_feat = feature_queue[-30]
        cosine_sim = F.cosine_similarity(current_feat, past_feat, dim=0).item()
        is_feat_curb = cosine_sim < feature_thresh

    # 최종 판단
    is_curb = is_grad_curb and is_feat_curb
    if is_curb:
        status="턱감지"
        enqueue_tts(status,priority=0)

    # print(f"[CURB] {status} ΔGradDev={max_deviation:.2f} | CosSim={cosine_sim:.3f}{threshold_std * recent_avg:.2f}")

    # 감지 후 초기화 및 쿨다운 설정
    if is_curb:
        last_curb_time = now
        grad_deviation_history.clear()
        feature_queue.clear()

    return {
        "curb_detected": is_curb,
        "gradient_deviation": float(max_deviation),
        "cosine_similarity": float(cosine_sim)
    }