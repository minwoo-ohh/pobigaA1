# import torch
# import cv2
# import numpy as np
# from torchvision.transforms import Compose
# from midas.dpt_depth import DPTDepthModel
# from midas.transforms import Resize, NormalizeImage, PrepareForNet

# # GPU 설정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 모델 초기화
# model = DPTDepthModel(backbone="vitb_rn50_384", non_negative=True)
# model.load_state_dict(torch.load("weights/dpt_hybrid.pt", map_location=device))
# model.to(device)
# model.eval()

# # 입력 전처리
# transform = Compose([
#     Resize(384, 384, resize_target=None, keep_aspect_ratio=True),
#     NormalizeImage(mean=[0.485, 0.456, 0.406],
#                    std=[0.229, 0.224, 0.225]),
#     PrepareForNet()
# ])

# def run_depth(frame):
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img_input = transform({"image": img})["image"]
#     img_input = torch.from_numpy(img_input).unsqueeze(0).to(device)

#     with torch.no_grad():
#         prediction = model.forward(img_input)
#         prediction = torch.nn.functional.interpolate(
#             prediction.unsqueeze(1),
#             size=img.shape[:2],
#             mode="bicubic",
#             align_corners=False,
#         ).squeeze()

#     depth_np = prediction.cpu().numpy()
#     return {"depth_map": depth_np}
