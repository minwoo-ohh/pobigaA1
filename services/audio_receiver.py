# audio_receiver.py

# from flask import Flask, request, jsonify
# import os
# from datetime import datetime
# from models.sst_engine import run_sst
# from models.nav_engine import run_navigation
# from models.vlm_short import run_short_vlm
# from models.vlm_long import run_vlm
# from services.output_queue import add_to_queue
# from models.parallel_inference import get_last_frame  # 최근 프레임

# app = Flask(__name__)
# UPLOAD_DIR = "./storage/audio"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# @app.route("/audio", methods=["POST"])
# def receive_audio():
#     file = request.files.get("audio")
#     if not file:
#         return jsonify({"error": "No file uploaded"}), 400

#     filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + file.filename
#     filepath = os.path.join(UPLOAD_DIR, filename)
#     file.save(filepath)

#     # 1. SST 처리
#     command_text = run_sst(filepath)
#     print(f"[SST] 인식된 텍스트: {command_text}")

#     # 2. 최근 프레임 가져오기
#     frame = get_last_frame()

#     # 3. 명령어 분기
#     if "길" in command_text or "어디" in command_text:
#         response = run_navigation(command_text)
#     elif "뭐" in command_text or "보여" in command_text:
#         response = run_short_vlm(frame, command_text)
#     else:
#         response = run_vlm(frame)  # 장면 전체 요약

#     print(f"[RESPONSE] {response}")
#     add_to_queue(response, priority=2)

#     return jsonify({"status": "ok", "text": command_text})

# def start_audio_server():
#     app.run(host="0.0.0.0", port=8081)
