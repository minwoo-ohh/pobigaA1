# import queue
# import time
# import uuid
# from models.tts_engine import synthesize_tts
# from services.audio_sender import send_to_pi

# # 전역 우선순위 큐 (작을수록 우선순위 높음)
# tts_queue = queue.PriorityQueue()

# def add_to_queue(text: str, priority: int = 3):
#     """
#     TTS용 텍스트를 우선순위 큐에 추가

#     Args:
#         text (str): 안내 문장
#         priority (int): 낮을수록 먼저 처리됨 (예: 1 = 위험 경고)
#     """
#     uid = str(uuid.uuid4())  # tie-breaker 용
#     tts_queue.put((priority, uid, text))
#     print(f"[QUEUE] 추가됨 (priority={priority}): {text}")

# def start_tts_output_loop():
#     """
#     큐에서 메시지를 꺼내 TTS → 라즈베리파이에 전송
#     """
#     print("[TTS] 우선순위 큐 처리 루프 시작")
#     while True:
#         if tts_queue.empty():
#             time.sleep(0.5)
#             continue

#         priority, uid, text = tts_queue.get()
#         print(f"[TTS] 처리 시작: {text} (priority={priority})")

#         try:
#             audio_path = synthesize_tts(text)          # 텍스트 → 음성 파일
#             send_to_pi(audio_path)                      # mp3 파일 라즈베리파이에 전송
#         except Exception as e:
#             print(f"[TTS] 처리 중 오류 발생: {e}")
