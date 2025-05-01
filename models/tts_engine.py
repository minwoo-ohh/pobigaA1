import threading
import queue
import os
from datetime import datetime
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import _play_with_simpleaudio as play_audio

# 우선순위 큐 정의 (낮은 숫자가 우선)
tts_queue = queue.PriorityQueue()

def synthesize_tts(text: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"tts_{timestamp}.mp3"
    filepath = os.path.join("/tmp", filename)
    tts = gTTS(text=text, lang='ko')
    tts.save(filepath)
    return filepath

def enqueue_tts(text: str, priority: int = 1):
    tts_queue.put((priority, text))

# def tts_worker():
#     while True:
#         try:
#             priority, text = tts_queue.get()
#             print(f"[TTS] 실행: {text} (우선순위 {priority})")
#             mp3_path = synthesize_tts(text)
#             sound = AudioSegment.from_mp3(mp3_path).speedup(playback_speed=1.5)
#             play_audio(sound).wait_done()
#             os.remove(mp3_path)
#         except Exception as e:
#             print(f"[TTS] 오류: {e}")

def tts_worker():
    while True:
        try:
            priority, text = tts_queue.get()
            print(f"[TTS] 실행: {text} (우선순위 {priority})")
            mp3_path = synthesize_tts(text)
            print(f"[TTS] 생성된 파일: {mp3_path}")

            sound = AudioSegment.from_mp3(mp3_path).speedup(playback_speed=1.1)
            print(f"[TTS] 길이: {sound.duration_seconds:.2f}초")
            print("[TTS] 재생 시작")
            play_audio(sound).wait_done()
            print("[TTS] 재생 완료")

            os.remove(mp3_path)
        except Exception as e:
            print(f"[TTS] 오류: {e}")


# # 객체 위험 감지 예시
# def process_yolo_detection(detected_labels):
#     for label in detected_labels:
#         warning_text = f"전방에 {label} 있습니다"
#         enqueue_tts(warning_text, priority=0)  # 위험 경고 → 우선

# 음성 명령 처리 예시
def voice_command_handler(text):
    if not (text.startswith("포리야") or text.startswith("포리")):
        return

    command_text = text.replace("포리야", "").replace("포리", "").strip()

    if any(keyword in command_text for keyword in ["일기", "다이어리"]):
        enqueue_tts("일기를 시작합니다", priority=1)
        run_long_vlm()
    elif "안내" in command_text:
        enqueue_tts("안내를 시작합니다", priority=1)
        run_navigation()
    elif "앞에" in command_text:
        enqueue_tts("주변 상황을 설명드릴게요", priority=1)
        run_short_vlm()

# 기능 실행 예시 함수
def run_long_vlm():
    print("[VLM] Long VLM 실행 중")

def run_navigation():
    print("[NAVIGATION] 안내 기능 실행 중")

def run_short_vlm():
    print("[VLM] Short VLM 실행 중")

# 백그라운드에서 TTS 루프 실행
threading.Thread(target=tts_worker, daemon=True).start()