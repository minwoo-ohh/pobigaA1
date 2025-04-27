# main.py
import threading
from services.frame_saver import start_frame_capture
from services.gps_server import start_gps_server
from services.audio_receiver import start_audio_server
from models.parallel_inference import process_video_stream
from services.output_queue import start_tts_output_loop

if __name__ == "__main__":
    # 프레임 저장 + 병렬 YOLO/Depth/VLM 실행
    threading.Thread(target=process_video_stream).start()

    # GPS HTTP 수신 서버
    threading.Thread(target=start_gps_server).start()

    # 음성 파일 수신 및 SST 처리 → 분기
    threading.Thread(target=start_audio_server).start()

    # TTS 우선순위 큐 처리 루프
    threading.Thread(target=start_tts_output_loop).start()
    
    ###
