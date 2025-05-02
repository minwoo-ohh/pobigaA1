import time
from models.parallel_inference import get_last_frame
from models.vlm_short import run_short_vlm

def main():
    kor_command = "앞에 뭐 있어?"  # 고정 명령어

    print("[시작] 영상 처리 중... n초 기다립니다.")
    time.sleep(5)

    # Step 1: 최근 프레임 가져오기
    frame_path = get_last_frame("storage/frames")

    # Step 2: VLM 실행
    result = run_short_vlm(frame_path, kor_command)
    print(f"[VLM 결과] {result}")

if __name__ == "__main__":
    main()
