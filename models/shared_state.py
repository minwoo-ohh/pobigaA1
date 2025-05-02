# models/shared_state.py

import threading

latest_frame_lock = threading.Lock()
latest_frame = None
stop_event = threading.Event()
