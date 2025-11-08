import cv2
import numpy as np
import cupy as cp
from contextlib import contextmanager

@contextmanager
def capture(video_path):
    cap = cv2.VideoCapture(video_path)
    try:
        yield cap
    finally:
        cap.release()

def get_meta(video_path: str) -> dict:
    with capture(video_path) as cap:
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps != 0 else 0.0
        return {
            'width': width,
            'height': height,
            'frame_count': frame_count,
            'fps': fps,
            'duration': duration
        }

def frames(video_path: str, batch_size: int=1, offset: int=0):
    with capture(video_path) as cap:
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        if not (isinstance(offset, int) and 0 <= offset < int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
            raise ValueError("Offset must be a non-negative integer smaller than total frame count.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, offset)

        while True:
            frames_cpu = []
            for i in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert to single-channel grayscale
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                frames_cpu.append(gray_frame)
            if frames_cpu:
                frames_gpu = cp.asarray(np.stack(frames_cpu), dtype=cp.float32)
                del frames_cpu
                yield frames_gpu
            else:
                break
