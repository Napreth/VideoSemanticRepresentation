"""
video.py
--------

Video input and preprocessing utilities for the VideoSemanticRepresentation framework.

This module handles low-level video operations including metadata extraction
and frame sequence generation. It provides GPU-accelerated frame streaming
via CuPy to efficiently feed data into the convolution-based feature extractor.

Functions
---------
- _capture(video_path): Context manager that safely opens and releases a video file.
- _get_meta(video_path): Extracts key video metadata such as resolution, FPS, and duration.
- frames(video_path, duration, offset): Generates frame batches as CuPy arrays.
"""


import cv2
import numpy as np
import cupy as cp
from contextlib import contextmanager

@contextmanager
def _capture(video_path: str):
    """
    Context manager for safely capturing a video file using OpenCV.

    Opens the specified video file, yields a `cv2.VideoCapture` object,
    and ensures proper resource release after use, even in the case of exceptions.

    Parameters
    ----------
    video_path : str
        Path to the input video file.

    Yields
    ------
    cv2.VideoCapture
        An active OpenCV video capture object.

    Raises
    ------
    ValueError
        If the specified video file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    try:
        yield cap
    finally:
        cap.release()

def _get_meta(video_path: str) -> dict:
    """
    Extract basic metadata from a video file.

    Reads video properties such as width, height, frame count,
    frames per second (FPS), and total duration.

    Parameters
    ----------
    video_path : str
        Path to the input video file.

    Returns
    -------
    dict
        A dictionary containing:
        - 'width': int, frame width in pixels
        - 'height': int, frame height in pixels
        - 'frame_count': int, total number of frames
        - 'fps': float, frames per second
        - 'duration': float, total video length in seconds

    Raises
    ------
    ValueError
        If the video file cannot be opened.
    """
    with _capture(video_path) as cap:
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

def frames(video_path: str, duration: float, offset: float=0.0):
    """
    Generate sequential batches of grayscale video frames as CuPy arrays.

    This function streams frames from the video in fixed-duration chunks,
    converts them to single-channel grayscale, and transfers them to GPU memory.
    It yields one CuPy array per iteration representing a temporal batch of frames.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    duration : float
        Duration (in seconds) of each frame batch to yield.
    offset : float, default=0.0
        Starting time offset (in seconds) from which to begin reading.

    Yields
    ------
    cupy.ndarray
        A 3D array of shape (T, H, W), where:
        - T: number of frames in the batch (`duration × fps`)
        - H, W: spatial dimensions of each frame (grayscale float32)

    Raises
    ------
    ValueError
        If `offset` is not within the valid duration range of the video.
    """
    meta = _get_meta(video_path)
    batch_size = int(duration * meta['fps'])
    with _capture(video_path) as cap:
        if not (isinstance(offset, float) and 0 <= offset < meta['duration'] + 1e-3):
            raise ValueError("Offset must be a non-negative float smaller than total frame count.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, offset * meta['fps'])

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
