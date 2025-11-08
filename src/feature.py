import time
import sys
from pathlib import Path
import hashlib
import numpy as np
import cupy as cp
from .video import get_meta, frames
from .cnn import cnn
from .kernels import build_kernels

src_dir = Path(__file__).resolve().parent
root_dir = src_dir.parent
features_dir = root_dir / "data" /"features"
features_dir.mkdir(parents=True, exist_ok=True)
kernels = build_kernels()

def _format_time(total: float) -> str:
    total_cs = round(total * 100)
    hours = total_cs // 360000
    minutes = (total_cs // 6000) % 60
    seconds = (total_cs // 100) % 60
    centiseconds = total_cs % 100
    return f"{hours}:{minutes:02}:{seconds:02}.{centiseconds:02}"

def _extract(video_path: str, block: float):
    meta = get_meta(video_path)
    print(f"Video: '{video_path}'",
          f"Resolution: {meta['width']}x{meta['height']}",
          f"FPS: {int(meta['fps'])}",
          f"Total duration: {_format_time(meta['duration'])}",
          sep='\t\t')
    start = time.monotonic()
    processed_duration = 0
    feature_list = []
    roll_back = 0
    for frame_block in frames(video_path, block):
        feature_block = cnn(frame_block, kernels, int(meta['fps']))
        cp.cuda.Stream.null.synchronize()
        elapsed = time.monotonic() - start
        processed_duration = min(processed_duration + block, meta['duration'])
        progress = processed_duration / meta['duration'] * 100
        msg = ("\n"
               f"[{progress:.1f}%] "
               f"Elapsed: {_format_time(elapsed)}, "
               f"ETA: {_format_time(elapsed / processed_duration * meta['duration'] - elapsed)}, "
               f"Duration {_format_time(processed_duration - block)}-{_format_time(processed_duration)}s"
               "\n"
               f"{feature_block}")
        print("\033[A\033[K" * roll_back, end='')
        print(msg)
        roll_back = msg.count('\n') + 1
        feature_list.append(feature_block)
    feature = cp.stack(feature_list)
    total = time.monotonic() - start
    print("\033[A\033[K" * roll_back, end='')
    print(f"Done. Calculation has completed in {_format_time(total)}.\n")
    return feature

def get_feature(video_path: str, block: float, use_cache: bool=True, save_cache: bool=True):
    if use_cache or save_cache:
        with open(video_path, 'rb') as video_file:
            sha256 = hashlib.new('sha256')
            sha256.update(video_file.read(100))
            sha256 = sha256.hexdigest()
            feature_file = str(features_dir / f"{sha256}_b{block}.npy")
    if use_cache:
        if Path(feature_file).exists():
            return cp.asarray(np.load(feature_file))

    feature = _extract(video_path, block)

    if save_cache:
        np.save(feature_file, cp.asnumpy(feature))

    return feature
