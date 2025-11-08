import time
import sys
from pathlib import Path
import hashlib
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import convolve
from .video import get_meta, frames


def _build_kernels():
    kernels = []
    # Motion: up, down, left, right
    # Left shift
    k_left = cp.zeros((3, 3, 3), cp.float32)
    k_left[0, 1, 2] = -1
    k_left[2, 1, 0] = 1
    kernels.append(k_left)
    # Right shift
    k_right = cp.zeros((3, 3, 3), cp.float32)
    k_right[0, 1, 0] = -1
    k_right[2, 1, 2] = 1
    kernels.append(k_right)
    # Upward shift
    k_up = cp.zeros((3, 3, 3), cp.float32)
    k_up[0, 2, 1] = -1
    k_up[2, 0, 1] = 1
    kernels.append(k_up)
    # Downward shift
    k_down = cp.zeros((3, 3, 3), cp.float32)
    k_down[0, 0, 1] = -1
    k_down[2, 2, 1] = 1
    kernels.append(k_down)

    # Shape transformation
    laplacian = cp.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], cp.float32)
    k_shape = cp.repeat(laplacian[None, :, :], 3, axis=0)
    kernels.append(k_shape / 9)

    # Global inversion
    k_invert = cp.array([
        [[1]],
        [[-1]],
    ], cp.float32)
    kernels.append(k_invert)

    # Edge emergence
    sobel = cp.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], cp.float32)
    k_emerge = cp.repeat(sobel[None, :, :], 3, axis=0)
    kernels.append(k_emerge / 9)

    return kernels


def _cnn(data, kernels, fps: float):
    features = []
    T = data.shape[0]

    for kernel in kernels:
        k = kernel.astype(cp.float32, copy=False)
        out = convolve(data, k, mode="constant", cval=0.0)
        pad = [(kk // 2, kk - kk // 2 - 1) for kk in k.shape]
        slices = tuple(slice(p[0], -p[1] or None) for p in pad)
        out_valid = out[slices]
        kT = k.shape[0]
        N_valid = max(T - kT + 1, 1)
        duration_seconds = N_valid / fps
        mean_per_second = out_valid.astype(cp.float64).sum() / duration_seconds
        features.append(mean_per_second.astype(cp.float32))
        del out, k
        cp._default_memory_pool.free_all_blocks()

    return cp.asarray(features, dtype=cp.float32)

def _format_time(total: float) -> str:
    total_cs = round(total * 100)
    hours = total_cs // 360000
    minutes = (total_cs // 6000) % 60
    seconds = (total_cs // 100) % 60
    centiseconds = total_cs % 100
    return f"{hours}:{minutes:02}:{seconds:02}.{centiseconds:02}"


src_dir = Path(__file__).resolve().parent
root_dir = src_dir.parent
features_dir = root_dir / "cache"
features_dir.mkdir(parents=True, exist_ok=True)
kernels = _build_kernels()

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
        feature_block = _cnn(frame_block, kernels, int(meta['fps']))
        cp.cuda.Stream.null.synchronize()
        elapsed = time.monotonic() - start
        processed_duration = min(processed_duration + block, meta['duration'])
        progress = processed_duration / meta['duration'] * 100
        msg = ("\n"
               f"[{progress:.1f}%] "
               f"Elapsed: {_format_time(elapsed)}, "
               f"ETA: {_format_time(elapsed / processed_duration * meta['duration'] - elapsed)}, "
               f"Duration {_format_time(processed_duration - block)}-{_format_time(processed_duration)}"
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
            print("Matched cache.",
                  f"Video: '{video_path}'",
                  f"sha256: {sha256}",
                  sep="\t")
            return cp.asarray(np.load(feature_file))

    feature = _extract(video_path, block)

    if save_cache:
        np.save(feature_file, cp.asnumpy(feature))
    return feature
