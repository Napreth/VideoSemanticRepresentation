"""
feature.py
-----------

Core feature extraction module for the VideoSemanticRepresentation framework.
This module performs semantic feature extraction from videos using
custom 3D convolution kernels implemented with CuPy for GPU acceleration.

Main Responsibilities
---------------------
1. Construct and manage convolution kernels for motion, shape, and edge detection.
2. Perform temporal-spatial 3D convolutions on video frame blocks.
3. Aggregate convolution outputs into temporal semantic feature vectors.
4. Handle feature caching for efficient re-computation avoidance.
"""


import time
from pathlib import Path
import json
import hashlib
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import convolve
from .video import _get_meta, frames

try:
    from uuid import uuid7 as gen_uuid
except Exception:
    try:
        from uuid6 import uuid7 as gen_uuid
    except Exception:
        from uuid import uuid4 as gen_uuid


# ==========================
#  Kernel Construction
# ==========================
def _build_kernels():
    """
    Construct a set of predefined 3D convolution kernels for video feature extraction.

    Returns
    -------
    list of cupy.ndarray
        A list of 3D kernels used for different spatial–temporal transformations:
        - Motion kernels: detect frame-to-frame movements (up/down/left/right)
        - Laplacian kernel: emphasize shape and local intensity change
        - Inversion kernel: capture global contrast
        - Sobel-like kernel: highlight emerging edges
    """
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


# ==========================
#  CNN Core
# ==========================
def _cnn(data, kernels, fps: float):
    """
    Perform 3D convolution-based feature extraction on a block of video frames.

    Parameters
    ----------
    data : cupy.ndarray
        A 3D tensor representing a sequence of grayscale frames (T * H * W).
    kernels : list of cupy.ndarray
        The convolution kernels built by `_build_kernels()`.
    fps : float
        Frame rate of the video, used for time normalization.

    Returns
    -------
    cupy.ndarray
        A 1D array of aggregated feature values per kernel,
        representing mean convolution responses per second.
    """
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
    """
    Convert a float time value (in seconds) to a formatted time string.

    Parameters
    ----------
    total : float
        Total duration in seconds.

    Returns
    -------
    str
        Formatted time string in the format "H:MM:SS.CS" (centiseconds precision).
    """
    total_cs = round(total * 100)
    hours = total_cs // 360000
    minutes = (total_cs // 6000) % 60
    seconds = (total_cs // 100) % 60
    centiseconds = total_cs % 100
    return f"{hours}:{minutes:02}:{seconds:02}.{centiseconds:02}"

# ==========================
#  Feature Extraction
# ==========================
src_dir = Path(__file__).resolve().parent
root_dir = src_dir.parent
cache_dir = root_dir / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)
kernels = _build_kernels()

def _extract(video_path: str, block: float):
    """
    Extract temporal semantic features from a video file.

    The video is processed in fixed-duration blocks, each passed through
    the 3D convolution pipeline to generate feature vectors. Progress
    and estimated remaining time are displayed dynamically in the console.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    block : float
        Duration (in seconds) of each processing block.

    Returns
    -------
    cupy.ndarray
        Feature tensor with shape (num_blocks, num_kernels).
    """
    meta = _get_meta(video_path)
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

def _load_cache(video_path: str, sha256: str, block: float):
    """
    Load a cached feature tensor from the cache index if available.

    The function looks up a JSON cache index for a matching video SHA-256
    and block size, verifies that the corresponding .npy file exists, and
    returns its contents as a CuPy array. If the entry is stale or invalid,
    it is automatically removed from the index.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    sha256 : str
        SHA-256 hash of the first bytes of the video file.
    block : float
        Duration (in seconds) of each processing block.

    Returns
    -------
    cupy.ndarray or None
        Loaded feature tensor if a valid cache is found, otherwise None.
    """

    cache_idx_path = cache_dir / "cache.json"
    if not cache_idx_path.exists():
        return None
    try:
        with cache_idx_path.open("r", encoding="utf-8") as f:
            cache_idx = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    entries = cache_idx.get(sha256)
    if not entries:
        return None
    for c in entries:
        if c.get("block") != block:
            continue
        cache_path = cache_dir / f"{c['uuid']}.npy"
        if cache_path.exists():
            print(
                "Matched cache.",
                f"Video: '{video_path}'",
                f"sha256: {sha256}",
                sep="\t",
            )
            return cp.asarray(np.load(str(cache_path)))
        # If the file does not exist, clean up invalid entries
        print(f"Stale cache entry found: {cache_path.name}, removing...")
        entries.remove(c)
        if not entries:
            cache_idx.pop(sha256, None)
        tmp_path = cache_idx_path.with_suffix(".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as f:
                json.dump(cache_idx, f, indent=2)
            tmp_path.replace(cache_idx_path)
        except:
            pass
        break
    return None

def _save_cache(video_path: str, sha256: str, block: float, feature):
    """
    Save a computed feature tensor to disk and update the cache index.

    Each entry is recorded in 'cache.json' with metadata including:
      - UUID v7 or v4 (file name stem)
      - block duration
      - creation timestamp

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    sha256 : str
        SHA-256 hash of the first bytes of the video file.
    block : float
        Duration (in seconds) of each processing block.
    feature : cupy.ndarray
        Computed feature tensor to be saved.
    """
    cache_idx_path = cache_dir / "cache.json"
    cache_dir.mkdir(parents=True, exist_ok=True)
    uid = gen_uuid().hex
    cache_path = cache_dir / f"{uid}.npy"
    try:
        np.save(str(cache_path), cp.asnumpy(feature))
    except Exception as e:
        print("[Warning] Failed to save feature file.")
        return
    cache_idx = {}
    try:
        if not cache_idx_path.exists():
            with cache_idx_path.open('w', encoding="utf-8") as f:
                f.write('{}')
        with cache_idx_path.open('r', encoding="utf-8") as f:
            cache_idx = json.load(f)
    except Exception as e:
        print("[Warning] Failed to open cache index.")
        return
    entry = {
        "uuid": uid,
        "block": block,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
    }
    entries = cache_idx.setdefault(sha256, [])
    entries.append(entry)
    tmp_path = cache_idx_path.with_suffix(".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(cache_idx, f, indent=2, ensure_ascii=False)
        tmp_path.replace(cache_idx_path)
    except Exception as e:
        print("[Warning] Failed to update cache index.")
        return

    print(
        "Saved new cache.",
        f"Video: '{video_path}'",
        f"sha256: {sha256}",
        f"UUID: {uid}",
        sep="\t"
    )


def get_feature(video_path: str, block: float, use_cache: bool=True, save_cache: bool=True):
    """
    Retrieve or compute semantic feature representation for a given video.

    This function manages caching by generating a hash key from the first 100 bytes
    of the video file. If a cached feature exists, it is loaded directly; otherwise,
    the feature is extracted and optionally saved.

    Parameters
    ----------
    video_path : str
        Path to the target video file.
    block : float
        Duration (in seconds) of each processing block.
    use_cache : bool, default=True
        Whether to load cached feature vectors if available.
    save_cache : bool, default=True
        Whether to save computed features to the cache directory.

    Returns
    -------
    cupy.ndarray
        The computed or loaded feature tensor.
    """
    if use_cache or save_cache:
        with open(video_path, 'rb') as video_file:
            sha256 = hashlib.new('sha256')
            sha256.update(video_file.read(100))
            sha256 = sha256.hexdigest()
    if use_cache:
        cache =_load_cache(video_path, sha256, block)
        if cache is not None:
            return cache

    feature = _extract(video_path, block)

    if save_cache:
        _save_cache(video_path, sha256, block, feature)
    return feature
