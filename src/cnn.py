import cupy as cp
from cupyx.scipy.ndimage import convolve

def cnn(data, kernels, fps: float):
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
