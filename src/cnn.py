import cupy as cp
from cupyx.scipy.ndimage import convolve


def cnn(data, kernels):
    features = []
    for kernel in kernels:
        k = kernel.astype(cp.float32, copy=False)
        out = convolve(data, k, mode="constant", cval=0.0)
        pad = [(kk // 2, kk - kk // 2 - 1) for kk in k.shape]
        slices = tuple(slice(p[0], -p[1] or None) for p in pad)
        out_valid = out[slices]
        features.append(cp.mean(out_valid))
    return cp.array(features, dtype=cp.float32)
