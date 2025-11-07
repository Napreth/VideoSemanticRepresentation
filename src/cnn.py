import numpy as np
from scipy.signal import convolve

def cnn(data: np.ndarray, kernels: np.ndarray):
    features = []
    for kernel in kernels:
        out = convolve(data, kernel, mode='valid')
        features.append(np.mean(out))
    return np.array(features)

