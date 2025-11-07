import numpy as np

def build_motion_kernels():
    kernels = []
    # Left shift
    k_left = np.zeros((3, 3, 3), np.float32)
    k_left[0, 1, 2] = -1
    k_left[2, 1, 0] = 1
    kernels.append(k_left)
    # Right shift
    k_right = np.zeros((3, 3, 3), np.float32)
    k_right[0, 1, 0] = -1
    k_right[2, 1, 2] = 1
    kernels.append(k_right)
    # Upward shift
    k_up = np.zeros((3, 3, 3), np.float32)
    k_up[0, 2, 1] = -1
    k_up[2, 0, 1] = 1
    kernels.append(k_up)
    # Downward shift
    k_down = np.zeros((3, 3, 3), np.float32)
    k_down[0, 0, 1] = -1
    k_down[2, 2, 1] = 1
    kernels.append(k_down)
    return kernels

def build_shape_kernels():
    kernels = []
    laplacian = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], np.float32)
    k_shape = np.repeat(laplacian[None, :, :], 3, axis=0)
    kernels.append(k_shape / 9)
    return kernels

def build_invert_kernel():
    return [np.array([
        [[1]],
        [[-1]],
    ], np.float32)]

def build_emerge_kernels():
    kernels = []
    sobel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], np.float32)
    k_emerge = np.repeat(sobel[None, :, :], 3, axis=0)
    kernels.append(k_emerge / 9)
    return kernels

def build_kernels():
    kernels = []
    kernels += build_motion_kernels()   # Motion: up, down, left, right
    kernels += build_shape_kernels()    # Shape transformation
    kernels += build_invert_kernel()    # Global inversion
    kernels += build_emerge_kernels()   # Edge emergence
    return kernels
