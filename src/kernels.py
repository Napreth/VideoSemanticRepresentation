import cupy as cp

def build_motion_kernels():
    kernels = []
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
    return kernels

def build_shape_kernels():
    kernels = []
    laplacian = cp.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], cp.float32)
    k_shape = cp.repeat(laplacian[None, :, :], 3, axis=0)
    kernels.append(k_shape / 9)
    return kernels

def build_invert_kernel():
    return [cp.array([
        [[1]],
        [[-1]],
    ], cp.float32)]

def build_emerge_kernels():
    kernels = []
    sobel = cp.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], cp.float32)
    k_emerge = cp.repeat(sobel[None, :, :], 3, axis=0)
    kernels.append(k_emerge / 9)
    return kernels

def build_kernels():
    kernels = []
    kernels += build_motion_kernels()   # Motion: up, down, left, right
    kernels += build_shape_kernels()    # Shape transformation
    kernels += build_invert_kernel()    # Global inversion
    kernels += build_emerge_kernels()   # Edge emergence
    return kernels
