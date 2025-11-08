import cupy as cp

def euclidean_distance(a, b):
    """计算两个矩阵 (n,d) 的逐行欧氏距离"""
    return cp.sqrt(cp.sum((a - b) ** 2, axis=1))

def locate_segmen(F, Q):
    N, D = F.shape
    M = Q.shape[0]
    scores = cp.zeros(N - M + 1, dtype=cp.float32)

    for i in range(N - M + 1):
        segment = F[i:i+M]
        d = euclidean_distance(segment, Q).mean()
        scores[i] = d

    best_idx = int(cp.argmin(scores))
    return best_idx, float(scores[best_idx])

def search(block: float, F: cp.ndarray, Q: cp.ndarray):
    print(f"\nSearching segment in feature space...")
    start_idx, score = locate_segmen(F, Q)
    start_time = start_idx * block
    end_time = (start_idx + Q.shape[0]) * block

    print(f"Best match time range: {start_time:.2f}s~{end_time:.2f}s")
    print(f"Euclidean distance score: {score:.6f}")

    return {
        "index": start_idx,
        "start_time": start_time,
        "end_time": end_time,
        "score": score
    }
