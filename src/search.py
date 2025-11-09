"""
search.py
---------

Feature-space video segment search module for the VideoSemanticRepresentation framework.

This module provides a simple yet effective segment-level matching mechanism
based on Euclidean distance between temporal feature vectors extracted from videos.
It enables locating the most similar segment of a reference video to a given query clip.

Functions
---------
- euclidean_distance(a, b): Compute row-wise Euclidean distances between two feature matrices.
- locate_segmen(F, Q): Find the segment index in F that best matches Q.
- search(block, F, Q): Perform a high-level search and print match statistics.
"""


import cupy as cp

def _euclidean_distance(a, b):
    """
    Compute the row-wise Euclidean distance between two matrices.

    Parameters
    ----------
    a : cupy.ndarray
        Feature matrix with shape (N, D).
    b : cupy.ndarray
        Feature matrix with shape (N, D), or (M, D) if used in segment comparison.

    Returns
    -------
    cupy.ndarray
        1D array of length N containing Euclidean distances for each row.
    """
    return cp.sqrt(cp.sum((a - b) ** 2, axis=1))

def _locate_segmen(F, Q):
    """
    Locate the segment in a reference feature matrix F that best matches query Q.

    For each possible temporal window of length len(Q) in F,
    the mean Euclidean distance to Q is computed. The segment
    with the smallest mean distance is returned.

    Parameters
    ----------
    F : cupy.ndarray
        Reference feature matrix with shape (N, D), representing a full video.
    Q : cupy.ndarray
        Query feature matrix with shape (M, D), representing a short video clip.

    Returns
    -------
    tuple of (int, float)
        - best_idx: The starting index of the best matching segment in F.
        - best_score: The minimum mean Euclidean distance value.
    """
    N, D = F.shape
    M = Q.shape[0]
    scores = cp.zeros(N - M + 1, dtype=cp.float32)

    for i in range(N - M + 1):
        segment = F[i:i+M]
        d = _euclidean_distance(segment, Q).mean()
        scores[i] = d

    best_idx = int(cp.argmin(scores))
    return best_idx, float(scores[best_idx])

def search(block: float, F: cp.ndarray, Q: cp.ndarray):
    print(f"\nSearching segment in feature space...")
    start_idx, score = _locate_segmen(F, Q)
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
