"""
VideoSemanticRepresentation
---------------------------

Lightweight framework for video semantic feature extraction and retrieval.

It converts video sequences into temporal–spatial feature vectors
through custom 3D convolution kernels accelerated by GPU.
The extracted features enable efficient video similarity search
and segment-level semantic analysis.

Modules
-------
- video   : Video loading, preprocessing, and frame batching.
- feature : 3D convolution–based feature extraction and caching.
- search  : Feature-space similarity computation and segment matching.
- __main__: Command-line interface for feature extraction and retrieval.

Author:  Napreth
Email:   dev@napreth.com
Version: v1.0.0
"""


__author__ = "Napreth"
__email__ = "dev@napreth.com"
__version__ = "v1.0.0"


__all__ = [
    "__author__",
    "__email__",
    "__version__",
]
