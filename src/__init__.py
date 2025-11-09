"""
VideoSemanticRepresentation Package
-----------------------------------

A lightweight video semantic feature extraction framework
built upon custom 3D convolution kernels.

This package converts videos into temporal-spatial feature vectors
using GPU-accelerated convolutional operations. The extracted features
can be applied to video retrieval, similarity matching, and semantic analysis.

Modules
-------
- video: Video reading, preprocessing, and frame batching.
- feature: Core convolution-based feature extraction.
- search: Feature-space similarity and segment matching.
- __main__: Command-line interface for pairwise video comparison.

Author: Napreth
Email:  dev@napreth.com
Version: v0.3.0
"""



__author__ = "Napreth"
__email__ = "dev@napreth.com"
__version__ = "0.3.0"


__all__ = [
    "__author__",
    "__email__",
    "__version__",
]
