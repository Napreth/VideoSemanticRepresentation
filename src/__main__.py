"""
__main__.py
-----------

Entry point for the VideoSemanticRepresentation framework.

This script compares two video feature representations and locates
the most similar segment between them. It is typically invoked as:

    python -m src <reference_video> <query_video>
"""

import sys
from pathlib import Path
from .feature import get_feature
from .search import search

src_dir = Path(__file__).resolve().parent
root_dir = src_dir.parent
block = 0.5    # Duration(second) per convolution block

def main(argv):
    F = get_feature(argv[0], block)
    Q = get_feature(argv[1], block)
    search(block, F, Q)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
