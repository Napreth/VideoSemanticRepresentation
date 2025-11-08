import sys
from pathlib import Path
from .feature import get_feature
from .video import get_meta
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
