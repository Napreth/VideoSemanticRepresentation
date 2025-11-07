import sys, os
import time
from datetime import timedelta
import numpy as np
from .video import get_meta, frames
from .cnn import cnn
from .kernels import build_kernels

block = 30    # Number of frames per convolution block

def main(argv):
    if len(argv) < 1:
        return 1
    meta = get_meta(argv[0])
    print(f"Video: '{argv[0]}'",
          f"Resolution: {meta['width']}x{meta['height']}",
          f"FPS: {int(meta['fps'])}",
          f"Duration: {timedelta(seconds=int(meta['duration']))}",
          sep='\t\t')
    kernels = build_kernels()

    start = time.monotonic()
    processed_frame = 0
    features = []
    for frame_block in frames(argv[0], block):
        feature = cnn(frame_block, kernels)
        elapsed = time.monotonic() - start
        processed_frame = min(processed_frame + 30, meta['frame_count'])
        progress = processed_frame / meta['frame_count'] * 100
        print("\n"
              f"[{progress:.1f}%] "
              f"Elapsed: {timedelta(seconds=int(elapsed))}, "
              f"ETA: {timedelta(seconds=int(elapsed / processed_frame * meta['frame_count'] - elapsed))}, "
              f"Frames {processed_frame - block + 1}-{processed_frame}")
        print(feature)
        features.append(feature)
        
    dirname = os.path.dirname(argv[1])
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    np.save(argv[1], np.stack(features))
    total = time.monotonic() - start
    print(f"\nDone. Saved {len(features)} feature blocks to '{argv[1]}' "
          f"in {timedelta(seconds=int(total))}.")
    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))