<div align="center">
  <h1>Video Semantic Representation</h1>
  <p>Napreth's video semantic representation and video retrieval experiments.</p>

  [简体中文](../README.md) / English

</div>

## **Overview**

**VideoSemanticRepresentation** is a lightweight framework for video semantic feature extraction and clip retrieval. It performs spatio‑temporal analysis on video sequences with custom 3D convolution kernels, converts videos into feature vectors, and uses Euclidean distance for clip retrieval.

The sample experiment is based on "Bad Apple!!". The current implementation includes:

- Frame‑level grayscaling and chunking by fixed duration
- Custom 3D convolution kernels (motion/shape/inversion/edge, etc.)
- Mean aggregation over the valid region of convolution results to obtain a per‑block feature vector

### **Version History**

**v0.3.1 (2025-11-09)**  
Fixes cache mis-hits caused by video hashing and improves cache index structure.

**v0.3.0 (2025-11-09)**  
Implements video retrieval and refactors the feature extraction architecture.

**v0.2.0 (2025-11-08)**  
Introduces GPU acceleration and enhances feature extraction accuracy.

**v0.1.0 (2025-11-07)**  
Implements the core prototype of the video semantic representation framework.

## **Quick Start**

### **Environment Requirements**

- Python 3.10+ (required by NumPy 2.x / SciPy 1.16)
- NVIDIA GPU and compatible CUDA driver (CuPy for GPU computation)
- See `requirements.txt` for dependencies

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src <reference_video_path> <query_video_path>
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m src <reference_video_path> <query_video_path>
```

Example:

```bash
python3 -m src data/raw/badapple_4k60.mp4 data/slice/4k60_5s/s012.mp4
```

After running, the program prints the match result (best‑match start/end times and the Euclidean distance score).

Features are automatically cached under `cache/`. The file name contains the SHA‑256 of the first 100 bytes of the video and the block duration, e.g., `cache/<sha256>_b0.5.npy`. The cache will be reused when processing the same video again.

---

## **Features & Kernels**

There are 7 feature dimensions by default (from 7 kernels):

1. Motion (left shift)
2. Motion (right shift)
3. Motion (up shift)
4. Motion (down shift)
5. Shape (Laplacian, stacked over three frames)
6. Global inversion (2‑frame difference)
7. Edges (Sobel, stacked over three frames)

Convolution uses `cupyx.scipy.ndimage.convolve` with mode `constant` (zero padding). We then take the mean over the valid region of each kernel as that dimension's feature. Each block yields a 7‑D vector, so a single video's feature tensor has shape approximately `(num_blocks, 7)`, where `num_blocks ≈ video_duration / block`.

---

## **Data Preparation**

You can use ffmpeg to generate video data, with or without audio. Segmenting the video makes batch processing and retrieval experiments easier.

```bash
# Remove audio (optional)
ffmpeg -i badapple_4k60.mp4 -c:v copy -an data/raw/badapple_4k60.mp4

# Slice every 5 seconds (not yet used in the project)
ffmpeg -i data/raw/badapple_4k60.mp4 -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p \
  -g 150 -keyint_min 150 -sc_threshold 0 \
  -force_key_frames "expr:gte(t,n_forced*5)" -vsync cfr \
  -f segment -segment_time 5 -segment_time_delta 0.01 -reset_timestamps 1 \
  data/slice/4k60_5s/s%03d.mp4
```

## **Data Download**

Because the raw and sliced videos are large, the data is hosted on cloud storage:

- OneDrive:
  https://1drv.ms/f/c/3e28e8749a82a883/Ekz3ihbtSNNNpFDcCWJQBdwBSsE6zJu_0vGuy0GboR88_Q?e=9VvZCd

- Baidu Netdisk:
  https://pan.baidu.com/s/1acP6LETBLVJZofjhxWkC4g?pwd=6rx5

- Quark:
  https://pan.quark.cn/s/f9a0ab709f08

- Nextcloud:
  https://dav.napreth.com/index.php/s/BZpkp9487w4tWwQ

## **Project Structure**

```
VideoSemanticRepresentation/
├─ data/
│  ├─ raw/                  # Raw videos
│  ├─ slice/                # Video slices (every 5 seconds)
│  └─ features/             # Output features (.npy)
├─ src/
│  ├─ __main__.py           # Entry point: python -m src (reference video vs query video)
│  ├─ video.py              # Video reading & grayscale (OpenCV -> CuPy)
│  ├─ feature.py            # Kernel construction, 3D convolution & feature aggregation, caching
│  ├─ search.py             # Segment retrieval in feature space (sliding window + Euclidean distance)
│  └─ __init__.py           # Package info
├─ docs/
│  └─ README_EN.md          # English documentation
├─ requirements.txt
├─ README.md
└─ LICENSE
```

## **Modules**

| Module | Function |
| --- | --- |
| `src/video.py` | Read video and generate grayscale frame sequence |
| `src/feature.py` | Define/manage kernels, perform 3D convolution and aggregate into temporal features, with caching |
| `src/search.py` | Slide a window over reference features and use Euclidean distance to locate the most similar segment to the query |
| `src/__main__.py` | Program entry point: compute features for two videos and perform retrieval |
