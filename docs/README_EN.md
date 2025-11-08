<div align="center">
  <h1>Video Semantic Representation</h1>
  <p>Napreth's video semantic representation and video retrieval experiment.</p>

  [简体中文](../README.md) / English

</div>

## **Overview**

**VideoSemanticRepresentation** is a lightweight framework for extracting semantic feature vectors from videos. It performs spatio‑temporal analysis on frame sequences using custom 3D convolution kernels and converts each video (or segmented block of frames) into a feature vector that can be used for video retrieval, similarity computation, and semantic matching.

The current sample experiment is based on the video "Bad Apple!!" and implements:

* Frame‑level grayscaling and processing in blocks of 30 frames (can be changed via `block` in `src/__main__.py`)
* Custom 3D convolution kernels (motion / shape / inversion / edge, etc.)
* Mean aggregation over the valid convolution region to obtain a per‑block feature vector

### **Version Update**

Version **v0.2.0** migrates computation to the GPU using CuPy, significantly improving performance.

---

## **Quick Start**

### **Environment Requirements**

* Python 3.10+ (required by NumPy 2.x / SciPy 1.16)
* NVIDIA GPU and compatible CUDA driver (CuPy for GPU acceleration)
* Dependencies listed in `requirements.txt`

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src <input_video_path> <output_npy_path>
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m src <input_video_path> <output_npy_path>
```

Example:

```bash
python3 -m src data/raw/badapple_4k60.mp4 data/features/badapple_4k60.npy
```

Output: saved as a NumPy array (`.npy`) with shape `(num_blocks, 7)`, where `num_blocks ≈ total_frames / 30`.

---

## **Features & Kernels**

By default there are 7 feature dimensions (from 7 kernels):

1. Motion (left shift)
2. Motion (right shift)
3. Motion (up shift)
4. Motion (down shift)
5. Shape (Laplacian, 3‑frame stack)
6. Global inversion (2‑frame difference)
7. Edge (Sobel, 3‑frame stack)

Convolution uses `cupyx.scipy.ndimage.convolve` with mode `constant` (zero padding). For each kernel we take the mean over the valid region as that dimension's feature value.

---

## **Data Preparation**

You can use ffmpeg to generate video data, optionally removing audio. Segmenting the video facilitates batch processing and retrieval experiments.

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

Because the raw and sliced videos are large, they are hosted on cloud storage:

* OneDrive:  
  https://1drv.ms/f/c/3e28e8749a82a883/Ekz3ihbtSNNNpFDcCWJQBdwBSsE6zJu_0vGuy0GboR88_Q?e=9VvZCd
* Baidu Netdisk:  
  https://pan.baidu.com/s/1acP6LETBLVJZofjhxWkC4g?pwd=6rx5
* Quark:  
  https://pan.quark.cn/s/f9a0ab709f08
* Nextcloud:  
  https://dav.napreth.com/index.php/s/BZpkp9487w4tWwQ

## **Project Structure**

```
VideoSemanticRepresentation/
├─ data/
│  ├─ raw/                  # Raw videos
│  ├─ slice/                # Video slices (every 5 seconds)
│  └─ features/             # Output features (.npy)
├─ src/
│  ├─ __main__.py           # Entry point: python -m src
│  ├─ video.py              # Video reading & grayscale (OpenCV -> CuPy)
│  ├─ kernels.py            # Custom 3D convolution kernels
│  ├─ cnn.py                # 3D convolution & feature aggregation
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
| `src/video.py` | Read video and produce grayscale frame sequence |
| `src/kernels.py` | Define spatio‑temporal kernels (motion, shape, edges, etc.) |
| `src/cnn.py` | 3D convolution using CuPy and feature aggregation |
| `src/__main__.py` | Program entry point & progress control |
