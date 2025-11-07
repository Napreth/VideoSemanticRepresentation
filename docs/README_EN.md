<div align="center">
  <h1>Video Semantic Representation</h1>
  <p>Napreth’s Video Semantic Representation and Video Retrieval Experiment Assessment.</p>

  [简体中文](../README.md) / English

</div>

## **Project Overview**

VideoSemanticRepresentation is a lightweight framework for extracting semantic features from videos. It performs spatio-temporal analysis on frame sequences using custom 3D convolution kernels, converting a video into a feature vector suitable for video retrieval, similarity analysis, and semantic matching.

The project uses the video “Bad Apple!!” as a sample and implements:

- Frame-level grayscaling and tiling

- Custom 3D convolution kernels

- Mean aggregation over convolution results to produce a feature vector

## **Environment Setup**

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m src input.mp4 output.npy
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m src input.mp4 output.npy
```

## **Usage**

Run the following command to extract a semantic feature vector from an input video:

- Argument 1: path to the input video

- Argument 2: path to save the output feature vector

```bash
python3 -m src data/raw/badapple_4k60.mp4 data/features/badapple_4k60.npy
```

The output file is a NumPy array that can be directly used for subsequent similarity computation or retrieval tasks.

## **Data Preparation**

You can use ffmpeg to produce the video data, with or without audio. Segmenting the video can help with batch processing and retrieval experiments.

```bash
# Remove audio (optional)
ffmpeg -i badapple_4k60.mp4 -c:v copy -an data/raw/badapple_4k60.mp4

# Segment every 5 seconds (not yet used in the project)
ffmpeg -i data/raw/badapple_4k60.mp4 -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p \
-g 150 -keyint_min 150 -sc_threshold 0 \
-force_key_frames "expr:gte(t,n_forced*5)" -vsync cfr \
-f segment -segment_time 5 -segment_time_delta 0.01 -reset_timestamps 1 \
data/slice/4k60_5s/s%03d.mp4
```

## **Data Download**

Because the raw and sliced videos are large, they are hosted on cloud storage:

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
│  ├─ slice/                # Video slices (5s each)
│  └─ features/             # Output features (.npy)
├─ src/
│  ├─ __main__.py           # Entry point: python -m src
│  ├─ video.py              # Video reading and grayscale conversion
│  ├─ kernels.py            # Custom 3D convolution kernels
│  ├─ cnn.py                # Convolution and feature aggregation
│  └─ __init__.py           # Package metadata
├─ docs/
│  └─ README_EN.md          # English documentation
├─ requirements.txt
├─ README.md
└─ LICENSE
```

## **Modules**

| Module | Description |
|---|---|
| `src/video.py` | Read video and generate grayscale frame sequence |
| `src/kernels.py` | Define multiple spatiotemporal kernels (motion, shape, edges, etc.) |
| `src/cnn.py` | 3D convolution with SciPy and feature aggregation |
| `src/__main__.py` | Program entry point and progress control |
