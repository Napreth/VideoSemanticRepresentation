<div align="center">
  <h1>Video Semantic Representation</h1>
  <p>Napreth 的视频语义表征和视频检索实验。</p>

  简体中文 / [English](./docs/README_EN.md)

</div>

## **项目简介**

**VideoSemanticRepresentation** 是一个用于视频语义特征提取的轻量级框架，通过自定义的三维卷积核对视频序列进行空间–时间分析，将视频转换为特征向量，用于视频检索、相似度分析与语义匹配等任务。

该项目以 《Bad Apple!!》 视频为实验样例，实现了：

- 视频帧级灰度化与分块处理

- 3D 卷积核的自定义与应用

- 卷积结果的均值聚合与特征向量输出

## **环境配置**

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

## 使用方法

运行以下命令从输入视频中提取语义特征向量：

- 第 1 个参数为传入视频的路径

- 第 2 个参数为特征向量的保存路径

```bash
python3 -m src data/raw/badapple_4k60.mp4 data/features/badapple_4k60.npy
```

输出文件为 NumPy 数组，可直接用于后续的相似度计算或检索任务。

## 数据准备

可使用 ffmpeg 生成视频数据，支持保留或去除音频。
分段切片便于批量处理与检索实验。

```bash
# 去除音频（可选）
ffmpeg -i badapple_4k60.mp4 -c:v copy -an data/raw/badapple_4k60.mp4

# 每 5 秒分片（尚未在项目中使用）
ffmpeg -i data/raw/badapple_4k60.mp4 -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p \
-g 150 -keyint_min 150 -sc_threshold 0 \
-force_key_frames "expr:gte(t,n_forced*5)" -vsync cfr \
-f segment -segment_time 5 -segment_time_delta 0.01 -reset_timestamps 1 \
data/slice/4k60_5s/s%03d.mp4
```

## 数据下载

由于原始与切片视频体积较大，数据已上传至网盘：

- **OneDrive：**
[https://1drv.ms/f/c/3e28e8749a82a883/Ekz3ihbtSNNNpFDcCWJQBdwBSsE6zJu_0vGuy0GboR88_Q?e=9VvZCd](https://1drv.ms/f/c/3e28e8749a82a883/Ekz3ihbtSNNNpFDcCWJQBdwBSsE6zJu_0vGuy0GboR88_Q?e=9VvZCd)

- **百度网盘：**
[https://pan.baidu.com/s/1acP6LETBLVJZofjhxWkC4g?pwd=6rx5](https://pan.baidu.com/s/1acP6LETBLVJZofjhxWkC4g?pwd=6rx5)

- **夸克网盘：**
[https://pan.quark.cn/s/f9a0ab709f08](https://pan.quark.cn/s/f9a0ab709f08)

- **Nextcloud：**
[https://dav.napreth.com/index.php/s/BZpkp9487w4tWwQ](https://dav.napreth.com/index.php/s/BZpkp9487w4tWwQ)

## 项目结构

```
VideoSemanticRepresentation/
├─ data/
│  ├─ raw/                  # 原始视频
│  ├─ slice/                # 视频切片（每5秒一段）
│  └─ features/             # 输出特征 (.npy)
├─ src/
│  ├─ __main__.py           # 主入口：python -m src
│  ├─ video.py              # 视频读取与灰度化
│  ├─ kernels.py            # 自定义3D卷积核
│  ├─ cnn.py                # 卷积与特征聚合
│  └─ __init__.py           # 包信息
├─ docs/
│  └─ README_EN.md          # 英文版说明文档
├─ requirements.txt
├─ README.md
└─ LICENSE
```

## 模块说明

|模块|功能|
|---|---|
|`src/video.py`|读取视频并按帧生成灰度图序列|
|`src/kernels.py`|定义多种时空卷积核（运动、形状、边缘等）|
|`src/cnn.py`|使用 SciPy 实现 3D 卷积与特征聚合|
|`src/__main__.py`|程序主入口与进度控制|
