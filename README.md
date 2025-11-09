<div align="center">
  <h1>Video Semantic Representation</h1>
  <p>Napreth 的视频语义表征和视频检索实验。</p>

  简体中文 / [English](./docs/README_EN.md)

</div>

## **概览**

**VideoSemanticRepresentation** 是一个用于视频语义特征提取与片段检索的轻量级框架，通过自定义三维卷积核对视频序列进行时空分析，将视频转换为特征向量，并以欧氏距离为依据进行片段检索。

实验样例基于《Bad Apple!!》，当前实现包含：

- 帧级灰度化与按固定时长分块
- 自定义 3D 卷积核（运动/形状/反相/边缘等）
- 对卷积结果做有效区域均值聚合，得到每块的特征向量

### **版本记录**

- **v0.3.1（2025-11-09）**  
修复视频哈希算法导致的缓存误命中问题，并改进缓存索引结构。

- **v0.3.0（2025-11-09）**  
实现视频检索功能并重构特征提取架构。

- **v0.2.0（2025-11-08）**  
引入 GPU 加速并提升特征提取精度。

- **v0.1.0（2025-11-07）**  
完成核心视频语义表征框架原型。

## **快速开始**

### **环境要求**

- Python 3.10+（因 NumPy 2.x / SciPy 1.16 要求）
- NVIDIA GPU 与兼容的 CUDA 驱动（使用 CuPy 进行 GPU 计算）
- 依赖见 `requirements.txt`

### Windows（PowerShell）

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src <参考视频路径> <查询视频路径>
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m src <参考视频路径> <查询视频路径>
```

示例：

```bash
python3 -m src data/raw/badapple_4k60.mp4 data/slice/4k60_5s/s012.mp4
```

运行后程序会打印匹配结果（最佳匹配起止时间与欧氏距离分数）。

特征会自动缓存到 `cache/` 目录，文件名包含视频首 100 字节的 `sha256` 与分块时长，例如：`cache/<sha256>_b0.5.npy`。若再次处理同一视频将直接复用缓存。

---

## **特征与卷积核**

默认一共 7 维特征（来自 7 个卷积核）：

1. 运动（左移）
2. 运动（右移）
3. 运动（上移）
4. 运动（下移）
5. 形状（Laplacian，三帧叠加）
6. 全局反相（2 帧差分）
7. 边缘（Sobel，三帧叠加）

卷积采用 `cupyx.scipy.ndimage.convolve`，模式为 `constant`（零填充），然后对每个核的有效区域取均值作为该维特征。每个分块会得到一个 7 维向量，因此单个视频的特征张量形状约为 `(块数, 7)`，其中 `块数 ≈ 视频时长 / block`。

---

## **数据准备**

可使用 ffmpeg 生成视频数据，支持保留或去除音频。
分段切片便于批量处理与检索实验。

```bash
# 去除音频（可选）
ffmpeg -i badapple_4k60.mp4 -c:v copy -an data/raw/badapple_4k60.mp4

# 每 5 秒切片（尚未在项目中使用）
ffmpeg -i data/raw/badapple_4k60.mp4 -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p \
  -g 150 -keyint_min 150 -sc_threshold 0 \
  -force_key_frames "expr:gte(t,n_forced*5)" -vsync cfr \
  -f segment -segment_time 5 -segment_time_delta 0.01 -reset_timestamps 1 \
  data/slice/4k60_5s/s%03d.mp4
```

## **数据下载**

由于原始与切片视频体积较大，数据已上传至网盘：

- **OneDrive：**
[https://1drv.ms/f/c/3e28e8749a82a883/Ekz3ihbtSNNNpFDcCWJQBdwBSsE6zJu_0vGuy0GboR88_Q?e=9VvZCd](https://1drv.ms/f/c/3e28e8749a82a883/Ekz3ihbtSNNNpFDcCWJQBdwBSsE6zJu_0vGuy0GboR88_Q?e=9VvZCd)

- **百度网盘：**
[https://pan.baidu.com/s/1acP6LETBLVJZofjhxWkC4g?pwd=6rx5](https://pan.baidu.com/s/1acP6LETBLVJZofjhxWkC4g?pwd=6rx5)

- **夸克网盘：**
[https://pan.quark.cn/s/f9a0ab709f08](https://pan.quark.cn/s/f9a0ab709f08)

- **Nextcloud：**
[https://dav.napreth.com/index.php/s/BZpkp9487w4tWwQ](https://dav.napreth.com/index.php/s/BZpkp9487w4tWwQ)

## **项目结构**

```
VideoSemanticRepresentation/
├─ data/
│  ├─ raw/                  # 原始视频
│  ├─ slice/                # 视频切片（每 5 秒一段）
│  └─ features/             # 输出特征 (.npy)
├─ src/
│  ├─ __main__.py           # 主入口：python -m src（参考视频 vs 查询视频）
│  ├─ video.py              # 视频读取与灰度化（OpenCV -> CuPy）
│  ├─ feature.py            # 卷积核构造、3D 卷积与特征聚合、缓存
│  ├─ search.py             # 特征空间片段检索（滑窗 + 欧氏距离）
│  └─ __init__.py           # 包信息
├─ docs/
│  └─ README_EN.md          # 英文文档
├─ requirements.txt
├─ README.md
└─ LICENSE
```

## **模块说明**

| 模块 | 功能 |
| --- | --- |
| `src/video.py` | 读取视频并按帧生成灰度图序列 |
| `src/feature.py` | 定义/管理卷积核，执行 3D 卷积并聚合为时序特征，含缓存机制 |
| `src/search.py` | 在参考视频特征中滑动窗口，按欧氏距离定位与查询视频最相似的片段 |
| `src/__main__.py` | 程序主入口：计算两段视频的特征并执行检索 |
