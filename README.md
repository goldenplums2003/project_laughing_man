# Laughing Man | 笑脸男

[English](#english) | [中文](#中文)

---

## 中文

### 📖 项目简介

**Laughing Man** 是一个基于 YOLOv8 人脸检测的视频处理项目。它能够自动检测视频或摄像头流中的人脸，并在人脸上实时叠加自定义图像或动画序列。项目提供离线视频处理和在线实时摄像头处理两种模式。

### ✨ 核心特性

- **智能人脸检测**：使用 YOLOv8-face 进行快速准确的人脸检测
- **支持多种贴图**：
  - 单张 PNG 图像
  - PNG 序列动画
  - 自定义贴图素材库
- **双模式处理**：
  - 📹 **离线模式**：批量处理本地视频文件
  - 🎥 **在线模式**：实时处理摄像头输入并输出虚拟摄像头
- **灵活的参数配置**：轻松调整贴图大小、位置、动画速率等
- **高效的视频处理**：支持多种视频格式（MP4、MOV、AVI、MKV、M4V）

### 📁 项目结构

```
project_laughing_man/
├── laughing_man_offline.py          # 离线视频处理脚本
├── laughing_man_online.py           # 在线摄像头处理脚本
├── yolov8n-face.pt                  # YOLOv8 人脸检测模型
├── laughing_man.png                 # 示例贴图
├── laughing_man_png_sequence/       # 标准PNG序列素材
├── laughing_man_png_sequence_with_glitch/  # 带故障特效的PNG序列
├── tests/                           # 输入视频文件夹
├── outputs/                         # 输出视频文件夹
└── LICENSE                          # 项目许可证
```

### 🚀 快速开始

#### 环境要求

- Python 3.8+
- CUDA 11.0+ （如果使用 GPU）

#### 安装依赖

```bash
# 创建虚拟环境
conda create -n laughing_man python=3.10
conda activate laughing_man

# 安装依赖
pip install opencv-python numpy ultralytics pyvirtualcam
```

#### 离线模式（处理视频文件）

```bash
python laughing_man_offline.py
```

**工作流程**：
1. 将待处理的视频文件放入 `tests/` 文件夹
2. 脚本会自动检测视频中的所有人脸
3. 在每个人脸上叠加贴图素材
4. 处理完成的视频输出到 `outputs/` 文件夹

#### 在线模式（实时摄像头处理）

```bash
python laughing_man_online.py
```

**工作流程**：
1. 启动脚本时会打开摄像头
2. 实时检测人脸并叠加贴图
3. 输出虚拟摄像头流，可用于直播、视频通话等场景

### ⚙️ 配置参数

在脚本中修改以下参数来自定义效果：

#### 贴图素材设置
```python
OVERLAY_PATH = "laughing_man_png_sequence/*.png"
```
支持三种写法：
- `"path/image.png"` - 单张PNG图像
- `"path/to/folder"` - PNG文件夹（自动加载所有PNG）
- `"path/to/*.png"` - 通配符模式（匹配符合条件的PNG文件）

#### 贴图大小缩放
```python
OVERLAY_SIZE_SCALE = 3.0  # 贴图相对于人脸框的缩放比例
```

#### 动画序列帧率
```python
OVERLAY_SEQUENCE_FPS = 30.0  # PNG序列的播放帧率
```

#### 输入/输出路径
```python
INPUT_DIR = Path("tests")          # 输入视频文件夹
OUTPUT_DIR = Path("outputs")       # 输出视频文件夹
```

### 📋 支持的视频格式

- MP4
- MOV
- AVI
- MKV
- M4V

### 🎨 自定义贴图

#### 准备单张贴图
1. 准备一张 PNG 图像（推荐使用透明背景）
2. 放入项目根目录或指定文件夹
3. 修改脚本中的 `OVERLAY_PATH` 指向该图像

#### 准备 PNG 序列
1. 导出动画为 PNG 序列（文件名应包含数字，如 `frame_001.png`, `frame_002.png` 等）
2. 将所有 PNG 文件放入同一文件夹
3. 修改脚本中的 `OVERLAY_PATH` 指向该文件夹或使用通配符

### 🔧 高级用法

#### 调整人脸追踪行为
在脚本中修改 YOLO 追踪器设置：
```python
results = model.track(
    frame,
    persist=True,
    tracker="bytetrack.yaml",
    verbose=False
)
```

#### 自定义贴图位置和大小
修改脚本中的叠加逻辑，例如：
- 改变 `OVERLAY_SIZE_SCALE` 调整贴图大小
- 修改坐标计算逻辑改变贴图位置

### ⚡ 性能优化

- **使用 GPU**：确保安装了 CUDA，脚本会自动使用 GPU 进行推理
- **调整视频分辨率**：处理较小分辨率的视频可以加速处理
- **选择轻量级模型**：已使用 yolov8n（nano）模型以获得最佳性能

### 📝 许可证

本项目采用 [MIT License](LICENSE)

### 🤝 贡献

欢迎提交 Issue 和 Pull Request！

### 📞 联系方式

如有问题或建议，请通过项目的 Issue 页面反馈。

---

## English

### 📖 Project Overview

**Laughing Man** is a video processing project based on YOLOv8 face detection. It automatically detects faces in video files or camera streams and overlays custom images or animated sequences on them in real-time. The project supports both offline video processing and online real-time camera processing modes.

### ✨ Key Features

- **Smart Face Detection**: Fast and accurate face detection using YOLOv8-face
- **Multiple Overlay Support**:
  - Single PNG images
  - PNG sequence animations
  - Custom material libraries
- **Dual Processing Modes**:
  - 📹 **Offline Mode**: Batch process local video files
  - 🎥 **Online Mode**: Real-time camera processing with virtual camera output
- **Flexible Configuration**: Easy to adjust overlay size, position, animation frame rate, etc.
- **Efficient Processing**: Supports multiple video formats (MP4, MOV, AVI, MKV, M4V)

### 📁 Project Structure

```
project_laughing_man/
├── laughing_man_offline.py          # Offline video processing script
├── laughing_man_online.py           # Online camera processing script
├── yolov8n-face.pt                  # YOLOv8 face detection model
├── laughing_man.png                 # Sample overlay image
├── laughing_man_png_sequence/       # Standard PNG sequence materials
├── laughing_man_png_sequence_with_glitch/  # PNG sequence with glitch effects
├── tests/                           # Input video folder
├── outputs/                         # Output video folder
└── LICENSE                          # Project license
```

### 🚀 Getting Started

#### Requirements

- Python 3.8+
- CUDA 11.0+ (optional, for GPU support)

#### Installation

```bash
# Create virtual environment
conda create -n laughing_man python=3.10
conda activate laughing_man

# Install dependencies
pip install opencv-python numpy ultralytics pyvirtualcam
```

#### Offline Mode (Process Video Files)

```bash
python laughing_man_offline.py
```

**Workflow**:
1. Place videos to be processed in the `tests/` folder
2. The script automatically detects all faces in the video
3. Overlays the material on each detected face
4. Outputs processed videos to the `outputs/` folder

#### Online Mode (Real-time Camera Processing)

```bash
python laughing_man_online.py
```

**Workflow**:
1. The script opens your camera when started
2. Detects faces in real-time and applies overlays
3. Outputs a virtual camera stream for live streaming, video calls, etc.

### ⚙️ Configuration Parameters

Modify the following parameters in the script to customize effects:

#### Overlay Material Settings
```python
OVERLAY_PATH = "laughing_man_png_sequence/*.png"
```
Supports three formats:
- `"path/image.png"` - Single PNG image
- `"path/to/folder"` - PNG folder (automatically loads all PNG files)
- `"path/to/*.png"` - Wildcard pattern (matches PNG files)

#### Overlay Size Scale
```python
OVERLAY_SIZE_SCALE = 3.0  # Scale factor relative to face bounding box
```

#### Animation Sequence Frame Rate
```python
OVERLAY_SEQUENCE_FPS = 30.0  # Playback frame rate for PNG sequences
```

#### Input/Output Paths
```python
INPUT_DIR = Path("tests")          # Input video folder
OUTPUT_DIR = Path("outputs")       # Output video folder
```

### 📋 Supported Video Formats

- MP4
- MOV
- AVI
- MKV
- M4V

### 🎨 Custom Overlays

#### Prepare Single Overlay Image
1. Create a PNG image (preferably with transparent background)
2. Place it in the project root or specified folder
3. Update the `OVERLAY_PATH` in the script to point to the image

#### Prepare PNG Sequence
1. Export animation as PNG sequence (filenames should include numbers, e.g., `frame_001.png`, `frame_002.png`)
2. Place all PNG files in the same folder
3. Update the `OVERLAY_PATH` in the script to point to the folder or use wildcard pattern

### 🔧 Advanced Usage

#### Adjust Face Tracking Behavior
Modify the YOLO tracker settings in the script:
```python
results = model.track(
    frame,
    persist=True,
    tracker="bytetrack.yaml",
    verbose=False
)
```

#### Customize Overlay Position and Size
Modify the overlay logic in the script, for example:
- Adjust `OVERLAY_SIZE_SCALE` to change overlay size
- Modify coordinate calculation to change overlay position

### ⚡ Performance Optimization

- **Use GPU**: Ensure CUDA is installed; the script will automatically use GPU for inference
- **Adjust Video Resolution**: Processing lower resolution videos speeds up processing
- **Use Lightweight Model**: Already using yolov8n (nano) model for best performance

### 📝 License

This project is licensed under the [MIT License](LICENSE)

### 🤝 Contributing

Issues and pull requests are welcome!

### 📞 Contact

For questions or suggestions, please submit an issue on the project's issue page.

---

**Created with ❤️ for face detection and video processing**
