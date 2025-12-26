<div align="center">

# Worm-PostureNet




[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v11-red.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An End-to-End Deep Learning Framework for High-Throughput Pose Estimation and Behavioral Profiling in Caenorhabditis elegans

[Paper](#-citation) | [Installation](#-installation) | [Quick Start](#-quick-start) | [Analysis](#-behavioral-analysis)

</div>

---

## Overview


The system provides an automated solution for nematode pose estimation and behavioral analysis using **YOLO11-Pose** with 5 keypoints (head, 3 body segments, tail), enabling comprehensive tracking and quantification of locomotion patterns.

---

## Project Structure

```
├── datasets/                    # Training datasets
├── ultralytics/                 # YOLO11 framework
│
├── train.py                     # Model training script
├── predict.py                   # Single image inference
├── track_modified.py            # Video tracking with trajectory visualization
├── Information-writing.py       # Tracking data extraction (frame-by-frame)
│
├── Single-worm-analysis.py      # Individual worm behavioral analysis
├── Multi-worms-analysis.py      # Population-level behavioral analysis
│
└── video2image.py               # Video frame extraction utility
```

---

## Key Features

### Keypoint-Based Pose Estimation
- **5 anatomical keypoints**: Head → Body1 → Body2 → Body3 → Tail
- YOLO11-Pose architecture for real-time detection
- Robust tracking with persistent ID assignment

### Multi-Dimensional Analysis

#### Single-Worm Analysis (`Single-worm-analysis.py`)
- **Kinematics**: Speed profiles for each body segment
- **Direction**: Body angle, movement angle, direction changes
- **Curvature**: Segmental curvature and undulation patterns
- **Behavior Classification**: Forward, backward, turn, omega-turn, pause
- **Frequency Analysis**: Undulation frequency via power spectral density

#### Population Analysis (`Multi-worms-analysis.py`)
- **Group Trajectory Visualization**: Spatial distribution and exploration patterns
- **Population Statistics**: Aggregated speed, curvature, behavioral metrics
- **Comparative Analysis**: Inter-individual variability
- **Heatmaps**: Spatial occupancy and behavioral time budgets

---

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.x compatible GPU (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/celegans-yolo-tracking.git
cd celegans-yolo-tracking

# Install dependencies
pip install ultralytics opencv-python pandas numpy scipy matplotlib seaborn
```

**Key Dependencies:**
```txt
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

---

## Quick Start

### 1. Training

Train YOLO11-Pose on your annotated dataset:

```bash
python train.py
```

**Configuration** (in `train.py`):
- Model: `yolo11s-pose.pt`
- Dataset: `data.yaml` (YOLO pose format)
- Epochs: 300
- Image size: 640
- Batch size: 16

### 2. Single Image Prediction

Test the model on a single image:

```bash
python predict.py
```

**Inputs/Outputs:**
- Input: `imgs.jpg`
- Output: `outputs/_manual.jpg` (with colored keypoints)

### 3. Video Tracking

Process a video with trajectory visualization:

```bash
python track_modified.py
```

**Features:**
- Real-time tracking with persistent IDs
- Trajectory trails (last 250 frames)
- Bounding boxes and labels
- Output: `result1_track_Box.mp4`

### 4. Extract Tracking Data

Generate frame-by-frame tracking data:

```bash
python Information-writing.py
```

**Output Format** (`worm_tracking_data_wh.txt`):
```
frame  worm_id  x1  y1  x2  y2  x3  y3  x4  y4  x5  y5  w  h
1      5        120 150 125 160 130 170 135 180 140 190 8  45
```

Where:
- `(x1,y1)` = Head
- `(x2,y2)` = Body1
- `(x3,y3)` = Body2
- `(x4,y4)` = Body3
- `(x5,y5)` = Tail
- `w, h` = Bounding box width/height

---

## Behavioral Analysis

### Single-Worm Analysis

Perform comprehensive analysis on an individual worm:

```bash
python Single-worm-analysis.py
```

**Configuration:**
```python
FRAME_RATE = 2              # Video frame rate (FPS)
TARGET_WORM_ID = 10         # Worm ID to analyze
SAVE_DIR = "outputs/"       # Output directory
```

**Generated Outputs:**

1. **`worm_{ID}_trajectory.png`**
   - Movement path with time-coded colors
   - Start/end markers
   - Head direction arrows

2. **`worm_{ID}_speed_analysis.png`**
   - Center speed vs 5-point average
   - Individual keypoint speeds
   - Spatiotemporal heatmap

3. **`worm_{ID}_direction_analysis.png`**
   - Body angle over time
   - Direction change rate
   - Angular distribution (histogram + polar plot)

4. **`worm_{ID}_curvature_analysis.png`**
   - Curvature time series
   - Curvature vs speed correlation
   - Segmental curvature heatmap

5. **`worm_{ID}_behavior_stats.png`**
   - Behavior time budget (pie chart)
   - Temporal distribution
   - Speed/curvature distributions by behavior

6. **`worm_{ID}_undulation_analysis.png`**
   - Power spectral density
   - Wave propagation across body
   - Phase delays relative to head

**Extracted Features:**

| Category | Features |
|----------|----------|
| **Speed** | `center_speed`, `speed_head`, `speed_body1/2/3`, `speed_tail`, `avg_speed` |
| **Direction** | `movement_angle`, `body_angle`, `direction_change` |
| **Curvature** | `curvature` (mean of 3 segments), `dominant_freq`, `power_ratio` |
| **Behavior** | `forward`, `backward`, `turn`, `omega`, `pause` |

### Population Analysis

Analyze multiple worms simultaneously:

```bash
python Multi-worms-analysis.py
```

**Configuration:**
```python
SPEED_THRESHOLD = 0.05           # Forward/backward threshold
PAUSE_THRESHOLD = 0.01           # Stationary threshold
OMEGA_ANGLE_THRESHOLD = 120      # Omega turn detection (degrees)
COILING_CURVATURE_THRESHOLD = 2.0
PIROUETTE_ANGULAR_VELOCITY = 30  # deg/s
```

**Generated Outputs:**

1. **`population_behavior_analysis.png`** (9-panel figure)
   - Group trajectory overlay
   - Population speed (mean ± std)
   - Population curvature (mean ± std)
   - Body angle dynamics
   - Contraction ratio
   - Behavioral metrics boxplot
   - Speed-curvature scatter
   - Spatial occupancy heatmap
   - Behavioral time budget (pie chart)

2. **`population_behavior_animation.mp4`**
   - Real-time trajectory animation
   - Synchronized speed curves
   - Synchronized curvature curves

**Metrics Calculated:**

| Metric | Description |
|--------|-------------|
| `mean_speed` | Average velocity (pixels/s) |
| `flip_frequency` | Body flips per second (>90° changes) |
| `pause_ratio` | Fraction of time stationary |
| `omega_turns` | Count of omega-turn events |
| `forward_ratio` | Fraction of time moving forward |
| `backward_ratio` | Fraction of time moving backward |
| `coiling_ratio` | Fraction of time in high-curvature state |
| `pirouette_ratio` | Fraction of time in pirouette |
| `wave_frequency` | Undulation frequency (Hz) |

---

## Utilities

### Video to Frames

Extract frames from video for dataset preparation:

```bash
python video2image.py
```

**Configuration:**
```python
video_path = "outputs/output_video_tracked-with-wh.avi"
output_dir = "outputs/frames"
save_interval = 1  # Extract every N frames
```

---

## Data Format

### Training Data (`data.yaml`)

YOLO pose format with 5 keypoints:

```yaml
train: datasets/train/images
val: datasets/val/images
test: datasets/test/images

nc: 1  # Number of classes (worm)
kpt_shape: [5, 2]  # 5 keypoints, (x,y) coordinates

names: ['worm']
```

**Annotation Format** (TXT files):
```
class_id x_center y_center width height x1 y1 v1 x2 y2 v2 ... x5 y5 v5
```
Where `v` is visibility flag (0=not labeled, 1=labeled but occluded, 2=visible)

---

## Example Results

### Speed Analysis
- **Mean Speed**: 12.5 ± 3.2 pixels/s
- **Flip Frequency**: 0.15 Hz
- **Pause Ratio**: 25.3%

### Behavioral Distribution
- Forward: 45.2%
- Backward: 18.7%
- Turn: 15.6%
- Omega: 2.3%
- Pause: 18.2%

### Undulation Characteristics
- **Dominant Frequency**: 0.42 Hz
- **Phase Delay** (Head→Tail): 0.18 s

---

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@article{liu2025celegans,
  title={High-Throughput Behavioral Phenotyping Cascade Framework for C. elegans: 
         From Segmentation to Tracking and Multi-Dimensional Quantification},
  author={Liu, Xiaoke and Li, Boao and Huo, Jing and Han, Xiaoqing},
  journal={},
  year={2025},
  affiliation={Shandong Second Medical University}
}
```

---

## Acknowledgments

This work was supported by:
- **Natural Science Foundation of Shandong Province** (Grant No. ZR2024QF228, ZR2024QA176)
- **National Natural Science Foundation of China** (Grant No. 82301666)

We thank the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) team for their open-source framework.

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

---

## Contact

- **Corresponding Author**: Xiaoqing Han (hanxiaoqing@sdsmu.edu.cn)
- **Institution**: Shandong Second Medical University
- **Issues**: [GitHub Issues](https://github.com/HeanLiu/Worm-PostureNet/issues)

---
