# Early Fire and Smoke Tracking and Detection using YOLOv8

A real-time deep learning pipeline for tracking and detecting fire and smoke in video streams using YOLOv8. The model is fine-tuned on custom datasets to provide highly accurate detection boundaries for fire and smoke.

Developed by **Saud Akbar**.

---

## 🚀 Getting Started

### 📋 Prerequisites
Ensure you have Python 3.8+ (Python 3.12 recommended) and `pip` installed.

Install the required dependencies:
```bash
pip install ultralytics roboflow
```

If using GPU acceleration for training or real-time inference, configure PyTorch with CUDA support.

---

## 💻 Running Inference

### Using the Python SDK
A custom Python script `run_fire_detection.py` is included in the root folder to load weights and output inference statistics.

Run the inference script:
```bash
python run_fire_detection.py
```

### Using the YOLO Command Line Interface
To run detection directly on a video or image via CLI:
```bash
yolo task=detect mode=predict model=runs/detect/train/weights/best.pt conf=0.25 source=demo.mp4 save=True
```
The outputs will be saved to `runs/detect/predict/`.

---

## 📈 Evaluation Results

The model has been evaluated on validation subsets, yielding the following performance curves and confusion matrices:

- **mAP50**: **85.7%**
- **Precision**: **82.8%**
- **Recall**: **87.8%**

### Training Metrics & Loss Curves
![results](/runs/detect/train/results.png)

### Confusion Matrix
![confusion_matrix](/runs/detect/train/confusion_matrix.png)

---

## ⚙️ Custom Dataset & Training

### 1. Preparing Dataset
Annotate and download your dataset from Roboflow Universe using:
```python
from roboflow import Roboflow
rf = Roboflow(api_key="your-api-key")
project = rf.workspace("custom-thxhn").project("fire-wrpgm")
dataset = project.version(8).download("yolov8")
```

### 2. Custom Training
Train a custom YOLOv8 model using:
```bash
yolo task=detect mode=train model=yolov8s.pt data=datasets/fire-8/data.yaml epochs=25 imgsz=800 plots=True
```

---

## 🛠️ Developer

- **Saud Akbar** - Sole Creator and Contributor
