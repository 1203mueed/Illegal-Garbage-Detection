# Illegal Garbage Detection

This repository contains models and scripts for detecting and segmenting garbage in images using state-of-the-art object detection and instance segmentation techniques. The primary objective is to address illegal dumping and enhance waste management through AI-driven solutions.

## Project Structure

```
Illegal-Garbage-Detection/
├── README.md                 # Project documentation
├── Detectron 2/              # Detectron 2 implementation
│   └── detectron2.ipynb      # Jupyter notebook for Detectron 2 model
├── Mask R Cnn/               # Mask R-CNN implementation
│   ├── maskrcnn_train.py     # Training script for Mask R-CNN
│   ├── training_metrics.csv  # Training metrics log
│   └── training_metricsold.csv # Previous training metrics log
├── YOLO/                     # YOLO-based implementations
│   ├── custom.yaml           # Configuration for custom training
│   ├── yolov11Train.py       # YOLOv11 training script
│   ├── yolov8Train.py        # YOLOv8 training script
│   └── runs/                 # Training outputs and results
│       └── segment/          # Segmentation outputs
│           ├── train1/       # Training session 1 (YOLOv11 results)
│           │   ├── args.yaml     # Training arguments
│           │   ├── results.csv   # Training results
│           │   └── weights/      # Model weights
│           │       ├── best.pt   # Best model weights
│           │       └── last.pt   # Last model weights
│           └── train2/       # Training session 2 (YOLOv8 results)
│               ├── args.yaml     # Training arguments
│               ├── results.csv   # Training results
│               └── weights/      # Model weights
│                   ├── best.pt   # Best model weights
│                   └── last.pt   # Last model weights
└── requirements.txt          # Python dependencies
```

## Models Implemented

### 1. Detectron 2
- Framework: Facebook AI's Detectron2
- Features: Instance segmentation using Mask R-CNN.
- Script: `detectron2.ipynb`
- Note: It is recommended to run Detectron 2 on **Google Colab** for optimal performance and ease of setup.

### 2. Mask R-CNN
- Framework: PyTorch
- Features: Detect and segment garbage in images.
- Script: `maskrcnn_train.py`
- Metrics:
  - `training_metrics.csv` contains detailed logs of training progress.

### 3. YOLO
- Framework: Ultralytics YOLO (v8 and custom modifications).
- Features: Object detection and segmentation.
- Scripts:
  - `yolov11Train.py`: Train YOLOv11 for segmentation.
  - `yolov8Train.py`: Train YOLOv8 for segmentation.
- Results: YOLO achieved the **best results** in the experiments.

## Dataset

The experiment is conducted on a **custom dataset**. The dataset will be published soon to allow others to reproduce the results.

The dataset is organized in COCO format:
- Training images: `train/images`
- Validation images: `val/images`
- Annotations: `coco_training_annotations.json`, `coco_validation_annotations.json`

### Custom Dataset Configuration
Edit the `YOLO/custom.yaml` file to point to the dataset paths:
```yaml
train: /path/to/train
val: /path/to/val
nc: 1
names: ['Garbage']
```

## Usage

### 1. Install Dependencies
Install all required Python libraries:
```bash
pip install -r requirements.txt
```

### 2. Train Mask R-CNN
```bash
python Mask\ R\ Cnn/maskrcnn_train.py
```

### 3. Train YOLOv8
```bash
python YOLO/yolov8Train.py
```

### 4. Train YOLOv11
```bash
python YOLO/yolov11Train.py
```

### 5. Detectron 2
Run the Jupyter notebook `Detectron 2/detectron2.ipynb` to train and test the model. It is recommended to use **Google Colab** for this.

## Results

### YOLO
- Achieved the **best results** among the implemented models.
- Training outputs are stored in `YOLO/runs/segment/`.
- Results:
  - **Train1**: Results for YOLOv11.
    - Best model weights: `YOLO/runs/segment/train1/weights/best.pt`
    - Last model weights: `YOLO/runs/segment/train1/weights/last.pt`
  - **Train2**: Results for YOLOv8.
    - Best model weights: `YOLO/runs/segment/train2/weights/best.pt`
    - Last model weights: `YOLO/runs/segment/train2/weights/last.pt`

### Mask R-CNN
Training metrics are logged in `training_metrics.csv` and `training_metricsold.csv`.

## Future Improvements
- Optimize YOLO models for faster inference on edge devices.
- Experiment with hybrid models combining YOLO and Mask R-CNN.
- Deploy the best model as a real-time API.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgements
- [Detectron2](https://github.com/facebookresearch/detectron2)
- [Ultralytics YOLO](https://github.com/ultralytics/yolov5)
- PyTorch for Mask R-CNN.

---
Feel free to raise issues or contribute to this project! 🚀
