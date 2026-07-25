# 🫒 OliveVision — Real-Time Olive Detection & Counting

**Final Year Project (FYP)**  
Architecture: **RT-DETR-L** with custom enhancements (CBAM + BiFPN + IoU-Aware Head)

---

## Why RT-DETR?

RT-DETR (Real-Time Detection Transformer) is the state-of-the-art architecture for this task because:

| Property | Benefit for Olive Detection |
|---|---|
| End-to-end (no NMS) | Faster inference, no hand-tuned thresholds |
| Transformer decoder | Handles dense, overlapping olives excellently |
| ResNet-101-D backbone | Rich hierarchical features from COCO pretraining |
| Real-time capable | 35+ FPS on RTX 3090 at 640×640 |

**Custom enhancements:**
- **CBAM** (Convolutional Block Attention Module) → better texture/shape discrimination
- **BiFPN** (Bi-directional FPN) → stronger multi-scale feature fusion
- **IoU-Aware Head** → ranking detections by predicted IoU, not just class score

---

## Project Structure

```
olive_detection/
├── config/
│   └── config.yaml          ← All hyperparameters & settings
│
├── data/
│   ├── dataset.py           ← Dataset, augmentation, DataLoaders
│   ├── raw/                 ← YOUR IMAGES GO HERE
│   │   ├── images/          ← .jpg / .png olive images
│   │   └── labels/          ← YOLO .txt labels
│   └── processed/           ← Auto-created after split
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/
│   ├── architecture.py      ← RT-DETR + CBAM + BiFPN + IoU-Head
│   ├── losses.py            ← Hungarian matching + Focal + GIoU loss
│   └── checkpoints/         ← Saved weights (best.pt, etc.)
│
├── inference/
│   └── infer.py             ← Real-time webcam/video inference + ByteTracker
│
├── utils/
│   └── utils.py             ← Metrics, visualization, checkpointing
│
├── logs/                    ← TensorBoard logs, training curves
│
├── train.ipynb              ← 📓 TRAINING NOTEBOOK (start here)
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare your dataset
Put your olive images in `data/raw/images/` and YOLO labels in `data/raw/labels/`.

**Label format** (one line per olive):
```
0  x_center  y_center  width  height
```
All values normalized 0–1. Class ID = 0.

### 3. Train
Open `train.ipynb` in Jupyter and run all cells.

```bash
jupyter notebook train.ipynb
```

Or run from CLI:
```bash
# The notebook handles everything — dataset split, training, evaluation, export
```

### 4. Run real-time inference
```bash
# Webcam
python inference/infer.py --weights models/checkpoints/best.pt --source 0

# Video file
python inference/infer.py --weights models/checkpoints/best.pt --source video.mp4

# Single image
python inference/infer.py --weights models/checkpoints/best.pt --image olive_image.jpg

# With ROI and lower confidence
python inference/infer.py --weights models/checkpoints/best.pt --source 0 --conf 0.3
```

**Keyboard shortcuts during inference:**
- `q` — quit
- `s` — save screenshot
- `r` — toggle ROI mode

---

## Configuration

All settings live in `config/config.yaml`. Key parameters:

```yaml
dataset:
  image_size: 640          # Input resolution
  train_ratio: 0.70        # 70% train, 20% val, 10% test

training:
  epochs: 150
  batch_size: 8            # Reduce to 4 if VRAM < 12GB
  learning_rate: 1.0e-4

inference:
  conf_threshold: 0.35     # Detection confidence threshold
  counting:
    use_tracking: true     # ByteTrack for stable video counting
    roi_enabled: false     # Enable to count only within a region
```

---

## Architecture Details

```
Input (640×640)
  → ResNet-101-D Backbone → [C3, C4, C5] feature maps
  → BiFPN (3 layers, d=256) → multi-scale fusion [P3, P4, P5]
  → CBAM on each level → channel + spatial attention
  → AIFI on P5 → global context via self-attention
  → Flatten + concat → Encoder Memory
  → 300 Object Queries → RT-DETR Decoder (6 layers)
  → IoU-Aware Head → class scores, boxes, IoU predictions
  → [Training] Hungarian matching + Focal + L1 + GIoU loss
  → [Inference] NMS-free output → ByteTrack → olive count
```

---

## Expected Results

| Metric | Expected Range |
|---|---|
| mAP@50 | 85–95% |
| mAP@50-95 | 60–75% |
| Count MAE | < 2 olives/image |
| Inference FPS | 35–60 FPS (RTX 3090) |

*Results depend heavily on dataset size and annotation quality.*

---

## Hardware Requirements

| GPU VRAM | Recommended batch_size |
|---|---|
| 24 GB (RTX 3090) | 8–16 |
| 12 GB (RTX 3080) | 4–8 |
| 8 GB (RTX 3070) | 4 |
| CPU only | Not recommended (very slow) |

**Google Colab** (free T4 GPU): set `batch_size: 4`, `image_size: 416`

---

## Files Description

| File | Purpose |
|---|---|
| `train.ipynb` | Main training notebook — run this |
| `config/config.yaml` | All hyperparameters |
| `data/dataset.py` | Data loading, augmentation, mosaic |
| `models/architecture.py` | CBAM, BiFPN, AIFI, IoU-Head modules |
| `models/losses.py` | Hungarian matcher, Focal, GIoU, IoU-aware loss |
| `inference/infer.py` | Real-time inference + ByteTrack counting |
| `utils/utils.py` | Metrics, visualization, checkpointing |

---

## Dataset Annotation Tips

For best results:
1. Annotate **all** olives (don't skip partially visible ones)
2. Use tight bounding boxes
3. Aim for **500+ images** minimum (1000+ ideal)
4. Include varied lighting, angles, and olive varieties
5. Tools: [Roboflow](https://roboflow.com), [LabelImg](https://github.com/HumanSignal/labelImg), [CVAT](https://cvat.ai)
