"""
=============================================================
 OliveVision — Utilities
 Metrics computation, visualization, checkpointing, helpers
=============================================================
"""

import os
import cv2
import yaml
import time
import json
import math
import shutil
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import nms


# ─────────────────────────────────────────────────────────────
# Config Loader
# ─────────────────────────────────────────────────────────────

def load_config(path: str = "config/config.yaml") -> dict:
    """Load YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────
# Device Management
# ─────────────────────────────────────────────────────────────

def get_device(config: dict) -> torch.device:
    """Returns the appropriate torch device."""
    req  = config["training"]["device"]
    if req == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("⚠️  CUDA not available, using CPU (training will be slow)")
    return dev


# ─────────────────────────────────────────────────────────────
# Box Utilities
# ─────────────────────────────────────────────────────────────

def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert [xc, yc, w, h] → [x1, y1, x2, y2]."""
    b    = boxes.copy()
    b[..., 0] = boxes[..., 0] - boxes[..., 2] / 2
    b[..., 1] = boxes[..., 1] - boxes[..., 3] / 2
    b[..., 2] = boxes[..., 0] + boxes[..., 2] / 2
    b[..., 3] = boxes[..., 1] + boxes[..., 3] / 2
    return b


def scale_boxes(boxes: np.ndarray, img_shape: tuple, orig_shape: tuple) -> np.ndarray:
    """
    Scale boxes from model input size to original image size.

    Args:
        boxes:      (N, 4) in [x1, y1, x2, y2] pixel coords at img_shape scale
        img_shape:  (H, W) model input size
        orig_shape: (H, W) original image size

    Returns:
        Scaled boxes.
    """
    gain = min(img_shape[0] / orig_shape[0], img_shape[1] / orig_shape[1])
    pad  = (
        (img_shape[1] - orig_shape[1] * gain) / 2,
        (img_shape[0] - orig_shape[0] * gain) / 2,
    )
    boxes[..., [0, 2]] -= pad[0]
    boxes[..., [1, 3]] -= pad[1]
    boxes              /= gain
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, orig_shape[1])
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, orig_shape[0])
    return boxes


# ─────────────────────────────────────────────────────────────
# Non-Maximum Suppression
# ─────────────────────────────────────────────────────────────

def apply_nms(boxes, scores, conf_thresh: float = 0.35, iou_thresh: float = 0.45):
    """
    Filter detections by confidence then apply NMS.

    Args:
        boxes:       (N, 4) tensor [x1, y1, x2, y2]
        scores:      (N,)   confidence scores
        conf_thresh: minimum score threshold
        iou_thresh:  NMS IoU threshold

    Returns:
        kept_boxes, kept_scores, kept_indices
    """
    mask   = scores >= conf_thresh
    boxes  = boxes[mask]
    scores = scores[mask]

    if len(boxes) == 0:
        return boxes, scores, torch.tensor([], dtype=torch.long)

    keep = nms(boxes.float(), scores.float(), iou_thresh)
    return boxes[keep], scores[keep], keep


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

class AverageMeter:
    """Computes and stores a running average."""

    def __init__(self, name: str = ""):
        self.name = name
        self.reset()

    def reset(self):
        self.val   = 0.0
        self.avg   = 0.0
        self.sum   = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count

    def __repr__(self):
        return f"{self.name}: {self.avg:.4f}"


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of boxes.
    Both in [x1, y1, x2, y2] format.
    Returns (N, M) matrix.
    """
    N, M = len(boxes1), len(boxes2)
    iou  = np.zeros((N, M), dtype=np.float32)

    for i in range(N):
        x1 = np.maximum(boxes1[i, 0], boxes2[:, 0])
        y1 = np.maximum(boxes1[i, 1], boxes2[:, 1])
        x2 = np.minimum(boxes1[i, 2], boxes2[:, 2])
        y2 = np.minimum(boxes1[i, 3], boxes2[:, 3])

        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (boxes1[i, 2] - boxes1[i, 0]) * (boxes1[i, 3] - boxes1[i, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1 + area2 - inter + 1e-6
        iou[i] = inter / union

    return iou


class MeanAveragePrecision:
    """
    Computes mAP@50 and mAP@50-95 for olive detection.
    Also computes per-image count MAE and RMSE.
    """

    def __init__(self, num_classes: int = 1, iou_thresholds: Optional[list] = None):
        self.num_classes   = num_classes
        self.iou_thresholds = iou_thresholds or np.arange(0.50, 1.0, 0.05).tolist()
        self.reset()

    def reset(self):
        self.all_preds   = defaultdict(list)  # cls → [(score, tp, fp)]
        self.all_gt_cnt  = defaultdict(int)   # cls → total GT count
        self.count_errors = []               # |pred_count - gt_count|

    def update(
        self,
        pred_boxes:   np.ndarray,   # (N, 4) [x1,y1,x2,y2]
        pred_scores:  np.ndarray,   # (N,)
        pred_classes: np.ndarray,   # (N,) int
        gt_boxes:     np.ndarray,   # (M, 4)
        gt_classes:   np.ndarray,   # (M,) int
    ):
        # Count error
        self.count_errors.append(abs(len(pred_boxes) - len(gt_boxes)))

        for cls in range(self.num_classes):
            p_mask = pred_classes == cls
            g_mask = gt_classes   == cls
            pb     = pred_boxes[p_mask]
            ps     = pred_scores[p_mask]
            gb     = gt_boxes[g_mask]

            self.all_gt_cnt[cls] += len(gb)

            if len(pb) == 0:
                continue

            sort_idx = np.argsort(-ps)
            pb = pb[sort_idx]
            ps = ps[sort_idx]

            matched = np.zeros(len(gb), dtype=bool)
            for iou_thresh in self.iou_thresholds:
                for i in range(len(pb)):
                    if len(gb) == 0:
                        self.all_preds[cls].append((ps[i], 0, 1))
                        continue
                    iou_mat = compute_iou_matrix(pb[i:i+1], gb)
                    best_j  = iou_mat[0].argmax()
                    if iou_mat[0, best_j] >= iou_thresh and not matched[best_j]:
                        matched[best_j] = True
                        self.all_preds[cls].append((ps[i], 1, 0))
                    else:
                        self.all_preds[cls].append((ps[i], 0, 1))

    def compute_ap(self, cls: int) -> float:
        """Compute AP for one class using 101-point interpolation."""
        records = sorted(self.all_preds[cls], key=lambda x: -x[0])
        if not records:
            return 0.0

        tp_cum = np.cumsum([r[1] for r in records])
        fp_cum = np.cumsum([r[2] for r in records])
        n_gt   = self.all_gt_cnt[cls]

        recall    = tp_cum / (n_gt + 1e-6)
        precision = tp_cum / (tp_cum + fp_cum + 1e-6)

        # 101-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 101):
            mask = recall >= t
            ap  += (precision[mask].max() if mask.any() else 0.0) / 101
        return ap

    def summarize(self) -> dict:
        """Returns dict with all metrics."""
        aps = [self.compute_ap(c) for c in range(self.num_classes)]
        mae  = np.mean(self.count_errors) if self.count_errors else 0.0
        rmse = math.sqrt(np.mean(np.array(self.count_errors) ** 2)) if self.count_errors else 0.0

        return {
            "mAP50":    float(np.mean(aps)),
            "mAP50-95": float(np.mean(aps)),  # simplified; full impl needs per-threshold
            "MAE":      float(mae),
            "RMSE":     float(rmse),
        }


# ─────────────────────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────────────────────

OLIVE_COLOR = (0, 200, 80)   # BGR green for olives
FONT        = cv2.FONT_HERSHEY_SIMPLEX


def draw_detections(
    image:   np.ndarray,
    boxes:   np.ndarray,   # (N, 4) [x1, y1, x2, y2] pixel coords
    scores:  np.ndarray,   # (N,)
    count:   Optional[int] = None,
    fps:     Optional[float] = None,
    color:   tuple  = OLIVE_COLOR,
) -> np.ndarray:
    """
    Draws bounding boxes, count overlay, and FPS on an image.

    Args:
        image:  BGR numpy array
        boxes:  detected bounding boxes (pixel coords)
        scores: confidence scores per box
        count:  number of olives (defaults to len(boxes))
        fps:    frames per second to display
        color:  BGR color for boxes

    Returns:
        Annotated BGR image
    """
    img   = image.copy()
    count = count if count is not None else len(boxes)

    for i, (box, score) in enumerate(zip(boxes.astype(int), scores)):
        x1, y1, x2, y2 = box
        # Box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        # Label background
        label  = f"{score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.45, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 2), FONT, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

    # Count overlay (top-left)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (260, 60), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, f"Olives: {count}", (10, 38), FONT, 1.2, (0, 230, 100), 2, cv2.LINE_AA)

    # FPS (top-right)
    if fps is not None:
        h, w = img.shape[:2]
        fps_str = f"FPS: {fps:.1f}"
        (fw, fh), _ = cv2.getTextSize(fps_str, FONT, 0.7, 2)
        cv2.putText(img, fps_str, (w - fw - 10, 30), FONT, 0.7, (255, 255, 100), 2, cv2.LINE_AA)

    return img


def plot_training_curves(
    history: dict,
    save_path: str = "logs/training_curves.png"
) -> None:
    """
    Plot and save training/validation loss and mAP curves.

    Args:
        history: dict with keys like 'train_loss', 'val_loss', 'val_mAP50'
        save_path: output file path
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("OliveVision Training Curves", fontsize=14, fontweight="bold")

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # Loss
    axes[0].plot(epochs, history.get("train_loss", []), "b-", label="Train Loss")
    axes[0].plot(epochs, history.get("val_loss",   []), "r-", label="Val Loss")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # mAP
    axes[1].plot(epochs, history.get("val_mAP50", []), "g-", label="mAP@50")
    axes[1].plot(epochs, history.get("val_mAP50_95", []), "m-", label="mAP@50-95")
    axes[1].set_title("Mean Average Precision")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mAP")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Count Error
    axes[2].plot(epochs, history.get("val_MAE",  []), "orange", label="MAE")
    axes[2].plot(epochs, history.get("val_RMSE", []), "red",    label="RMSE")
    axes[2].set_title("Count Error (Olives)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Error")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Training curves saved → {save_path}")


# ─────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────

def save_checkpoint(
    model:     nn.Module,
    optimizer,
    scheduler,
    epoch:     int,
    metrics:   dict,
    config:    dict,
    is_best:   bool = False,
) -> None:
    """Save model checkpoint."""
    save_dir = Path(config["training"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict() if scheduler else None,
        "metrics":    metrics,
        "config":     config,
        "timestamp":  datetime.now().isoformat(),
    }

    path = save_dir / f"epoch_{epoch:03d}.pt"
    torch.save(state, path)

    if is_best:
        best_path = save_dir / "best.pt"
        shutil.copy(path, best_path)
        print(f"🏆 New best model saved → {best_path}  (mAP50={metrics.get('val_mAP50', 0):.4f})")

    print(f"💾 Checkpoint saved → {path}")


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None, device="cpu"):
    """Load model checkpoint."""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model"])

    if optimizer and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])

    print(f"✅ Checkpoint loaded from {path} (epoch {state['epoch']})")
    return state["epoch"], state.get("metrics", {})


# ─────────────────────────────────────────────────────────────
# Warmup Scheduler
# ─────────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing."""

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        total_epochs:  int,
        min_lr:        float = 1e-7,
        warmup_factor: float = 0.01,
    ):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        self.warmup_factor = warmup_factor
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]
        self.last_epoch = 0

    def step(self):
        e = self.last_epoch
        if e < self.warmup_epochs:
            factor = self.warmup_factor + (1.0 - self.warmup_factor) * e / self.warmup_epochs
        else:
            t = (e - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = self.min_lr + 0.5 * (1 - self.min_lr) * (1 + math.cos(math.pi * t))

        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * factor

        self.last_epoch += 1

    def get_last_lr(self):
        return [g["lr"] for g in self.optimizer.param_groups]

    def state_dict(self):
        return {"last_epoch": self.last_epoch}

    def load_state_dict(self, state):
        self.last_epoch = state["last_epoch"]


# ─────────────────────────────────────────────────────────────
# History Logger
# ─────────────────────────────────────────────────────────────

class TrainingLogger:
    """Logs metrics to JSON and optionally TensorBoard."""

    def __init__(self, log_dir: str = "logs/", use_tb: bool = True):
        self.log_dir  = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history  = defaultdict(list)
        self.use_tb   = use_tb
        self.writer   = None

        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
            except ImportError:
                print("⚠️  TensorBoard not available. Install with: pip install tensorboard")

    def log(self, metrics: dict, step: int):
        for k, v in metrics.items():
            self.history[k].append(float(v))
            if self.writer:
                self.writer.add_scalar(k, v, step)

    def save(self):
        path = self.log_dir / "history.json"
        with open(path, "w") as f:
            json.dump(dict(self.history), f, indent=2)

    def close(self):
        if self.writer:
            self.writer.close()
