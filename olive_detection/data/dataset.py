"""
=============================================================
 OliveVision — Dataset Module
 Handles data loading, augmentation, and preprocessing
 for olive detection and counting.
=============================================================
"""

import os
import cv2
import yaml
import random
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────────────────────
# Dataset Splitter
# ─────────────────────────────────────────────────────────────
def split_dataset(config: dict) -> None:
    """
    Splits raw dataset into train / val / test folders.
    Expects YOLO-format labels alongside images.

    Args:
        config: Loaded YAML config dict
    """
    dataset_cfg = config["dataset"]
    images_dir  = Path(dataset_cfg["images_dir"])
    labels_dir  = Path(dataset_cfg["labels_dir"])
    processed   = Path(dataset_cfg["processed_dir"])

    train_r = dataset_cfg["train_ratio"]
    val_r   = dataset_cfg["val_ratio"]

    # Collect image paths
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() in image_extensions
    ])

    if not image_files:
        raise FileNotFoundError(f"No images found in {images_dir}")

    random.seed(config["training"]["seed"])
    random.shuffle(image_files)

    n       = len(image_files)
    n_train = int(n * train_r)
    n_val   = int(n * val_r)

    splits = {
        "train": image_files[:n_train],
        "val":   image_files[n_train:n_train + n_val],
        "test":  image_files[n_train + n_val:],
    }

    for split_name, files in splits.items():
        for sub in ["images", "labels"]:
            (processed / split_name / sub).mkdir(parents=True, exist_ok=True)

        for img_path in files:
            lbl_path = labels_dir / (img_path.stem + ".txt")
            shutil.copy(img_path, processed / split_name / "images" / img_path.name)
            if lbl_path.exists():
                shutil.copy(lbl_path, processed / split_name / "labels" / lbl_path.name)

    print(f"✅ Dataset split complete → "
          f"Train: {len(splits['train'])} | "
          f"Val: {len(splits['val'])} | "
          f"Test: {len(splits['test'])}")


# ─────────────────────────────────────────────────────────────
# Augmentation Pipelines
# ─────────────────────────────────────────────────────────────
def build_train_transforms(image_size: int) -> A.Compose:
    """Returns heavy augmentation pipeline for training."""
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size, min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT, fill=(114, 114, 114)
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Rotate(limit=15, p=0.4, border_mode=cv2.BORDER_CONSTANT),
        A.RandomScale(scale_limit=0.4, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=50, val_shift_limit=40, p=0.5
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=0.1),
        A.ImageCompression(quality_range=(75, 100), p=0.2),
        A.GridDistortion(p=0.1),
        A.CoarseDropout(
            num_holes_range=(8, 8), hole_height_range=(32, 32), hole_width_range=(32, 32),
            fill=114, p=0.2
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3
    ))


def build_val_transforms(image_size: int) -> A.Compose:
    """Returns light transform pipeline for validation/test."""
    return A.Compose([
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(
            min_height=image_size, min_width=image_size,
            border_mode=cv2.BORDER_CONSTANT, fill=(114, 114, 114)
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.1
    ))


# ─────────────────────────────────────────────────────────────
# Mosaic Augmentation
# ─────────────────────────────────────────────────────────────
class MosaicAugmentation:
    """
    Mosaic augmentation: combines 4 images into one.
    Greatly improves detection of small objects (small olives).
    """
    def __init__(self, dataset, image_size: int = 640, p: float = 0.8):
        self.dataset    = dataset
        self.image_size = image_size
        self.p          = p

    def __call__(self, image, bboxes, class_labels):
        if random.random() > self.p:
            return image, bboxes, class_labels

        s = self.image_size
        # Choose center randomly
        cx = random.randint(s // 4, 3 * s // 4)
        cy = random.randint(s // 4, 3 * s // 4)

        mosaic_img = np.full((s, s, 3), 114, dtype=np.uint8)
        all_boxes, all_labels = [], []

        # Pick 3 additional random samples + current
        indices = [random.randint(0, len(self.dataset) - 1) for _ in range(3)]

        for i, (tile_img, tile_boxes, tile_labels) in enumerate(
            [self.dataset.get_raw(idx) for idx in indices]
        ):
            h, w = tile_img.shape[:2]

            # Determine placement quadrant
            if i == 0:  # top-left
                x1a, y1a = max(cx - w, 0), max(cy - h, 0)
                x2a, y2a = cx, cy
            elif i == 1:  # top-right
                x1a, y1a = cx, max(cy - h, 0)
                x2a, y2a = min(cx + w, s), cy
            elif i == 2:  # bottom-left
                x1a, y1a = max(cx - w, 0), cy
                x2a, y2a = cx, min(cy + h, s)
            else:  # bottom-right
                x1a, y1a = cx, cy
                x2a, y2a = min(cx + w, s), min(cy + h, s)

            x1b = w - (x2a - x1a)
            y1b = h - (y2a - y1a)
            x2b, y2b = w, h

            mosaic_img[y1a:y2a, x1a:x2a] = tile_img[y1b:y2b, x1b:x2b]

            # Adjust boxes to mosaic coordinates
            padw = x1a - x1b
            padh = y1a - y1b

            for box, lbl in zip(tile_boxes, tile_labels):
                xc, yc, bw, bh = box
                # Convert YOLO → pixel
                xc_p = xc * w + padw
                yc_p = yc * h + padh
                bw_p = bw * w
                bh_p = bh * h
                # Convert back to YOLO (relative to mosaic)
                xc_n = xc_p / s
                yc_n = yc_p / s
                bw_n = bw_p / s
                bh_n = bh_p / s

                if 0 < xc_n < 1 and 0 < yc_n < 1:
                    all_boxes.append([
                        np.clip(xc_n, 0.01, 0.99),
                        np.clip(yc_n, 0.01, 0.99),
                        np.clip(bw_n, 0.01, 0.99),
                        np.clip(bh_n, 0.01, 0.99),
                    ])
                    all_labels.append(lbl)

        # Also process original image
        original_boxes_adj, original_labels_adj = [], []
        for box, lbl in zip(bboxes, class_labels):
            xc, yc, bw, bh = box
            h_orig, w_orig = image.shape[:2]
            xc_p = xc * w_orig + (cx - w_orig)
            yc_p = yc * h_orig + (cy - h_orig)
            bw_p = bw * w_orig
            bh_p = bh * h_orig
            xc_n = np.clip(xc_p / s, 0.01, 0.99)
            yc_n = np.clip(yc_p / s, 0.01, 0.99)
            bw_n = np.clip(bw_p / s, 0.01, 0.99)
            bh_n = np.clip(bh_p / s, 0.01, 0.99)
            if 0 < xc_n < 1 and 0 < yc_n < 1:
                original_boxes_adj.append([xc_n, yc_n, bw_n, bh_n])
                original_labels_adj.append(lbl)

        all_boxes  = original_boxes_adj + all_boxes
        all_labels = original_labels_adj + all_labels

        return mosaic_img, all_boxes, all_labels


# ─────────────────────────────────────────────────────────────
# Olive Dataset
# ─────────────────────────────────────────────────────────────
class OliveDataset(Dataset):
    """
    PyTorch Dataset for olive detection.
    Loads images + YOLO-format labels.

    Label format per line:
        class_id  x_center  y_center  width  height   (all 0–1 normalized)
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        transform: Any = None,
        use_mosaic: bool = False,
        mosaic_prob: float = 0.8,
        image_size: int = 640,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform  = transform
        self.use_mosaic = use_mosaic
        self.image_size = image_size

        img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        self.image_files = sorted([
            f for f in self.images_dir.iterdir()
            if f.suffix.lower() in img_exts
        ])

        if not self.image_files:
            raise RuntimeError(f"No images found in {self.images_dir}")

        self.mosaic = MosaicAugmentation(self, image_size, mosaic_prob) \
            if use_mosaic else None

        print(f"📦 Dataset loaded: {len(self.image_files)} images from {self.images_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def _load_label(self, img_stem: str):
        """Returns list of [cls, xc, yc, w, h] in YOLO format."""
        lbl_path = self.labels_dir / (img_stem + ".txt")
        boxes, classes = [], []
        if lbl_path.exists():
            with open(lbl_path) as f:
                for line in f.read().strip().splitlines():
                    parts = line.split()
                    if len(parts) == 5:
                        cls_id = int(parts[0])
                        coords = list(map(float, parts[1:]))
                        boxes.append(coords)
                        classes.append(cls_id)
        return boxes, classes

    def get_raw(self, idx: int):
        """Return (image_np, boxes, class_labels) without transforms."""
        img_path = self.image_files[idx]
        image    = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        image    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes, class_labels = self._load_label(img_path.stem)
        return image, boxes, class_labels

    def __getitem__(self, idx: int):
        image, bboxes, class_labels = self.get_raw(idx)

        # Apply mosaic
        if self.mosaic and random.random() < 0.8:
            image, bboxes, class_labels = self.mosaic(image, bboxes, class_labels)

        # Apply albumentations transforms
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=bboxes,
                class_labels=class_labels
            )
            image        = transformed["image"]
            bboxes       = transformed["bboxes"]
            class_labels = transformed["class_labels"]

        # Build target tensor [N, 6]: [batch_idx(0), cls, xc, yc, w, h]
        if bboxes:
            boxes_t  = torch.tensor(bboxes, dtype=torch.float32)
            labels_t = torch.tensor(class_labels, dtype=torch.long).unsqueeze(1)
            targets  = torch.cat([
                torch.zeros(len(bboxes), 1),  # batch_idx placeholder
                labels_t.float(),
                boxes_t
            ], dim=1)
        else:
            targets = torch.zeros((0, 6), dtype=torch.float32)

        return image, targets

    @staticmethod
    def collate_fn(batch):
        """Custom collate: handles variable-length target tensors."""
        images, targets = zip(*batch)
        images = torch.stack(images, dim=0)

        for i, t in enumerate(targets):
            if t.shape[0] > 0:
                t[:, 0] = i  # Set batch index

        targets = torch.cat(targets, dim=0)
        return images, targets


# ─────────────────────────────────────────────────────────────
# DataLoader Factory
# ─────────────────────────────────────────────────────────────
def build_dataloaders(config: dict):
    """
    Builds train, val, and test DataLoaders from config.

    Returns:
        train_loader, val_loader, test_loader
    """
    ds_cfg   = config["dataset"]
    tr_cfg   = config["training"]
    aug_cfg  = config["augmentation"]
    img_size = ds_cfg["image_size"]
    proc_dir = Path(ds_cfg["processed_dir"])

    train_transform = build_train_transforms(img_size)
    val_transform   = build_val_transforms(img_size)

    train_ds = OliveDataset(
        images_dir=proc_dir / "train" / "images",
        labels_dir=proc_dir / "train" / "labels",
        transform=train_transform,
        use_mosaic=aug_cfg["train"].get("mosaic", 0) > 0,
        mosaic_prob=aug_cfg["train"].get("mosaic", 0.8),
        image_size=img_size,
    )

    val_ds = OliveDataset(
        images_dir=proc_dir / "val" / "images",
        labels_dir=proc_dir / "val" / "labels",
        transform=val_transform,
        use_mosaic=False,
        image_size=img_size,
    )

    test_ds = OliveDataset(
        images_dir=proc_dir / "test" / "images",
        labels_dir=proc_dir / "test" / "labels",
        transform=val_transform,
        use_mosaic=False,
        image_size=img_size,
    )

    batch_size = int(tr_cfg["batch_size"])
    num_workers = int(tr_cfg["num_workers"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=OliveDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=OliveDataset.collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=OliveDataset.collate_fn,
    )

    print(f"✅ DataLoaders ready | Train: {len(train_ds)} | "
          f"Val: {len(val_ds)} | Test: {len(test_ds)}")

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────
# YOLO data.yaml generator
# ─────────────────────────────────────────────────────────────
def generate_data_yaml(config: dict, output_path: str = "data/data.yaml") -> str:
    """
    Generates a data.yaml file compatible with Ultralytics training format.
    """
    ds_cfg   = config["dataset"]
    proc_dir = Path(ds_cfg["processed_dir"]).resolve()

    data = {
        "path":  str(proc_dir),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    ds_cfg["num_classes"],
        "names": ds_cfg["classes"],
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"✅ data.yaml written → {output}")
    return str(output)
