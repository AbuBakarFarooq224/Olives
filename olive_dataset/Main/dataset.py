"""
Dataset classes for loading olive images and masks

Handles loading image-mask pairs from patchify/ and patchify_mask/ folders
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image


class OliveSegmentationDataset(Dataset):
    """
    Dataset for loading images and segmentation masks
    
    Dataset structure:
        patchify/              <- Original images
            image0_00.png
            image0_01.png
            ...
        patchify_mask/         <- Segmentation masks
            image0_00.png
            image0_01.png
            ...
    """
    
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir: Path to patchify folder (original images)
            mask_dir: Path to patchify_mask folder (segmentation masks)
            transform: Optional transforms
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        
        # Get all image files
        self.image_files = sorted(list(self.image_dir.glob('*.png')))
        
        # Verify masks exist
        print(f"Found {len(self.image_files)} images in {image_dir}")
        
        # Check for matching masks
        self.valid_pairs = []
        for img_path in self.image_files:
            mask_path = self.mask_dir / img_path.name
            if mask_path.exists():
                self.valid_pairs.append((img_path, mask_path))
        
        print(f"Found {len(self.valid_pairs)} matching image-mask pairs")
        
        if len(self.valid_pairs) == 0:
            raise ValueError("No matching image-mask pairs found!")
        
        # Standard normalization
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.valid_pairs[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        image = np.array(image).astype(np.float32) / 255.0
        
        # Normalize
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - self.mean[i]) / self.std[i]
        
        # Convert to tensor [C, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Load mask
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        
        # Convert to binary (0 or 1)
        mask = (mask > 127).astype(np.int64)
        mask = torch.from_numpy(mask).long()
        
        return image, mask


def create_data_loaders(image_dir, mask_dir, batch_size=8, train_split=0.8):
    """
    Create train and validation data loaders
    
    Args:
        image_dir: Path to patchify folder
        mask_dir: Path to patchify_mask folder
        batch_size: Batch size for training
        train_split: Fraction of data for training (rest for validation)
    
    Returns:
        train_loader, val_loader
    """
    # Create full dataset
    dataset = OliveSegmentationDataset(image_dir, mask_dir)
    
    # Split into train/val
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\nDataset split:")
    print(f"  Training: {train_size} images")
    print(f"  Validation: {val_size} images")
    
    return train_loader, val_loader
