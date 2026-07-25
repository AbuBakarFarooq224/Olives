"""
Data Augmentation for Olive Segmentation

Adds augmentation to help model generalize better:
- Random flips
- Random rotations  
- Color jittering
- Random crops
- Brightness/contrast changes

This should SIGNIFICANTLY improve your model!
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance
import random


class AugmentedOliveDataset(Dataset):
    """
    Olive dataset with heavy augmentation
    
    This will make your model MUCH more robust!
    """
    
    def __init__(self, image_dir, mask_dir, augment=True, img_size=512):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.augment = augment
        self.img_size = img_size
        
        # Get all image files
        self.image_files = sorted(list(self.image_dir.glob('*.png')))
        
        # Check for matching masks
        self.valid_pairs = []
        for img_path in self.image_files:
            mask_path = self.mask_dir / img_path.name
            if mask_path.exists():
                self.valid_pairs.append((img_path, mask_path))
        
        print(f"Found {len(self.valid_pairs)} image-mask pairs")
        
        if len(self.valid_pairs) == 0:
            raise ValueError("No matching image-mask pairs found!")
        
        # Normalization (ImageNet stats)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def augment_pair(self, image, mask):
        """
        Apply same augmentation to both image and mask
        
        This is CRITICAL: image and mask must get same geometric transforms!
        """
        # Resize
        image = TF.resize(image, (self.img_size, self.img_size))
        mask = TF.resize(mask, (self.img_size, self.img_size), 
                        interpolation=Image.NEAREST)
        
        if self.augment:
            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Random vertical flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            
            # Random rotation (0, 90, 180, 270 degrees)
            if random.random() > 0.5:
                angle = random.choice([0, 90, 180, 270])
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)
            
            # Color jittering (ONLY on image, not mask!)
            if random.random() > 0.5:
                # Brightness
                factor = random.uniform(0.8, 1.2)
                image = ImageEnhance.Brightness(image).enhance(factor)
            
            if random.random() > 0.5:
                # Contrast
                factor = random.uniform(0.8, 1.2)
                image = ImageEnhance.Contrast(image).enhance(factor)
            
            if random.random() > 0.5:
                # Saturation
                factor = random.uniform(0.8, 1.2)
                image = ImageEnhance.Color(image).enhance(factor)
            
            # Random crop and resize back
            if random.random() > 0.3:
                i, j, h, w = transforms.RandomCrop.get_params(
                    image, output_size=(int(self.img_size*0.85), int(self.img_size*0.85))
                )
                image = TF.crop(image, i, j, h, w)
                mask = TF.crop(mask, i, j, h, w)
                image = TF.resize(image, (self.img_size, self.img_size))
                mask = TF.resize(mask, (self.img_size, self.img_size), 
                                interpolation=Image.NEAREST)
        
        return image, mask
    
    def __getitem__(self, idx):
        img_path, mask_path = self.valid_pairs[idx]
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Apply augmentations
        image, mask = self.augment_pair(image, mask)
        
        # Convert to numpy
        image = np.array(image).astype(np.float32) / 255.0
        mask = np.array(mask)
        
        # Normalize image
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - self.mean[i]) / self.std[i]
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        
        # Convert mask to binary
        mask = (mask > 127).astype(np.int64)
        mask = torch.from_numpy(mask).long()
        
        return image, mask


def create_augmented_loaders(image_dir, mask_dir, batch_size=8, 
                             train_split=0.8, img_size=512):
    """
    Create data loaders with augmentation
    
    Args:
        image_dir: Path to images
        mask_dir: Path to masks
        batch_size: Batch size
        train_split: Train/val split ratio
        img_size: Image size (512 recommended)
    
    Returns:
        train_loader, val_loader
    """
    
    # Create datasets (train with augmentation, val without)
    train_dataset = AugmentedOliveDataset(
        image_dir, mask_dir, 
        augment=True,  # ← Augment training data
        img_size=img_size
    )
    
    val_dataset = AugmentedOliveDataset(
        image_dir, mask_dir,
        augment=False,  # ← Don't augment validation
        img_size=img_size
    )
    
    # Split
    total_size = len(train_dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size
    
    # Use same seed for consistent splits
    generator = torch.Generator().manual_seed(42)
    
    train_indices, val_indices = torch.utils.data.random_split(
        range(total_size), [train_size, val_size], generator=generator
    )
    
    # Create subset datasets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices.indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"\n📊 Dataset Info:")
    print(f"  Total images: {total_size}")
    print(f"  Training: {train_size} (with augmentation)")
    print(f"  Validation: {val_size} (no augmentation)")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}x{img_size}\n")
    
    return train_loader, val_loader


# ==================== VISUALIZE AUGMENTATIONS ====================

def visualize_augmentations(image_path, mask_path, num_samples=6):
    """
    Visualize what augmentations look like
    
    Useful to verify augmentations are working correctly!
    """
    import matplotlib.pyplot as plt
    
    print("Generating augmentation examples...")
    
    # Create dataset
    dataset = AugmentedOliveDataset(
        image_dir=str(Path(image_path).parent),
        mask_dir=str(Path(mask_path).parent),
        augment=True
    )
    
    # Get multiple augmented versions of same image
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples*3))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for i in range(num_samples):
        # Get augmented sample (same index, different augmentation each time)
        image, mask = dataset[0]
        
        # Denormalize image
        img = image.numpy().transpose(1, 2, 0)
        img = (img * std + mean).clip(0, 1)
        
        mask_np = mask.numpy()
        
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Augmented Image {i+1}')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title(f'Augmented Mask {i+1}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_examples.png', dpi=200)
    plt.show()
    print("✓ Saved to augmentation_examples.png")


if __name__ == "__main__":
    # Example usage
    IMAGE_DIR = r'C:\Users\Welcome\OneDrive\Desktop\FYP MATERIAL\patchify'
    MASK_DIR = r'C:\Users\Welcome\OneDrive\Desktop\FYP MATERIAL\patchify_mask'
    
    # Create augmented loaders
    train_loader, val_loader = create_augmented_loaders(
        IMAGE_DIR, MASK_DIR,
        batch_size=8,
        train_split=0.8,
        img_size=512
    )
    
    print(f"✓ Created augmented data loaders!")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Visualize augmentations
    image_files = list(Path(IMAGE_DIR).glob('*.png'))
    if image_files:
        visualize_augmentations(
            str(image_files[0]),
            str(Path(MASK_DIR) / image_files[0].name),
            num_samples=6
        )
