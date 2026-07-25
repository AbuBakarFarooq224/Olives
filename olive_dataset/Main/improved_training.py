"""
IMPROVED TRAINING SCRIPT - Fixes the black mask problem

Key improvements:
1. Train for 30+ epochs (not just 1!)
2. Validation after EVERY epoch
3. Early stopping to prevent overfitting
4. Learning rate scheduling
5. Better monitoring and logging
6. Save best model based on IoU

Run this instead of your train.ipynb!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Import your modules
from models import FCNBlobDetector
from dataset import create_data_loaders


class ImprovedTrainer:
    """Enhanced trainer with proper validation and monitoring"""
    
    def __init__(self, image_dir, mask_dir, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create data loaders
        self.train_loader, self.val_loader = create_data_loaders(
            image_dir=image_dir,
            mask_dir=mask_dir,
            batch_size=8,
            train_split=0.8
        )
        
        # Initialize model
        self.model = FCNBlobDetector().to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-5)
        
        # Learning rate scheduler (reduces LR when validation stops improving)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_iou': [],
            'val_precision': [],
            'val_recall': [],
            'learning_rates': []
        }
        
        self.best_iou = 0
        self.epochs_without_improvement = 0
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        num_batches = 0
        
        for images, masks in self.train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device).long()
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        return epoch_loss / num_batches
    
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        val_loss = 0
        num_batches = 0
        
        total_iou = 0
        total_precision = 0
        total_recall = 0
        num_samples = 0
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Predict
                outputs = self.model(images)
                loss = self.criterion(outputs, masks.long())
                val_loss += loss.item()
                num_batches += 1
                
                # Get predictions
                preds = torch.argmax(outputs, dim=1)
                
                # Calculate metrics per image
                for pred, mask in zip(preds, masks):
                    pred_np = pred.cpu().numpy()
                    mask_np = mask.cpu().numpy()
                    
                    # IoU
                    intersection = ((pred_np == 1) & (mask_np == 1)).sum()
                    union = ((pred_np == 1) | (mask_np == 1)).sum()
                    if union > 0:
                        total_iou += intersection / union
                    
                    # Precision
                    true_pos = ((pred_np == 1) & (mask_np == 1)).sum()
                    pred_pos = (pred_np == 1).sum()
                    if pred_pos > 0:
                        total_precision += true_pos / pred_pos
                    
                    # Recall
                    actual_pos = (mask_np == 1).sum()
                    if actual_pos > 0:
                        total_recall += true_pos / actual_pos
                    
                    num_samples += 1
        
        avg_val_loss = val_loss / num_batches
        avg_iou = total_iou / num_samples if num_samples > 0 else 0
        avg_precision = total_precision / num_samples if num_samples > 0 else 0
        avg_recall = total_recall / num_samples if num_samples > 0 else 0
        
        return avg_val_loss, avg_iou, avg_precision, avg_recall
    
    def train(self, epochs=30, early_stop_patience=10):
        """
        Train the model with validation
        
        Args:
            epochs: Number of epochs to train
            early_stop_patience: Stop if no improvement for this many epochs
        """
        print("\n" + "="*80)
        print(f"TRAINING FCN WITH PROPER VALIDATION")
        print(f"Epochs: {epochs} | Early stopping patience: {early_stop_patience}")
        print(f"Training samples: {len(self.train_loader)*8}")
        print(f"Validation samples: {len(self.val_loader)*8}")
        print("="*80 + "\n")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_iou, val_precision, val_recall = self.validate()
            
            # Update learning rate based on validation IoU
            self.scheduler.step(val_iou)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_iou'].append(val_iou)
            self.history['val_precision'].append(val_precision)
            self.history['val_recall'].append(val_recall)
            self.history['learning_rates'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            # Print progress
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val IoU: {val_iou:.4f} | "
                  f"Precision: {val_precision:.4f} | "
                  f"Recall: {val_recall:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.1f}s", end='')
            
            # Save best model
            if val_iou > self.best_iou:
                self.best_iou = val_iou
                torch.save(self.model.state_dict(), 'fcn_segmenter_best.pth')
                self.epochs_without_improvement = 0
                print(" ★ BEST")
            else:
                self.epochs_without_improvement += 1
                print()
            
            # Early stopping
            if self.epochs_without_improvement >= early_stop_patience:
                print(f"\n⚠️  Early stopping! No improvement for {early_stop_patience} epochs")
                break
        
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"✓ Training complete!")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Best validation IoU: {self.best_iou:.4f}")
        print(f"  Model saved to: fcn_segmenter_best.pth")
        print("="*80)
    
    def plot_training_history(self, save_path='training_curves_improved.png'):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # IoU
        axes[0, 1].plot(epochs, self.history['val_iou'], 'g-', linewidth=2)
        axes[0, 1].axhline(y=self.best_iou, color='r', linestyle='--', 
                           label=f'Best: {self.best_iou:.4f}')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('IoU')
        axes[0, 1].set_title('Validation IoU over Epochs')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision & Recall
        axes[1, 0].plot(epochs, self.history['val_precision'], 'b-', 
                       label='Precision', linewidth=2)
        axes[1, 0].plot(epochs, self.history['val_recall'], 'r-', 
                       label='Recall', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Precision and Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(epochs, self.history['learning_rates'], 'm-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training curves saved to {save_path}")
        plt.show()
    
    def test_predictions(self, num_samples=4, save_path='predictions_after_training.png'):
        """Visualize predictions after training"""
        self.model.eval()
        
        # Get validation batch
        val_images, val_masks = next(iter(self.val_loader))
        val_images = val_images.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(val_images)
            pred_masks = torch.argmax(outputs, dim=1).cpu().numpy()
        
        val_masks_np = val_masks.numpy()
        
        # Denormalize images
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        num_show = min(num_samples, len(val_images))
        fig, axes = plt.subplots(num_show, 3, figsize=(12, 4*num_show))
        
        if num_show == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_show):
            img = val_images[i].cpu().numpy().transpose(1, 2, 0)
            img = (img * std + mean).clip(0, 1)
            
            axes[i, 0].imshow(img)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(val_masks_np[i], cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_masks[i], cmap='gray')
            axes[i, 2].set_title(f'Prediction (IoU: {self.best_iou:.3f})')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Predictions saved to {save_path}")
        plt.show()


# ==================== MAIN EXECUTION ====================

def main():
    """Run improved training"""
    
    # Set your data paths
    IMAGE_DIR = r'C:\Users\Welcome\OneDrive\Desktop\FYP MATERIAL\patchify'
    MASK_DIR = r'C:\Users\Welcome\OneDrive\Desktop\FYP MATERIAL\patchify_mask'
    
    # Initialize trainer
    trainer = ImprovedTrainer(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        device='cuda'  # Use 'cpu' if no GPU
    )
    
    # Train for 30 epochs (not just 1!)
    trainer.train(epochs=30, early_stop_patience=10)
    
    # Plot results
    trainer.plot_training_history()
    
    # Show predictions
    trainer.test_predictions()
    
    print("\n🎉 Training complete! Check the saved images for results.")


if __name__ == "__main__":
    main()
