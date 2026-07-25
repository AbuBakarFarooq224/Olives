"""
Complete 3-stage Pipeline for Olive Counting

Manages training and inference for all three stages:
1. FCN Blob Detector (Segmentation)
2. CNN Counting Network
3. Linear Regression Correction
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from scipy import ndimage

from models import FCNBlobDetector, CNNCountingNetwork, LinearRegressionCorrection


class OliveCountingPipeline:
    """
    Complete 3-stage pipeline as described in the paper:
    Image -> FCN Segmentation -> Extract Blobs -> CNN Count per Blob -> 
    Sum Counts -> Linear Regression -> Final Count
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Initialize models
        self.fcn_segmenter = FCNBlobDetector().to(device)
        self.cnn_counter = CNNCountingNetwork(max_count=20).to(device)
        self.linear_regressor = LinearRegressionCorrection()
        
        # Image preprocessing
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
    
    def train_stage1_segmentation(self, train_loader, val_loader, epochs=50):
        """
        Train FCN blob detector
        
        Expected data format:
        - train_loader yields (images, masks) where:
          - images: [B, 3, H, W] RGB images
          - masks: [B, H, W] binary masks (1=fruit, 0=background)
        """
        print("\n" + "="*60)
        print("STAGE 1: Training FCN Blob Detector")
        print("="*60)
        
        self.fcn_segmenter.train()
        optimizer = torch.optim.Adam(self.fcn_segmenter.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        best_iou = 0
        
        for epoch in range(epochs):
            # Training
            train_loss = 0
            self.fcn_segmenter.train()
            
            for images, masks in train_loader:
                images = images.to(self.device)
                masks = masks.to(self.device).long()
                
                optimizer.zero_grad()
                outputs = self.fcn_segmenter(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            val_iou = self._validate_segmentation(val_loader)
            
            avg_train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}, Val IoU: {val_iou:.4f}")
            
            # Save best model
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(self.fcn_segmenter.state_dict(), 'fcn_segmenter_best.pth')
                print(f"  → New best IoU: {val_iou:.4f}")
        
        print(f"\n✓ Stage 1 complete! Best IoU: {best_iou:.4f}")
    
    def train_stage2_counting(self, train_loader, val_loader, epochs=50):
        """
        Train CNN counting network
        
        Expected data format:
        - train_loader yields (blob_images, counts) where:
          - blob_images: [B, 3, H, W] cropped blob regions
          - counts: [B] integer counts (0 to max_count)
        """
        print("\n" + "="*60)
        print("STAGE 2: Training CNN Counting Network")
        print("="*60)
        
        self.cnn_counter.train()
        optimizer = torch.optim.Adam(self.cnn_counter.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        best_acc = 0
        
        for epoch in range(epochs):
            # Training
            train_loss = 0
            self.cnn_counter.train()
            
            for images, counts in train_loader:
                images = images.to(self.device)
                counts = counts.to(self.device).long()
                
                optimizer.zero_grad()
                outputs = self.cnn_counter(images)
                loss = criterion(outputs, counts)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            val_acc, val_mae = self._validate_counting(val_loader)
            
            avg_train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f}, "
                  f"Val Acc: {val_acc:.4f}, MAE: {val_mae:.2f}")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.cnn_counter.state_dict(), 'cnn_counter_best.pth')
                print(f"  → New best accuracy: {val_acc:.4f}")
        
        print(f"\n✓ Stage 2 complete! Best accuracy: {best_acc:.4f}")
    
    def train_stage3_regression(self, image_paths, ground_truth_counts):
        """
        Train linear regression
        
        Args:
            image_paths: List of image file paths
            ground_truth_counts: List of corresponding true counts
        """
        print("\n" + "="*60)
        print("STAGE 3: Training Linear Regression")
        print("="*60)
        
        # Get CNN predictions for all images
        cnn_predictions = []
        
        self.fcn_segmenter.eval()
        self.cnn_counter.eval()
        
        for img_path in image_paths:
            count, _ = self.predict_without_regression(img_path)
            cnn_predictions.append(count)
        
        # Fit linear regression
        self.linear_regressor.fit(cnn_predictions, ground_truth_counts)
        
        # Show improvement
        before_mae = np.mean(np.abs(np.array(cnn_predictions) - np.array(ground_truth_counts)))
        after_predictions = [self.linear_regressor.predict(c) for c in cnn_predictions]
        after_mae = np.mean(np.abs(np.array(after_predictions) - np.array(ground_truth_counts)))
        
        print(f"\nMAE before regression: {before_mae:.2f}")
        print(f"MAE after regression: {after_mae:.2f}")
        print(f"Improvement: {before_mae - after_mae:.2f}")
        
        print("\n✓ Stage 3 complete!")
    
    def predict(self, image_path, visualize=False):
        """
        Complete pipeline prediction
        
        Args:
            image_path: Path to input image
            visualize: If True, return visualization
            
        Returns:
            final_count: Predicted olive count
            visualization: (optional) Image with segmentation overlay
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        img_array = np.array(image)
        
        # Prepare for FCN
        img_tensor = self._preprocess_image(image).unsqueeze(0).to(self.device)
        
        # Stage 1: Segment blobs
        self.fcn_segmenter.eval()
        with torch.no_grad():
            seg_output = self.fcn_segmenter(img_tensor)
            seg_mask = torch.argmax(seg_output, dim=1).cpu().numpy()[0]
        
        # Extract individual blobs using connected components
        labeled_mask, num_blobs = ndimage.label(seg_mask)
        
        # Stage 2: Count fruits in each blob
        total_count = 0
        self.cnn_counter.eval()
        
        for blob_id in range(1, num_blobs + 1):
            # Extract blob region
            blob_mask = (labeled_mask == blob_id)
            
            # Get bounding box
            rows, cols = np.where(blob_mask)
            if len(rows) == 0:
                continue
            
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            
            # Add padding
            padding = 10
            min_row = max(0, min_row - padding)
            max_row = min(img_array.shape[0], max_row + padding)
            min_col = max(0, min_col - padding)
            max_col = min(img_array.shape[1], max_col + padding)
            
            # Extract and count
            blob_crop = img_array[min_row:max_row, min_col:max_col]
            
            # Skip very small blobs
            if blob_crop.shape[0] < 20 or blob_crop.shape[1] < 20:
                continue
            
            blob_pil = Image.fromarray(blob_crop)
            blob_tensor = self._preprocess_image(blob_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                count_output = self.cnn_counter(blob_tensor)
                blob_count = torch.argmax(count_output, dim=1).item()
            
            total_count += blob_count
        
        # Stage 3: Apply linear regression
        final_count = self.linear_regressor.predict(total_count)
        
        if visualize:
            # Create visualization
            vis_image = img_array.copy()
            # Overlay segmentation
            overlay = np.zeros_like(vis_image)
            overlay[seg_mask == 1] = [0, 255, 0]  # Green for fruit
            vis_image = cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0)
            
            # Add text
            cv2.putText(vis_image, f"Count: {final_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            return final_count, vis_image
        
        return final_count
    
    def predict_without_regression(self, image_path):
        """Predict without stage 3 regression (for training regression)"""
        # Same as predict but skip regression
        image = Image.open(image_path).convert('RGB')
        img_array = np.array(image)
        img_tensor = self._preprocess_image(image).unsqueeze(0).to(self.device)
        
        self.fcn_segmenter.eval()
        with torch.no_grad():
            seg_output = self.fcn_segmenter(img_tensor)
            seg_mask = torch.argmax(seg_output, dim=1).cpu().numpy()[0]
        
        labeled_mask, num_blobs = ndimage.label(seg_mask)
        
        total_count = 0
        self.cnn_counter.eval()
        
        for blob_id in range(1, num_blobs + 1):
            blob_mask = (labeled_mask == blob_id)
            rows, cols = np.where(blob_mask)
            if len(rows) == 0:
                continue
            
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            
            padding = 10
            min_row = max(0, min_row - padding)
            max_row = min(img_array.shape[0], max_row + padding)
            min_col = max(0, min_col - padding)
            max_col = min(img_array.shape[1], max_col + padding)
            
            blob_crop = img_array[min_row:max_row, min_col:max_col]
            
            if blob_crop.shape[0] < 20 or blob_crop.shape[1] < 20:
                continue
            
            blob_pil = Image.fromarray(blob_crop)
            blob_tensor = self._preprocess_image(blob_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                count_output = self.cnn_counter(blob_tensor)
                blob_count = torch.argmax(count_output, dim=1).item()
            
            total_count += blob_count
        
        return total_count, seg_mask
    
    def _preprocess_image(self, pil_image):
        """Preprocess image for models"""
        # Resize to 512x512 (adjust as needed)
        pil_image = pil_image.resize((512, 512))
        img_array = np.array(pil_image).astype(np.float32) / 255.0
        
        # Normalize
        for i in range(3):
            img_array[:, :, i] = (img_array[:, :, i] - self.mean[i]) / self.std[i]
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        return img_tensor
    
    def _validate_segmentation(self, val_loader):
        """Calculate IoU for segmentation"""
        self.fcn_segmenter.eval()
        total_iou = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.fcn_segmenter(images)
                preds = torch.argmax(outputs, dim=1)
                
                # Calculate IoU
                intersection = ((preds == 1) & (masks == 1)).float().sum((1, 2))
                union = ((preds == 1) | (masks == 1)).float().sum((1, 2))
                iou = (intersection / (union + 1e-6)).mean()
                
                total_iou += iou.item()
        
        return total_iou / len(val_loader)
    
    def _validate_counting(self, val_loader):
        """Calculate accuracy and MAE for counting"""
        self.cnn_counter.eval()
        correct = 0
        total = 0
        abs_errors = []
        
        with torch.no_grad():
            for images, counts in val_loader:
                images = images.to(self.device)
                counts = counts.to(self.device)
                
                outputs = self.cnn_counter(images)
                preds = torch.argmax(outputs, dim=1)
                
                correct += (preds == counts).sum().item()
                total += counts.size(0)
                
                abs_errors.extend((preds - counts).abs().cpu().numpy())
        
        accuracy = correct / total
        mae = np.mean(abs_errors)
        
        return accuracy, mae
    
    def load_models(self, fcn_path=None, cnn_path=None, regression_params=None):
        """Load trained models"""
        if fcn_path:
            self.fcn_segmenter.load_state_dict(torch.load(fcn_path))
            print(f"Loaded FCN from {fcn_path}")
        
        if cnn_path:
            self.cnn_counter.load_state_dict(torch.load(cnn_path))
            print(f"Loaded CNN from {cnn_path}")
        
        if regression_params:
            self.linear_regressor.a = regression_params['a']
            self.linear_regressor.b = regression_params['b']
            self.linear_regressor.trained = True
            print(f"Loaded regression: y = {self.linear_regressor.a:.4f}x + {self.linear_regressor.b:.4f}")
