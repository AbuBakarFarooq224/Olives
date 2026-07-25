"""
Data Preparation for Segmentation-Based Counting
Creates segmentation masks from point annotations for the Chen et al. method
"""

import cv2
import numpy as np
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy import ndimage

class SegmentationMaskCreator:
    """
    Convert point annotations to segmentation masks
    Several strategies available:
    1. Circle-based: Draw circles around each point
    2. Watershed: Use watershed segmentation  
    3. GrabCut: Semi-automated segmentation
    4. Manual polygon annotation
    """
    
    @staticmethod
    def create_circle_masks(annotation_file, output_path, radius=15):
        """
        Simple approach: Draw circles around each annotated point
        
        Args:
            annotation_file: JSON with point annotations
            output_path: Where to save mask image
            radius: Radius of circles in pixels
        """
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        h, w = data['image_size']
        mask = np.zeros((h, w), dtype=np.uint8)
        
        # Draw circles at each point
        for x, y in data['points']:
            cv2.circle(mask, (int(x), int(y)), radius, 1, -1)
        
        # Save mask
        cv2.imwrite(str(output_path), mask * 255)
        
        return mask
    
    @staticmethod
    def create_watershed_masks(image_path, annotation_file, output_path):
        """
        Use watershed segmentation to separate touching olives
        More sophisticated than simple circles
        """
        # Load image and annotations
        image = cv2.imread(str(image_path))
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        h, w = image.shape[:2]
        
        # Create markers for watershed
        markers = np.zeros((h, w), dtype=np.int32)
        
        # Mark each point with a unique ID
        for i, (x, y) in enumerate(data['points'], start=1):
            markers[int(y), int(x)] = i
        
        # Apply watershed
        markers = cv2.watershed(image, markers)
        
        # Create binary mask (all fruit regions = 1, background = 0)
        mask = (markers > 0).astype(np.uint8)
        
        # Save
        cv2.imwrite(str(output_path), mask * 255)
        
        return mask
    
    @staticmethod
    def create_grabcut_masks(image_path, annotation_file, output_path, margin=20):
        """
        Use GrabCut for semi-automatic segmentation
        Creates a bounding box around all points and runs GrabCut
        """
        image = cv2.imread(str(image_path))
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        if len(data['points']) == 0:
            # No points, return empty mask
            mask = np.zeros(data['image_size'], dtype=np.uint8)
            cv2.imwrite(str(output_path), mask)
            return mask
        
        # Get bounding box of all points
        points = np.array(data['points'])
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        
        # Add margin
        x_min = max(0, int(x_min) - margin)
        y_min = max(0, int(y_min) - margin)
        x_max = min(image.shape[1], int(x_max) + margin)
        y_max = min(image.shape[0], int(y_max) + margin)
        
        # Rectangle for GrabCut
        rect = (x_min, y_min, x_max - x_min, y_max - y_min)
        
        # Run GrabCut
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Create binary mask
        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Save
        cv2.imwrite(str(output_path), mask * 255)
        
        return mask
    
    @staticmethod
    def create_color_based_masks(image_path, annotation_file, output_path):
        """
        Use color-based segmentation
        Good if olives have distinct color from background
        """
        image = cv2.imread(str(image_path))
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Sample colors at annotated points
        colors = []
        for x, y in data['points']:
            if 0 <= int(x) < image.shape[1] and 0 <= int(y) < image.shape[0]:
                colors.append(hsv[int(y), int(x)])
        
        if len(colors) == 0:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.imwrite(str(output_path), mask)
            return mask
        
        colors = np.array(colors)
        
        # Get color range
        h_min, s_min, v_min = colors.min(axis=0)
        h_max, s_max, v_max = colors.max(axis=0)
        
        # Expand range a bit
        h_margin = 10
        s_margin = 30
        v_margin = 30
        
        lower = np.array([max(0, h_min - h_margin), 
                         max(0, s_min - s_margin), 
                         max(0, v_min - v_margin)])
        upper = np.array([min(180, h_max + h_margin),
                         min(255, s_max + s_margin),
                         min(255, v_max + v_margin)])
        
        # Create mask
        mask = cv2.inRange(hsv, lower, upper)
        
        # Clean up with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Convert to binary
        mask = (mask > 0).astype(np.uint8)
        
        # Save
        cv2.imwrite(str(output_path), mask * 255)
        
        return mask


class BlobDatasetCreator:
    """
    Create dataset for Stage 2 (CNN counting)
    Extracts individual blobs and their counts
    """
    
    @staticmethod
    def extract_blobs_from_mask(image_path, mask_path, annotation_file, 
                                output_dir, min_area=100):
        """
        Extract individual blob regions from segmentation mask
        
        Args:
            image_path: Original image
            mask_path: Segmentation mask
            annotation_file: Point annotations (for counting per blob)
            output_dir: Where to save blob crops
            min_area: Minimum blob area in pixels
            
        Returns:
            List of (blob_image_path, count) tuples
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), 0)
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        points = np.array(data['points'])
        
        # Find connected components (blobs)
        labeled_mask, num_blobs = ndimage.label(mask > 0)
        
        blob_data = []
        
        for blob_id in range(1, num_blobs + 1):
            # Get blob mask
            blob_mask = (labeled_mask == blob_id).astype(np.uint8)
            
            # Get bounding box
            rows, cols = np.where(blob_mask)
            if len(rows) == 0:
                continue
            
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            
            # Check area
            area = (max_row - min_row) * (max_col - min_col)
            if area < min_area:
                continue
            
            # Add padding
            padding = 10
            min_row = max(0, min_row - padding)
            max_row = min(image.shape[0], max_row + padding)
            min_col = max(0, min_col - padding)
            max_col = min(image.shape[1], max_col + padding)
            
            # Extract crop
            blob_crop = image[min_row:max_row, min_col:max_col]
            
            # Count points in this blob
            count = 0
            for x, y in points:
                if min_col <= x <= max_col and min_row <= y <= max_row:
                    # Check if point is in blob
                    rel_x = int(x - min_col)
                    rel_y = int(y - min_row)
                    if (0 <= rel_y < blob_crop.shape[0] and 
                        0 <= rel_x < blob_crop.shape[1]):
                        if blob_mask[int(y), int(x)] > 0:
                            count += 1
            
            # Save blob
            blob_filename = f"blob_{Path(image_path).stem}_{blob_id}.jpg"
            blob_path = output_dir / blob_filename
            cv2.imwrite(str(blob_path), blob_crop)
            
            blob_data.append({
                'image': str(blob_path),
                'count': count,
                'bbox': [min_row, max_row, min_col, max_col]
            })
        
        return blob_data


class InteractiveMaskEditor:
    """
    Interactive tool to manually create/edit segmentation masks
    """
    
    def __init__(self, image_path, initial_mask=None):
        self.image = cv2.imread(str(image_path))
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        
        if initial_mask is not None:
            if isinstance(initial_mask, str):
                self.mask = cv2.imread(initial_mask, 0)
            else:
                self.mask = initial_mask.copy()
        else:
            self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        
        self.drawing = False
        self.brush_size = 15
        self.mode = 'draw'  # 'draw' or 'erase'
    
    def edit_mask(self, output_path):
        """
        Interactive mask editing
        
        Controls:
        - Left mouse: Draw/Erase
        - 'd': Switch to draw mode
        - 'e': Switch to erase mode
        - '+/-': Change brush size
        - 's': Save
        - 'q': Quit
        """
        clone = self.image.copy()
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                if self.mode == 'draw':
                    cv2.circle(self.mask, (x, y), self.brush_size, 255, -1)
                else:  # erase
                    cv2.circle(self.mask, (x, y), self.brush_size, 0, -1)
        
        cv2.namedWindow('Mask Editor')
        cv2.setMouseCallback('Mask Editor', mouse_callback)
        
        while True:
            # Create overlay
            overlay = self.image_rgb.copy()
            overlay[self.mask > 0] = [0, 255, 0]
            display = cv2.addWeighted(self.image_rgb, 0.6, overlay, 0.4, 0)
            
            # Add status text
            mode_text = f"Mode: {self.mode.upper()}, Brush: {self.brush_size}"
            cv2.putText(display, mode_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Mask Editor', cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('d'):
                self.mode = 'draw'
            elif key == ord('e'):
                self.mode = 'erase'
            elif key == ord('+') or key == ord('='):
                self.brush_size = min(50, self.brush_size + 2)
            elif key == ord('-'):
                self.brush_size = max(5, self.brush_size - 2)
            elif key == ord('s'):
                cv2.imwrite(str(output_path), self.mask)
                print(f"Mask saved to {output_path}")
            elif key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        return self.mask


def batch_create_masks(image_dir, annotation_dir, output_dir, method='circle'):
    """
    Create segmentation masks for all images in batch
    
    Args:
        image_dir: Directory with images
        annotation_dir: Directory with JSON annotations
        output_dir: Where to save masks
        method: 'circle', 'watershed', 'grabcut', or 'color'
    """
    image_dir = Path(image_dir)
    annotation_dir = Path(annotation_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    creator = SegmentationMaskCreator()
    
    for ann_file in annotation_dir.glob('*.json'):
        img_file = image_dir / f"{ann_file.stem}.jpg"
        if not img_file.exists():
            img_file = image_dir / f"{ann_file.stem}.png"
        
        if not img_file.exists():
            print(f"Warning: No image found for {ann_file.name}")
            continue
        
        mask_file = output_dir / f"{ann_file.stem}.png"
        
        print(f"Creating mask for {ann_file.stem}...", end=' ')
        
        try:
            if method == 'circle':
                creator.create_circle_masks(ann_file, mask_file)
            elif method == 'watershed':
                creator.create_watershed_masks(img_file, ann_file, mask_file)
            elif method == 'grabcut':
                creator.create_grabcut_masks(img_file, ann_file, mask_file)
            elif method == 'color':
                creator.create_color_based_masks(img_file, ann_file, mask_file)
            
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print(f"\nCreated masks in {output_dir}")


def visualize_mask_quality(image_path, mask_path, annotation_file, save_path=None):
    """Visualize segmentation mask quality"""
    
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(str(mask_path), 0)
    
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image with points
    axes[0].imshow(image)
    for x, y in data['points']:
        axes[0].plot(x, y, 'r+', markersize=10, markeredgewidth=2)
    axes[0].set_title(f"Original ({len(data['points'])} olives)")
    axes[0].axis('off')
    
    # Mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title("Segmentation Mask")
    axes[1].axis('off')
    
    # Overlay
    overlay = image.copy()
    overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    for x, y in data['points']:
        cv2.circle(overlay, (int(x), int(y)), 5, (255, 0, 0), -1)
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (Points = Red, Mask = Green)")
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    print("="*60)
    print("SEGMENTATION MASK CREATION")
    print("="*60)
    print("\nExample usage:")
    print("\n1. Create masks from annotations:")
    print("   batch_create_masks('images/', 'annotations/', 'masks/', method='watershed')")
    print("\n2. Create blob dataset for CNN training:")
    print("   BlobDatasetCreator.extract_blobs_from_mask(...)")
    print("\n3. Interactive editing:")
    print("   editor = InteractiveMaskEditor('image.jpg')")
    print("   editor.edit_mask('mask.png')")
