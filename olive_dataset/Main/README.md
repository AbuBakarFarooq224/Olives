# Olive Counting System - Modular Structure

## Overview
This code has been refactored from a single 785-line `main.py` into multiple organized modules for better maintainability.

## File Structure

```
main/
├── models.py          # Neural network architectures (~235 lines)
├── dataset.py         # Data loading classes (~145 lines)  
├── pipeline.py        # Training & prediction pipeline (~390 lines)
├── train.py           # Main training script (~70 lines)
└── main.py            # Import wrapper for backward compatibility (~45 lines)
```

## Module Descriptions

### `models.py`
Contains the three neural network models:
- **FCNBlobDetector**: Fully Convolutional Network for segmentation (Stage 1)
- **CNNCountingNetwork**: CNN for counting olives in blobs (Stage 2)
- **LinearRegressionCorrection**: Linear regression for final count correction (Stage 3)

### `dataset.py`
Data loading utilities:
- **OliveSegmentationDataset**: PyTorch Dataset class for loading image-mask pairs
- **create_data_loaders()**: Function to create train/validation DataLoaders

### `pipeline.py`
Complete training and prediction pipeline:
- **OliveCountingPipeline**: Main class that orchestrates all 3 stages
  - `train_stage1_segmentation()`: Train FCN
  - `train_stage2_counting()`: Train CNN
  - `train_stage3_regression()`: Train linear regression
  - `predict()`: Run inference on new images

### `train.py`
Main execution script:
- Entry point for training
- Set up data directories
- Initialize pipeline
- Start training (uncomment code to begin)

### `main.py`
Lightweight import wrapper:
- Imports all components from other modules
- Provides backward compatibility
- Can be imported by other scripts

## How to Use

### Run Training
```bash
cd "C:\Users\Welcome\OneDrive\Desktop\FYP MATERIAL\patchify"
& ".venv\Scripts\python.exe" main\train.py
```

### Import in Your Code
```python
from main import OliveCountingPipeline, create_data_loaders

# Initialize
pipeline = OliveCountingPipeline()

# Create data loaders
train_loader, val_loader = create_data_loaders('patchify/', 'patchify_mask/')

# Train
pipeline.train_stage1_segmentation(train_loader, val_loader, epochs=30)
```

## Data Requirements

**Current Status:**
- ✅ Images: `patchify/` (909 PNG files)
- ✅ Masks: `patchify_mask/` (909 PNG files)
- ❌ JSON annotations: Not available (needed for Stage 2 & 3)

**What You Can Train:**
- **Stage 1 (Segmentation)**: ✅ Ready to train with existing images + masks
- **Stage 2 (Counting)**: ❌ Requires point annotations to create blob datasets
- **Stage 3 (Regression)**: ❌ Requires total olive counts for each image

## Benefits of Modular Structure

1. **Easier to Navigate**: Find specific functionality quickly
2. **Better Testing**: Test individual components separately
3. **Reusability**: Import specific classes without loading everything
4. **Maintainability**: Modify one module without affecting others
5. **Collaboration**: Multiple people can work on different files
6. **Cleaner Code**: Each file has a single, clear purpose

## File Sizes (Approximate)

- **Old**: `main.py` - 785 lines (everything in one file)
- **New**: 
  - `models.py` - 235 lines (neural networks)
  - `dataset.py` - 145 lines (data loading)
  - `pipeline.py` - 390 lines (training/inference)
  - `train.py` - 70 lines (execution)
  - `main.py` - 45 lines (imports)
  - **Total**: ~885 lines (includes README documentation)

## Next Steps

1. **Test Stage 1 Training**: Uncomment training code in [train.py](train.py)
2. **Monitor Progress**: Training will save `fcn_segmenter_best.pth` when improving
3. **Evaluate Results**: Check validation IoU scores during training
4. **Prepare for Stage 2**: Create point annotations if you want to train counting network

---

*Refactored: February 7, 2026*
