"""
Olive Counting System - Main Module
Based on Chen et al. (2017): "Counting Apples and Oranges with Deep Learning"

This file imports all components from separate modules:
- models.py: Neural network architectures
- dataset.py: Data loading utilities
- pipeline.py: Complete training and prediction pipeline
- train.py: Main training script

For backward compatibility, all classes are imported here.
To run training, use: python train.py
"""

# Import all components
from models import FCNBlobDetector, CNNCountingNetwork, LinearRegressionCorrection
from dataset import OliveSegmentationDataset, create_data_loaders
from pipeline import OliveCountingPipeline

# For convenience, export main classes
__all__ = [
    'FCNBlobDetector',
    'CNNCountingNetwork', 
    'LinearRegressionCorrection',
    'OliveSegmentationDataset',
    'create_data_loaders',
    'OliveCountingPipeline'
]


if __name__ == "__main__":
    print("="*60)
    print("OLIVE COUNTING SYSTEM")
    print("="*60)
    print("\nThis is the main module file.")
    print("All components have been organized into separate files:")
    print("  - models.py: Neural network architectures")
    print("  - dataset.py: Data loading utilities")
    print("  - pipeline.py: Training and prediction pipeline")
    print("  - train.py: Main training script")
    print("\nTo start training, run:")
    print("  python train.py")
    print("="*60)
