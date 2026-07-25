"""
Neural Network Models for Olive Counting System

Contains the three main models:
1. FCNBlobDetector - Segmentation network (Stage 1)
2. CNNCountingNetwork - Counting network (Stage 2)
3. LinearRegressionCorrection - Regression model (Stage 3)
"""

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


class FCNBlobDetector(nn.Module):
    """
    Fully Convolutional Network for fruit blob segmentation
    Based on FCN-8s architecture from the paper
    
    Input: RGB image
    Output: Binary segmentation mask (fruit vs background)
    """
    
    def __init__(self):
        super(FCNBlobDetector, self).__init__()
        
        # Load pretrained VGG16
        vgg16 = models.vgg16(pretrained=True)
        features = list(vgg16.features.children())
        
        # Encoder (extract features at different scales)
        self.pool3 = nn.Sequential(*features[:17])   # 1/8 resolution
        self.pool4 = nn.Sequential(*features[17:24]) # 1/16 resolution  
        self.pool5 = nn.Sequential(*features[24:])   # 1/32 resolution
        
        # Fully convolutional classifier (replaces FC layers)
        self.fc6 = nn.Conv2d(512, 4096, 7, padding=3)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d(p=0.5)
        
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d(p=0.5)
        
        # Score layer (predict class at each pixel)
        self.score_fr = nn.Conv2d(4096, 2, 1)  # 2 classes: fruit/background
        
        # Score layers for skip connections
        self.score_pool3 = nn.Conv2d(256, 2, 1)
        self.score_pool4 = nn.Conv2d(512, 2, 1)
        
        # Upsampling (transpose convolution)
        self.upscore2 = nn.ConvTranspose2d(2, 2, 4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(2, 2, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(2, 2, 16, stride=8, bias=False)
        
        # Initialize upsampling with bilinear weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize upsampling layers with bilinear interpolation"""
        for m in [self.upscore2, self.upscore_pool4, self.upscore8]:
            c1, c2, h, w = m.weight.data.size()
            weight = self._get_bilinear_kernel(h)
            m.weight.data.copy_(weight)
    
    def _get_bilinear_kernel(self, size):
        """Generate bilinear interpolation kernel"""
        factor = (size + 1) // 2
        center = factor - 1 if size % 2 == 1 else factor - 0.5
        og = np.ogrid[:size, :size]
        kernel = (1 - abs(og[0] - center) / factor) * \
                 (1 - abs(og[1] - center) / factor)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        return kernel.repeat(2, 2, 1, 1)
    
    def forward(self, x):
        # Get features from different pool layers
        pool3 = self.pool3(x)
        pool4 = self.pool4(pool3)
        pool5 = self.pool5(pool4)
        
        # Fully convolutional layers
        fc6 = self.drop6(self.relu6(self.fc6(pool5)))
        fc7 = self.drop7(self.relu7(self.fc7(fc6)))
        
        # Score and upsample
        score_fr = self.score_fr(fc7)
        upscore2 = self.upscore2(score_fr)
        
        # Add skip connection from pool4
        score_pool4 = self.score_pool4(pool4)
        
        # Crop both to the minimum size
        min_h = min(score_pool4.size()[2], upscore2.size()[2])
        min_w = min(score_pool4.size()[3], upscore2.size()[3])
        
        # Center crop both tensors
        if score_pool4.size()[2] > min_h or score_pool4.size()[3] > min_w:
            diff_h = score_pool4.size()[2] - min_h
            diff_w = score_pool4.size()[3] - min_w
            score_pool4 = score_pool4[:, :, diff_h//2:diff_h//2+min_h, diff_w//2:diff_w//2+min_w]
        
        if upscore2.size()[2] > min_h or upscore2.size()[3] > min_w:
            diff_h = upscore2.size()[2] - min_h
            diff_w = upscore2.size()[3] - min_w
            upscore2 = upscore2[:, :, diff_h//2:diff_h//2+min_h, diff_w//2:diff_w//2+min_w]
        
        upscore_pool4 = self.upscore_pool4(score_pool4 + upscore2)
        
        # Add skip connection from pool3
        score_pool3 = self.score_pool3(pool3)
        
        # Crop both to the minimum size
        min_h = min(score_pool3.size()[2], upscore_pool4.size()[2])
        min_w = min(score_pool3.size()[3], upscore_pool4.size()[3])
        
        # Center crop both tensors
        if score_pool3.size()[2] > min_h or score_pool3.size()[3] > min_w:
            diff_h = score_pool3.size()[2] - min_h
            diff_w = score_pool3.size()[3] - min_w
            score_pool3 = score_pool3[:, :, diff_h//2:diff_h//2+min_h, diff_w//2:diff_w//2+min_w]
        
        if upscore_pool4.size()[2] > min_h or upscore_pool4.size()[3] > min_w:
            diff_h = upscore_pool4.size()[2] - min_h
            diff_w = upscore_pool4.size()[3] - min_w
            upscore_pool4 = upscore_pool4[:, :, diff_h//2:diff_h//2+min_h, diff_w//2:diff_w//2+min_w]
        
        # Final upsampling
        upscore8 = self.upscore8(score_pool3 + upscore_pool4)
        
        # Center crop to match input size
        if upscore8.size()[2] != x.size()[2] or upscore8.size()[3] != x.size()[3]:
            diff_h = upscore8.size()[2] - x.size()[2]
            diff_w = upscore8.size()[3] - x.size()[3]
            upscore8 = upscore8[:, :,
                               diff_h//2:diff_h//2+x.size()[2],
                               diff_w//2:diff_w//2+x.size()[3]]
        
        return upscore8


class CNNCountingNetwork(nn.Module):
    """
    CNN that counts number of fruits in a segmented blob
    
    Input: Image crop of a single blob
    Output: Integer count (0 to max_count)
    """
    
    def __init__(self, max_count=20):
        super(CNNCountingNetwork, self).__init__()
        
        self.max_count = max_count
        
        # Convolutional feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
        )
        
        # Classifier (outputs count as classification)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, max_count + 1)  # +1 for count=0
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class LinearRegressionCorrection:
    """
    Simple linear regression to map CNN count estimates to final counts
    Final_Count = a * CNN_Count + b
    """
    
    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self.trained = False
    
    def fit(self, cnn_counts, true_counts):
        """
        Train linear regression
        
        Args:
            cnn_counts: List of CNN predicted counts
            true_counts: List of ground truth counts
        """
        cnn_counts = np.array(cnn_counts).reshape(-1, 1)
        true_counts = np.array(true_counts)
        
        # Simple least squares: y = ax + b
        X = np.column_stack([cnn_counts, np.ones(len(cnn_counts))])
        theta = np.linalg.lstsq(X, true_counts, rcond=None)[0]
        
        self.a = theta[0]
        self.b = theta[1]
        self.trained = True
        
        print(f"Linear regression trained: y = {self.a:.4f}x + {self.b:.4f}")
    
    def predict(self, cnn_count):
        """Apply linear correction"""
        if not self.trained:
            return int(round(cnn_count))
        return int(round(self.a * cnn_count + self.b))
