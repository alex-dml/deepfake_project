# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 11:15:46 2025

@author: Duminil

Model architecture

This module defines the neural network architecture for face classification,
including convolutional, activation and any regularization layers needed.

FaceClassification:
    Neural network with a CNN based architecture for face classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

    
class FaceClassification(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            
            nn.Conv2d(in_channels, 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(0.2),
        
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(0.2),
            
            nn.Flatten(),
            nn.Linear(262144,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout1d(0.2),
            nn.Linear(64,num_classes)
        )
        

    def forward(self, x):
        return self.network(x)
