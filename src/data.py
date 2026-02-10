# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 10:47:38 2025

@author: Duminil

Data management and dataloaders

Classes and Functions:
----------------------
CustomImageDataset:
    A Dataset class that loads images on-the-fly, avoiding full storage in RAM.
    Supports datasets with or without associated labels.

raw_to_data:
    Converts raw files into numpy arrays.

split_data:
    Splits the training dataset into training, validation and test sets to evaluate model performance.

get_transforms / augmentations:
    Provides image transformations including normalization and optional data augmentations.

get_dataloaders:
    Constructs and returns dataloaders required for training, validation and test
    with annotated labels.
    
get_test_dataloader:
    Constructs and returns dataloader required for the exercice evaluation (without labels).

"""

import numpy as np
from sklearn.model_selection import train_test_split
import  torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

class CustomImageDataset(Dataset):
   def __init__(self, root_dir, transform=None, augment=None):
        
     self.root_dir = root_dir
     self.transform = transform
       
     self.image_paths = []
     self.labels = []
       
     class_names = sorted(os.listdir(root_dir))
     self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
       
     for cls_name in class_names:
           cls_dir = os.path.join(root_dir, cls_name)
           
           for fname in os.listdir(cls_dir):
               
               if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                   self.image_paths.append(os.path.join(cls_dir, fname))
                   self.labels.append(self.class_to_idx[cls_name])
       
       
   def __len__(self):
       return len(self.image_paths)
   
   def __getitem__(self, idx):
       img_path = self.image_paths[idx]
       label = self.labels[idx]
       
       image = Image.open(img_path).convert('RGB')
       if self.transform:
           image = self.transform(image)
       
       return image, label
    

def split_data(images, labels):
    
    # Split train_db into train, val test sets (70% 20% 10%)
    # add random state for reproductibility
    X_train, X_temp, y_train, y_temp = train_test_split(
    images, labels, test_size=0.3, random_state=42, stratify=labels)

    X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=1/3, random_state=42, stratify=y_temp)

    
    return  X_train, X_val, X_test, y_train, y_val, y_test

def transforms_norm():
    
    return transforms.Compose([
        transforms.ToTensor(),                
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])

def augmentations():
    
    return transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(30),
])


def get_dataloaders(data_path, batch_size, image_size):
    
    train_dataset = CustomImageDataset(os.path.join(data_path, 'Train/' ), transform=transforms_norm())
    val_dataset  = CustomImageDataset(os.path.join(data_path, 'Validation/' ), transform=transforms_norm())
    test_dataset = CustomImageDataset(os.path.join(data_path, 'Test/' ), transform=transforms_norm())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Train: {len(train_loader.dataset)} images | \
          Val: {len(val_loader.dataset)} images | \
          Test: {len(test_loader.dataset)} images")

    
    return train_loader, val_loader, test_loader
