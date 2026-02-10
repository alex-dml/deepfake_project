# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:47:03 2025

@author: alxdn

Utilitary functions

Functions
---------
plot_curves:
    Provides a plot of training and validation losses over epochs.
    
plot_images_with_labels:
    Provides a vizualisation of a sample of images from the dataset along with their labels.
    
get_hter_metric:
    Calculates the HTER metric from the confusion matrix to evaluate model performance.

balanced_loss:
    Adjusts the loss function to give higher weight to the minority class 
    (class 0) and mitigate class imbalance during training.

"""
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import random
import numpy as np

def plot_curves(train_loss, val_loss):
    
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss,label='val_loss')
    plt.title("Training and validation loss")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
def plot_images_with_labels(images, labels, num_images):
    
    plt.figure(figsize=(10, 10))
    for i in range(9):
        rd = random.randint(0, num_images) 
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[rd]).astype("uint8"))
        plt.title(int(labels[rd]))
        plt.axis("off")
    plt.show()
    
def get_hter_metric(labels, preds):
    
    # Avoid division by 0
    epsilon = 1e-8
    
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    fnr = fn / (fn + tn + epsilon)
    fpr = fp / (fp + tp + epsilon)
    hter = (fnr + fpr) / 2
    
    return hter

def balanced_loss(y_train, device):
    
    # minority class management class 0
    # number of 0 and 1
    y_train = torch.tensor(y_train, dtype=torch.long)
    # Count number of image for each class
    class_counts = torch.bincount(y_train)
    
    total = class_counts.sum().float()
    # Give more weight to class 0
    weights = total / class_counts.float() 
    # Normalisation
    weights = weights / weights.sum()
    weights = weights.to(device)
    
    return weights

class FocalLossWithAlpha(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLossWithAlpha, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = torch.nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()