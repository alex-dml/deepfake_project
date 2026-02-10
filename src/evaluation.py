# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 12:20:09 2025

@author: Duminil

Evaluation on the validation set and predictions

Functions
--------------------
evaluate_val:
    Evaluates the trained model on the validation set.
    Returns the HTER metric to quantify performance, and confusion matrix every 10 epochs.
    
evaluate:
    Evaluates the annotated tests set.
    Returns HTER, Precision, Recall metrics.

predictions:
    Generates predictions on the test set using the trained model.
    Returns predicted label.
"""
import torch, os
from utils import get_hter_metric
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from matplotlib import pyplot as plt
import numpy as np

def evaluate_val(model, loader, num_epoch, criterion, device):
    
    model.eval()
    val_loss = 0.0
    res_preds = []
    res_labels = []
    
    # Avoid calculating gradient for validation
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            val_loss += loss.item()
            
            _ , preds = torch.max(outputs, 1)
            res_preds.extend(preds)
            res_labels.extend(target)
            
    res_preds = torch.tensor(res_preds)
    res_labels = torch.tensor(res_labels)  
    
    # HTER metric
    hter = get_hter_metric(res_labels, res_preds)
    
    # Plot confusion matrix every 10 epochs
    if num_epoch%10 == 0:
        cm = confusion_matrix(res_labels, res_preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized)
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix - Epoch {num_epoch}')
        plt.savefig(os.path.join('../results/', f'cm_epoch_{num_epoch}.png'))
        plt.close()
    
    return val_loss/len(loader), hter

def evaluate(model, loader, device):
    
    model.eval()
    res_preds = []
    res_labels = []
    
    # Avoid calculating gradient for validation
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            
            _ , preds = torch.max(outputs, 1)
            res_preds.extend(preds)
            res_labels.extend(target)
            
    res_preds = torch.tensor(res_preds)
    res_labels = torch.tensor(res_labels)  
    
    # HTER metric
    hter = get_hter_metric(res_labels, res_preds)
    
    # Precision and Recall scores
    precision = precision_score(res_labels, res_preds, average='binary')
    recall = recall_score(res_labels, res_preds, average='binary')

    return precision, recall, hter

def predictions(model, loader, device):
    
    model.eval()
    res_preds = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            outputs = model(data)
            _ , preds = torch.max(outputs, 1)
            res_preds.extend(preds.cpu().numpy())
    
    return res_preds