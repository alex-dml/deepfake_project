# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 11:16:00 2025

@author: Duminil

Model training

This file provides functions to train a model, manage parameters. 
It includes the training loop, loss computation, and overall workflow 
management for model training.

Functions
---------
training:
    Implements the model training loop.
    Computes the loss and updates model weights for each batch.

main:
    Central function to manage the overall training workflow.
    Handles argument parsing and training management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from data import get_dataloaders
from model import FaceClassification
from evaluation import evaluate_val
from utils import plot_curves, balanced_loss, FocalLossWithAlpha
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torchvision import models

def training(model, epochs, train_loader, val_loader, optimizer, criterion, 
             patience, device):
    
    train_losses = []
    val_losses = []
    best_hter = 1
    best_val = 1
    cpt = 0
    
    for epoch in range(epochs):
        
        model.train()
        total_loss = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss/len(train_loader)
        train_losses.append(train_loss)
        
        # Evaluation
        val_loss, hter = evaluate_val(model, val_loader, epoch, criterion, device)
        val_losses.append(val_loss)
        
        # Model saving and early stopping depending on the HTER metric : 
        # if it does not evolve after a certain number of epochs, the training stopped.
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "../models/best_model_resnet_crossentr_unbalanced.pth")
            cpt=0
            print(f"Best model saved for (HTER={hter:.4f})")
            
        else:
            cpt +=1
            if cpt >= patience:
                print(f" Early stopping : No improvement since {patience} epochs")
                break
        
            
        print(f"Epoch {epoch+1}/{epochs}", 
              f"train_loss: {train_loss:.4f}",
              f"val_loss: {val_loss:.4f}",
              f"HTER: {hter:.4f}"
              )
        
    # Plot and save training and validation loss curves after training
    plot_curves(train_losses, val_losses)
    

def main():
    
    # Argument Parser
    parser = argparse.ArgumentParser(description='training')
    
    parser.add_argument('--data_path', default='D:/0-Works/02_DATA/deepfake_dataset/Light_Dataset_unbalanced/', 
                        type=str, help='Training dataset')
    parser.add_argument('--img_size', type=float, default=(56, 56, 3), help='Image size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--bs', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader, test_loader = get_dataloaders(data_path=args.data_path,     
                                                            batch_size=args.bs,
                                                            image_size=args.img_size)
    
    # model_ft = FaceClassification(in_channels=args.img_size[2], num_classes=2).to(device)
    
    model_ft = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)


    # criterion = FocalLossWithAlpha()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_ft.parameters(), lr=args.lr)
    
    training(model_ft, args.epochs, train_loader, val_loader, optimizer, 
             criterion, 5, device)
    


if __name__ == "__main__":
    main()