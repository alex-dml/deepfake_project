# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 18:37:26 2025

@author: Duminil

Model testing

This file loads the best saved model, evaluates its performance on the test set,
and saves the resulting predictions into a text file.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from data import get_dataloaders
from model import FaceClassification
from evaluation import evaluate, predictions
from utils import plot_curves, balanced_loss
import numpy as np
from torchvision import models


def test(model_path, test_loader, in_channels, num_classes, device):
    
    # Load model
    # model = FaceClassification(in_channels, num_classes).to(device)
    
    model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    model = model.to(device)
    
    
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    
    # Test on intermediate annotated test set
    precision, recall, hter = evaluate(model, test_loader, device)
        
    print(f"On the annotated test set : P: {precision}, R: {recall}, HTER: {hter}")
    

    # Test on db_val.raw
    # preds = predictions(model, test_loader_without_labels, device)
    # np.savetxt("../results/pred_val.txt", preds, fmt="%d")


def main():
    
    # Argument Parser
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--model', default='../models/best_model_resnet_crossentr_unbalanced.pth', 
                    type=str, help='test dataset')
    parser.add_argument('--test_path', default='D:/0-Works/02_DATA/deepfake_dataset/Light_Dataset_unbalanced/', 
                        type=str, help='test dataset')
    parser.add_argument('--labels_path', default='../data/label_train.txt', 
                        type=str, help='Labels')
    parser.add_argument('--img_size', type=float, default=(56, 56, 3), help='Image size')
    parser.add_argument('--bs', type=int, default=1, help='Batch size')

    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    

    # Test loader with annotated labels
    _, _, test_loader = get_dataloaders(data_path=args.test_path, 
                                        batch_size=args.bs,
                                        image_size=args.img_size)
    
    test(model_path=args.model, 
         test_loader=test_loader, 
         in_channels=args.img_size[2], 
         num_classes=2, 
         device=device)
    

if __name__ == "__main__":
    main()
