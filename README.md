## Face image classification with a CNN

This project aims to classify real and fake faces from images. 
The goal is to build a binary classifier capable of identifying fake (class 1) from real (class 0) images.

## Installation

pip install -r requirements.txt

# Training
python src/train.py --epochs 50 --batch_size 64
# Test
python src/test.py --model models/best_model.pth

## Some results
| models | Precision | Recall | HTER |
| CNN v1   | 0.765   | 0.913   | 0.172    |
| CNN v2   | 0.749   | 0.918    | 0.178    |
| ResNet v1| 0.841   | 0.967    | 0.099    |
| ResNet v2| 0.795   | 0.966   | 0.124    |

First tests :
CNN v1 : From scratch with crossentropy loss
CNN v2 : From scratch with focal loss
ResNet v1 : Pretrained Resnet18 with crossentropy loss
ResNet v2 : Pretrained Resnet18 crossentropy loss