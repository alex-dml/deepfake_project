# -*- coding: utf-8 -*-
"""
Created on Thu Dec 11 15:22:10 2025

@author: alxdn
"""

import os
import random
import shutil

def sample_images(src_dir, dst_dir, n=10000):
    os.makedirs(dst_dir, exist_ok=True)
    
    images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]
    sampled = random.sample(images, min(n, len(images)))

    for img in sampled:
        shutil.copy(os.path.join(src_dir, img), os.path.join(dst_dir, img))

base = "D:/0-Works/02_DATA/deepfake_dataset/"
sample_images(os.path.join(base, "Dataset/Train/Real"), os.path.join(base, "Light_Dataset/Train/Real"), n=10000)
sample_images(os.path.join(base, "Dataset/Train/Fake"), os.path.join(base, "Light_Dataset/Train/Fake"), n=10000)

sample_images(os.path.join(base, "Dataset/Validation/Real"), os.path.join(base, "Light_Dataset/Validation/Real"), n=2800)
sample_images(os.path.join(base, "Dataset/Validation/Fake"), os.path.join(base, "Light_Dataset/Validation/Fake"), n=2800)

sample_images(os.path.join(base, "Dataset/Test/Real"), os.path.join(base, "Light_Dataset/Test/Real"), n=1000)
sample_images(os.path.join(base, "Dataset/Test/Fake"), os.path.join(base, "Light_Dataset/Test/Fake"), n=1000)