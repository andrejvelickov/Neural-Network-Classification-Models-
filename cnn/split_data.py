import os
import shutil
import random
from pathlib import Path

# Set seed for reproducibility
random.seed(123)

# Source and destination paths
source_dir = './data/original'
dest_dir = './data/Tennis Player Actions Dataset for Human Pose Estimation'

# Ratios for splitting
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# Make sure destination folders exist
for split in ['train', 'val', 'test']:
    for class_name in os.listdir(source_dir):
        os.makedirs(os.path.join(dest_dir, split, class_name), exist_ok=True)

# Process each class
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    total = len(images)
    train_cutoff = int(total * train_ratio)
    val_cutoff = int(total * (train_ratio + val_ratio))

    train_imgs = images[:train_cutoff]
    val_imgs = images[train_cutoff:val_cutoff]
    test_imgs = images[val_cutoff:]

    for img_list, split in [(train_imgs, 'train'), (val_imgs, 'val'), (test_imgs, 'test')]:
        for img in img_list:
            src_path = os.path.join(class_path, img)
            dst_path = os.path.join(dest_dir, split, class_name, img)
            shutil.copy2(src_path, dst_path)

print("✅ Dataset successfully split into train/val/test.")
