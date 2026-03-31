"""
LIDC-IDRI Dataset Loader for ProtoPFormer
==========================================
Expects preprocessed data in one of two formats:

FORMAT A — Folder structure (recommended):
    datasets/LIDC/
        train/
            benign/   *.png or *.jpg   (malignancy score 1-2)
            malignant/ *.png or *.jpg  (malignancy score 4-5)
        val/
            benign/
            malignant/
        test/
            benign/
            malignant/

FORMAT B — CSV manifest + image folder:
    datasets/LIDC/
        images/  <nodule_id>.png
        train.csv   columns: filename, label   (label: 0=benign, 1=malignant)
        val.csv
        test.csv

Set args.lidc_format = 'folder' or 'csv' accordingly.
"""

import os
import csv
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# ── CT window/level normalisation ─────────────────────────────────────────────
# Lung window: W=1500, L=-600  →  [-1350, 150] HU
# If your images are already saved as 8-bit PNGs you can skip windowing and
# just use standard ImageNet normalisation (mean/std below are fine).
LIDC_MEAN = [0.485, 0.456, 0.406]   # ImageNet — good starting point
LIDC_STD  = [0.229, 0.224, 0.225]

# swap these for CT-specific stats once you compute them on your dataset:
# LIDC_MEAN = [0.197, 0.197, 0.197]
# LIDC_STD  = [0.198, 0.198, 0.198]


def build_lidc_transforms(is_train: bool, img_size: int = 224):
    """
    Training: random flips + slight rotation + colour jitter (brightness/contrast
              only — no hue/saturation, not meaningful for CT).
    Val/Test: deterministic resize + centre crop.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=LIDC_MEAN, std=LIDC_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=LIDC_MEAN, std=LIDC_STD),
        ])


# ── Dataset classes ────────────────────────────────────────────────────────────

class LIDCFolderDataset(Dataset):
    """
    Reads from folder structure:  root/{split}/{benign|malignant}/*.png
    """
    CLASSES = ['benign', 'malignant']

    def __init__(self, root: str, split: str = 'train',
                 transform=None, img_size: int = 224):
        self.root      = root
        self.split     = split
        self.transform = transform or build_lidc_transforms(split == 'train', img_size)
        self.samples   = []   # list of (path, label)

        split_dir = os.path.join(root, split)
        for label_idx, cls_name in enumerate(self.CLASSES):
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(
                    f"Expected class folder not found: {cls_dir}\n"
                    "Check your dataset structure (see docstring at top of file)."
                )
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                    self.samples.append((os.path.join(cls_dir, fname), label_idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

    @property
    def class_counts(self):
        counts = {c: 0 for c in self.CLASSES}
        for _, lbl in self.samples:
            counts[self.CLASSES[lbl]] += 1
        return counts


class LIDCCSVDataset(Dataset):
    """
    Reads from a CSV manifest:
        filename,label
        nodule_001.png,0
        nodule_002.png,1
    Images are expected in  root/images/<filename>.
    """
    def __init__(self, root: str, split: str = 'train',
                 transform=None, img_size: int = 224):
        self.root      = root
        self.transform = transform or build_lidc_transforms(split == 'train', img_size)
        self.samples   = []

        csv_path = os.path.join(root, f'{split}.csv')
        img_dir  = os.path.join(root, 'images')

        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row['filename']
                label = int(row['label'])
                self.samples.append((os.path.join(img_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Public API called by main.py ───────────────────────────────────────────────

def build_dataset(is_train: bool, args):
    """
    Drop-in replacement for the original build_dataset in the CUB/Dogs repo.
    Returns (dataset, num_classes).
    """
    split = 'train' if is_train else 'val'

    fmt = getattr(args, 'lidc_format', 'folder')

    if fmt == 'folder':
        dataset = LIDCFolderDataset(
            root     = args.data_path,
            split    = split,
            img_size = args.img_size,
        )
    elif fmt == 'csv':
        dataset = LIDCCSVDataset(
            root     = args.data_path,
            split    = split,
            img_size = args.img_size,
        )
    else:
        raise ValueError(f"Unknown lidc_format='{fmt}'. Choose 'folder' or 'csv'.")

    num_classes = 2   # benign / malignant — always 2
    return dataset, num_classes
