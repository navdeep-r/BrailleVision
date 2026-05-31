"""
training/cell_dataset.py

PyTorch Dataset and DataLoader utilities for the Braille cell CNN classifier.

Crops are 64×64 grayscale PNGs. During training we apply aggressive augmentation
because cells are tiny and generalisation is hard. Weighted sampling ensures
rare Braille patterns (z, x, q) are not starved vs common ones (e, t, a).
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transforms(split: str) -> transforms.Compose:
    """
    Return appropriate transforms for the given split.
    Training uses aggressive augmentation; val/test normalise only.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split == "train":
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=3),  # 3ch for pretrained backbone
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            normalize,
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            normalize,
        ])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BrailleCellDataset(Dataset):
    """
    Loads cell crops from a CSV file.

    CSV columns: filename, pattern_int, character, source
    Image files live in image_dir/{filename}.
    """

    def __init__(
        self,
        csv_path: str,
        image_dir: str,
        transform: Optional[transforms.Compose] = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.transform = transform

        # Build a zero tensor for missing files
        self._zero_tensor = torch.zeros(3, 64, 64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])
        label = int(row["pattern_int"])

        # Load with OpenCV → PIL for torchvision transforms
        img_bgr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img_bgr is None:
            # Return zeros + label 0 for missing files (should not happen in practice)
            return (self._zero_tensor.clone(), 0)

        pil_img = Image.fromarray(img_bgr)

        if self.transform is not None:
            tensor = self.transform(pil_img)
        else:
            tensor = transforms.ToTensor()(pil_img)

        return (tensor, label)


# ---------------------------------------------------------------------------
# Weighted sampler
# ---------------------------------------------------------------------------

def build_weighted_sampler(csv_path: str) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler that over-samples rare Braille patterns.

    Strategy: sample_weight[i] = 1 / count_of_class_i
    This makes z, x, q appear as frequently as e, t, a during training.
    """
    df = pd.read_csv(csv_path)
    class_counts = df["pattern_int"].value_counts().to_dict()

    weights = [
        1.0 / class_counts[int(row["pattern_int"])]
        for _, row in df.iterrows()
    ]
    weights_tensor = torch.tensor(weights, dtype=torch.double)

    return WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=len(weights_tensor),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------------------------

def build_dataloaders(crops_dir: str, batch_size: int = 64) -> dict:
    """
    Build DataLoaders for all three splits.
    Train uses WeightedRandomSampler; val/test use sequential order.
    """
    loaders = {}
    for split in ["train", "val", "test"]:
        csv_path = os.path.join(crops_dir, f"{split}_labels.csv")
        img_dir  = os.path.join(crops_dir, split)

        if not os.path.isfile(csv_path):
            print(f"  [WARN] CSV not found: {csv_path}  — skipping {split}")
            loaders[split] = None
            continue

        transform = get_transforms(split)
        dataset   = BrailleCellDataset(csv_path, img_dir, transform)

        if split == "train":
            sampler = build_weighted_sampler(csv_path)
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
            )
        else:
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

    return loaders
