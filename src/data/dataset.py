"""PyTorch Dataset cho LiTS packed cache (images_u8.npy) + patient-level split."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(path: str) -> dict:
    return json.load(open(path, encoding="utf-8"))


def fold_patients(split: dict, fold: int):
    """Trả về (train_patients, val_patients, test_patients).

    fold = -1 → train = toàn bộ trainval (mọi patient không thuộc test), val rỗng.
    """
    test = set(split["test"])
    if fold < 0:
        allp = set()
        for f in split["folds"]:
            allp.update(f["train"]); allp.update(f["val"])
        return allp, set(), test
    f = split["folds"][fold]
    return set(f["train"]), set(f["val"]), test


def build_transforms(size: int, train: bool):
    """Albumentations pipeline (input HWC uint8 → CHW float tensor)."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15,
                               border_mode=0, p=0.5),
            A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
            A.Resize(size, size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class LitsSliceDataset(Dataset):
    """Đọc slice từ mảng gộp memmap theo cột `row`; nở 3 kênh; label nhị phân."""

    def __init__(self, manifest_df: pd.DataFrame, images_path: str, patients, transform=None):
        self.df = manifest_df[manifest_df.patient_id.isin(patients)].reset_index(drop=True)
        self.arr = np.load(images_path, mmap_mode="r")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = np.asarray(self.arr[int(r.row)], dtype=np.uint8)      # [H,W]
        img = np.repeat(img[:, :, None], 3, axis=2)                 # HWC 3ch
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, torch.tensor(float(r.label)), str(r.patient_id)

    def pos_weight(self) -> float:
        pos = int((self.df.label == 1).sum())
        neg = int((self.df.label == 0).sum())
        return neg / max(1, pos)
