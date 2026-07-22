"""Dataset LLD-MMRI (Tầng 2): mỗi bệnh nhân = 1 mẫu, ghép các thì thành ĐA KÊNH.

Mỗi (bệnh nhân × thì) là 1 ROI trong `images_u8_lld.npy` (cột `row`). Dataset gom các thì
của 1 bệnh nhân theo THỨ TỰ cố định `phases` → tensor [K,H,W]; thì thiếu → kênh 0.
Nhãn = `category` (7 lớp). Split ở mức bệnh nhân (tái dùng load_split/fold_patients).
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class LldDataset(Dataset):
    def __init__(self, manifest_df, images_path: str, patients, phases, size: int, train: bool):
        df = manifest_df[manifest_df.patient_id.isin(patients)]
        self.phases = list(phases)
        self.size = size
        self.train = train
        self.arr = np.load(images_path, mmap_mode="r")
        self.items = []                                  # (pid, {phase:row}, label)
        for pid, g in df.groupby("patient_id"):
            ph2row = dict(zip(g.phase, g.row.astype(int)))
            self.items.append((pid, ph2row, int(g.category.iloc[0])))

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, row):
        return np.asarray(self.arr[int(row)], dtype=np.uint8)

    def __getitem__(self, i):
        pid, ph2row, label = self.items[i]
        chans = [self._load(ph2row[ph]) if ph in ph2row else np.zeros((self.size, self.size), np.uint8)
                 for ph in self.phases]
        x = np.stack(chans, 0).astype(np.float32) / 255.0     # [K,H,W]
        if self.train:                                        # aug nhẹ, nhất quán mọi kênh
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1]
            if np.random.rand() < 0.5:
                x = x[:, ::-1, :]
            k = np.random.randint(4)
            if k:
                x = np.rot90(x, k, axes=(1, 2))
        x = np.ascontiguousarray(x)
        x = (x - 0.5) / 0.5                                    # ~[-1,1]
        return torch.from_numpy(x), int(label), pid

    def class_counts(self, num_classes: int):
        c = np.zeros(num_classes, dtype=np.int64)
        for _, _, y in self.items:
            c[y] += 1
        return c
