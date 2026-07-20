"""Suy nhãn slice từ mask segmentation (label transfer).

LiTS labels: 0=nền, 1=gan, 2=u. Vùng gan = (seg==1) | (seg==2) (u nằm trong gan).
"""
from __future__ import annotations

import numpy as np

EXCLUDE = -1  # slice không đủ gan → loại khỏi tập


def slice_areas(seg_slice: np.ndarray, liver_label: int = 1, tumor_label: int = 2):
    """Trả về (liver_area_px, tumor_area_px) cho 1 slice."""
    liver_area = int(((seg_slice == liver_label) | (seg_slice == tumor_label)).sum())
    tumor_area = int((seg_slice == tumor_label).sum())
    return liver_area, tumor_area


def make_label(liver_area: int, tumor_area: int, tau_area: int = 10, tau_liver: int = 50) -> int:
    """1=positive (có u), 0=negative (gan bình thường), EXCLUDE=không đủ gan."""
    if liver_area < tau_liver:
        return EXCLUDE
    return 1 if tumor_area >= tau_area else 0