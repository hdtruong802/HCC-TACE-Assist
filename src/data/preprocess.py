"""Windowing CT, liver-ROI crop, resize → uint8 cache (1 kênh)."""
from __future__ import annotations

import cv2
import numpy as np


def window_ct(hu: np.ndarray, wl: float = 50, ww: float = 350) -> np.ndarray:
    """Áp cửa sổ gan → clip HU → scale [0,255] uint8."""
    lo, hi = wl - ww / 2.0, wl + ww / 2.0
    x = np.clip(hu, lo, hi)
    x = (x - lo) / (hi - lo)  # [0,1]
    return (x * 255.0).astype(np.uint8)


def liver_bbox(seg: np.ndarray, liver_label: int = 1, tumor_label: int = 2, margin: int = 16):
    """Bbox in-plane bao gan(+u) trên TOÀN volume → crop nhất quán. seg: [H,W,Z]."""
    mask = np.isin(seg, [liver_label, tumor_label]).any(axis=2)
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    H, W = mask.shape
    y0 = max(0, int(ys.min()) - margin)
    x0 = max(0, int(xs.min()) - margin)
    y1 = min(H, int(ys.max()) + 1 + margin)
    x1 = min(W, int(xs.max()) + 1 + margin)
    return y0, y1, x0, x1


def crop_resize(slice_u8: np.ndarray, bbox, size: int = 256) -> np.ndarray:
    """Crop theo bbox (nếu có) rồi resize về size×size."""
    if bbox is not None:
        y0, y1, x0, x1 = bbox
        slice_u8 = slice_u8[y0:y1, x0:x1]
    if slice_u8.shape[0] == 0 or slice_u8.shape[1] == 0:
        slice_u8 = np.zeros((size, size), dtype=np.uint8)
    return cv2.resize(slice_u8, (size, size), interpolation=cv2.INTER_AREA)