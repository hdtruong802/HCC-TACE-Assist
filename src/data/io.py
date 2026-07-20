"""Khám phá cặp volume/segmentation + đọc NIfTI cho LiTS."""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import nibabel as nib
import numpy as np


@dataclass
class VolumePair:
    patient_id: str
    volume_path: str
    seg_path: str


def _vid(path: str) -> str:
    """Lấy id số từ tên file volume-<id>.nii / segmentation-<id>.nii."""
    m = re.search(r"(?:volume|segmentation)-(\d+)\.nii", os.path.basename(path))
    return m.group(1) if m else os.path.basename(path)


def discover_pairs(
    data_root: str,
    volume_glob: str = "**/volume-*.nii*",
    seg_glob: str = "**/segmentation-*.nii*",
) -> list[VolumePair]:
    """Ghép volume với segmentation theo id (robust với cấu trúc thư mục)."""
    vols = {_vid(p): p for p in glob.glob(os.path.join(data_root, volume_glob), recursive=True)}
    segs = {_vid(p): p for p in glob.glob(os.path.join(data_root, seg_glob), recursive=True)}
    pairs: list[VolumePair] = []
    for vid in sorted(vols, key=lambda x: int(x) if x.isdigit() else 10**9):
        if vid in segs:
            pairs.append(
                VolumePair(patient_id=f"lits-{vid}", volume_path=vols[vid], seg_path=segs[vid])
            )
    return pairs


def load_volume(path: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Trả về (array[H,W,Z] float32 HU, spacing=(x,y,z) mm). Chuẩn orientation RAS."""
    img = nib.as_closest_canonical(nib.load(path))
    arr = np.asarray(img.dataobj, dtype=np.float32)
    zooms = img.header.get_zooms()[:3]
    return arr, tuple(float(z) for z in zooms)


def load_pair(pair: VolumePair):
    """Trả về (vol float32 HU, seg int16, spacing). Kiểm shape khớp."""
    vol, spacing = load_volume(pair.volume_path)
    seg, _ = load_volume(pair.seg_path)
    seg = np.rint(seg).astype(np.int16)
    if vol.shape != seg.shape:
        raise ValueError(f"shape mismatch: vol{vol.shape} != seg{seg.shape}")
    return vol, seg, spacing