"""Đọc 3D-IRCADb-01 (DICOM) → volume HU + seg 0/1/2 giống LiTS, để tái dùng nguyên
pipeline W1 (window → liver-crop → resize → label-transfer).

Cấu trúc chuẩn mỗi bệnh nhân (dò đệ quy, bền với nesting):
    3Dircadb1.<n>/
        PATIENT_DICOM/            # ảnh CT (DICOM, thường không đuôi)
        MASKS_DICOM/
            liver/               # mask gan
            livertumor01/ ...    # 0..n ổ u (bệnh nhân KHÔNG có folder này = ca ÂM)
            <các cấu trúc khác>  # bỏ qua

Chuẩn hoá về encoding LiTS: seg = 0 nền · 1 gan · 2 u (u đè lên gan) → dùng lại
`liver_bbox`, `slice_areas`, `make_label` không đổi.

Lưu ý (đưa vào limitations): xử lý slice ở orientation gốc của DICOM (không reorient
RAS như LiTS). Slice-level classification độc lập theo lát nên thứ tự z không ảnh hưởng;
rủi ro còn lại là lật trên/dưới ảnh — model có HFlip-aug (che trái/phải) nhưng không có
VFlip. Nếu external tệ bất thường, kiểm tra lại orientation trước khi kết luận.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass

import numpy as np


@dataclass
class IrcadPatient:
    patient_id: str      # 'ircad-<n>'
    root: str            # thư mục 3Dircadb1.<n>


def _num(name: str) -> int:
    m = re.findall(r"\d+", name)
    return int(m[-1]) if m else 10 ** 9


def _unwrap(d: str, depth: int = 5) -> str:
    """Bóc tầng lồng thừa cùng tên (vd PATIENT_DICOM/PATIENT_DICOM/…) tới nơi có nội dung thật."""
    cur = d
    for _ in range(depth):
        try:
            entries = os.listdir(cur)
        except OSError:
            break
        if len(entries) == 1 and os.path.isdir(os.path.join(cur, entries[0])):
            cur = os.path.join(cur, entries[0])          # bọc 1 thư mục con → đi xuống
            continue
        break
    return cur


def discover_patients(data_root: str) -> list[IrcadPatient]:
    """Bệnh nhân = thư mục có CẢ PATIENT_DICOM lẫn MASKS_DICOM (loại trùng do nesting)."""
    roots: dict[str, str] = {}
    for p in glob.glob(os.path.join(data_root, "**", "PATIENT_DICOM"), recursive=True):
        parent = os.path.dirname(p)
        if os.path.isdir(os.path.join(parent, "MASKS_DICOM")):
            roots.setdefault(os.path.abspath(parent), os.path.basename(parent))
    out = [IrcadPatient(patient_id=f"ircad-{_num(name)}", root=root)
           for root, name in roots.items()]
    return sorted(out, key=lambda p: _num(p.patient_id))


def _read_series(folder: str):
    """Đọc 1 series DICOM → list dataset đã sort theo InstanceNumber (fallback tên file)."""
    import pydicom

    files = [f for f in glob.glob(os.path.join(folder, "*")) if os.path.isfile(f)]
    ds = []
    for f in files:
        try:
            d = pydicom.dcmread(f, force=True)
            _ = d.pixel_array                        # bỏ file không có ảnh
        except Exception:  # noqa: BLE001
            continue
        ds.append(d)
    if ds and all(getattr(d, "InstanceNumber", None) is not None for d in ds):
        ds.sort(key=lambda d: int(d.InstanceNumber))
    else:
        ds.sort(key=lambda d: _num(os.path.basename(getattr(d, "filename", ""))))
    return ds


def _hu(d) -> np.ndarray:
    slope = float(getattr(d, "RescaleSlope", 1.0) or 1.0)
    inter = float(getattr(d, "RescaleIntercept", 0.0) or 0.0)
    return d.pixel_array.astype(np.float32) * slope + inter


def _read_ct(folder: str):
    ds = _read_series(folder)
    if not ds:
        raise RuntimeError(f"Không đọc được DICOM nào trong {folder}")
    vol = np.stack([_hu(d) for d in ds], axis=-1)    # [H,W,Z] HU
    inst = [int(getattr(d, "InstanceNumber", i)) for i, d in enumerate(ds)]
    px = getattr(ds[0], "PixelSpacing", [1.0, 1.0])
    th = float(getattr(ds[0], "SliceThickness", 1.0) or 1.0)
    spacing = (float(px[0]), float(px[1]), th)
    return vol, inst, spacing


def _read_mask_aligned(folder: str, inst_order, shape_hw) -> np.ndarray:
    """Mask nhị phân [H,W,Z] khớp thứ tự lát của CT (theo InstanceNumber; fallback theo index)."""
    ds = _read_series(folder)
    Z = len(inst_order)
    m = np.zeros((shape_hw[0], shape_hw[1], Z), dtype=bool)
    if not ds:
        return m
    if all(getattr(d, "InstanceNumber", None) is not None for d in ds):
        by_inst = {int(d.InstanceNumber): (d.pixel_array > 0) for d in ds}
        for zi, inst in enumerate(inst_order):
            if inst in by_inst:
                m[:, :, zi] = by_inst[inst]
    else:                                             # cùng số lát → khớp theo vị trí
        for zi in range(min(Z, len(ds))):
            m[:, :, zi] = ds[zi].pixel_array > 0
    return m


def _mask_dirs(patient_root: str):
    """Trả (liver_dir | None, [tumor_dirs]) từ MASKS_DICOM (đã bóc tầng lồng thừa)."""
    md = _unwrap(os.path.join(patient_root, "MASKS_DICOM"))
    if not os.path.isdir(md):
        return None, []
    liver_dir, tumor_dirs = None, []
    for name in sorted(os.listdir(md)):
        full = os.path.join(md, name)
        if not os.path.isdir(full):
            continue
        low = name.lower()
        if low == "liver":
            liver_dir = _unwrap(full)
        elif low.startswith("livertumor") or ("tumor" in low and "liver" in low):
            tumor_dirs.append(_unwrap(full))
    return liver_dir, tumor_dirs


def load_patient(p: IrcadPatient):
    """Trả (vol[H,W,Z] HU, seg[H,W,Z] {0,1,2}, spacing, has_tumor_folder)."""
    vol, inst, spacing = _read_ct(_unwrap(os.path.join(p.root, "PATIENT_DICOM")))
    H, W, _ = vol.shape
    liver_dir, tumor_dirs = _mask_dirs(p.root)
    liver = _read_mask_aligned(liver_dir, inst, (H, W)) if liver_dir else np.zeros_like(vol, bool)
    tumor = np.zeros_like(vol, bool)
    for td in tumor_dirs:
        tumor |= _read_mask_aligned(td, inst, (H, W))
    seg = np.zeros(vol.shape, dtype=np.int16)
    seg[liver] = 1
    seg[tumor] = 2                                    # u đè lên gan (như LiTS)
    return vol, seg, spacing, bool(tumor_dirs)
