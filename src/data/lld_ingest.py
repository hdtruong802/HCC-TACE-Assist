"""Đọc LLD-MMRI (MRI đa thì) → ROI tổn thương theo bbox + nhãn 7 lớp, cho Tầng 2.

Annotation (`LLD_MMRI_Annotation.json`):
    { "Category_info": {tên_lớp: id, "Benign":[...], "Malignant":[...]},
      "Annotation_info": { "MR-xxxxxx": [ {phase, pixel_spacing, ...,
          annotation:{num_targets, lesion:{"0":{category, bbox:{2D_box:[{slice_idx,x_min,y_min,x_max,y_max,area},...]}}}}}, ... 8 thì ] } }

Đặc điểm (đã verify): mỗi bệnh nhân = **1 tổn thương, 1 nhãn, đủ 8 thì**.
Ảnh: `images/MR-<id>_<study>_<suffix>_0000.nii` (suffix = C-pre/C+A/C+V/C+Delay/T2WI/DWI/InPhase/OutPhase).

Chống rủi ro trục: trục qua-lát = trục có kích thước NHỎ NHẤT (số lát ít hơn in-plane) → tự dò,
không phụ thuộc thứ tự lưu. Quy ước cắt [y0:y1, x0:x1]; nếu QC thấy lệch → bật `bbox_transpose`.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np


def load_annotation(root: str, ann_name: str):
    data = json.load(open(os.path.join(root, ann_name), encoding="utf-8"))
    ann = data["Annotation_info"]
    cat = data["Category_info"]
    id2name = {v: k for k, v in cat.items() if isinstance(v, int)}
    malignant = set(cat.get("Malignant", []))
    return ann, id2name, malignant


def patient_category(entries) -> int:
    """Nhãn bệnh nhân = category của lesion (nhất quán qua các thì)."""
    for e in entries:
        for lv in e.get("annotation", {}).get("lesion", {}).values():
            c = lv.get("category")
            if c is not None:
                return int(c)
    return -1


def _phase_key(name: str) -> str:
    """Chuẩn hoá tên thì (json 'In Phase' → khớp file 'InPhase')."""
    return name.replace(" ", "")


def entry_by_phase(entries) -> dict:
    """Map suffix-thì → entry (đã chuẩn hoá tên)."""
    return {_phase_key(e.get("phase", "")): e for e in entries}


def find_image(images_dir: str, patient: str, phase_suffix: str) -> str | None:
    hits = glob.glob(os.path.join(images_dir, f"{patient}_*_{phase_suffix}_0000.nii*"))
    return sorted(hits)[0] if hits else None


def representative_box(entry):
    """Trả (slice_idx, x0, y0, x1, y1) của 2D_box có DIỆN TÍCH lớn nhất (lát đại diện)."""
    boxes = []
    for lv in entry.get("annotation", {}).get("lesion", {}).values():
        boxes += lv.get("bbox", {}).get("2D_box", [])
    if not boxes:
        return None
    b = max(boxes, key=lambda d: d.get("area", 0.0))
    return int(b["slice_idx"]), float(b["x_min"]), float(b["y_min"]), float(b["x_max"]), float(b["y_max"])


def _robust_u8(sl: np.ndarray) -> np.ndarray:
    """MRI → [0,255] uint8 bằng clip percentile [1,99] (bền với outlier)."""
    sl = sl.astype(np.float32)
    lo, hi = np.percentile(sl, 1), np.percentile(sl, 99)
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((sl - lo) / (hi - lo), 0, 1) * 255.0).astype(np.uint8)


def load_slice(nii_path: str, slice_idx: int):
    """Trả 2D slice (native, không reorient) tại slice_idx trên trục qua-lát (nhỏ nhất)."""
    import nibabel as nib

    arr = np.asanyarray(nib.load(nii_path).dataobj)
    if arr.ndim == 2:
        return arr
    z_axis = int(np.argmin(arr.shape))               # trục ít lát nhất = qua-lát
    idx = min(max(slice_idx, 0), arr.shape[z_axis] - 1)
    return np.take(arr, idx, axis=z_axis)


def crop_roi(slice2d: np.ndarray, box, margin: float, size: int, transpose: bool):
    """[bbox-JSON, dự phòng] Cắt ROI quanh bbox rồi resize size×size, robust-u8."""
    import cv2

    _, x0, y0, x1, y1 = box
    sl = slice2d.T if transpose else slice2d
    H, W = sl.shape
    bw, bh = (x1 - x0), (y1 - y0)
    x0 = int(max(0, x0 - margin * bw)); x1 = int(min(W, x1 + margin * bw))
    y0 = int(max(0, y0 - margin * bh)); y1 = int(min(H, y1 + margin * bh))
    roi = sl[y0:y1, x0:x1]
    if roi.size == 0:
        roi = sl
    roi = _robust_u8(roi)
    return cv2.resize(roi, (size, size), interpolation=cv2.INTER_AREA)


# ================= ROI từ MASK MedSAM2 (khuyến nghị — cùng không gian với ảnh) =================
def find_mask(labels_dir: str, patient: str, phase_suffix: str) -> str | None:
    """Mask cùng (bệnh nhân, thì): labels/MR-<id>_<study>_<suffix>.nii (không có _0000)."""
    hits = glob.glob(os.path.join(labels_dir, f"{patient}_*_{phase_suffix}.nii*"))
    return sorted(hits)[0] if hits else None


def _z_axis(shape) -> int:
    return int(np.argmin(shape))


def mask_roi(image_path: str, mask_path: str, margin: float, size: int):
    """ROI = bbox vùng mask>0 trên lát có DIỆN TÍCH mask lớn nhất. Không lo trục vì mask cùng
    hệ mảng với ảnh. Trả (roi_u8[size,size], slice_idx, (x0,y0,x1,y1), mask_area) hoặc None."""
    import cv2
    import nibabel as nib

    img = np.asanyarray(nib.load(image_path).dataobj)
    msk = np.asanyarray(nib.load(mask_path).dataobj)
    if img.shape != msk.shape:
        return None
    za = _z_axis(img.shape)
    other = tuple(a for a in range(msk.ndim) if a != za)
    areas = (msk > 0).sum(axis=other)
    if int(areas.max()) == 0:
        return None
    zi = int(np.argmax(areas))
    m2 = np.take(msk > 0, zi, axis=za)
    i2 = np.take(img, zi, axis=za)
    ys, xs = np.where(m2)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    bh, bw = (y1 - y0), (x1 - x0)
    y0 = int(max(0, y0 - margin * bh)); y1 = int(min(m2.shape[0], y1 + margin * bh))
    x0 = int(max(0, x0 - margin * bw)); x1 = int(min(m2.shape[1], x1 + margin * bw))
    roi = _robust_u8(i2[y0:y1, x0:x1])
    roi = cv2.resize(roi, (size, size), interpolation=cv2.INTER_AREA)
    return roi, zi, (x0, y0, x1, y1), int(areas[zi])
