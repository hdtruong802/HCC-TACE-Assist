"""Build cache ROI + manifest cho LLD-MMRI (Tầng 2). Mỗi (bệnh nhân × thì) = 1 ROI.

Xuất `images_u8_lld.npy` [N,256,256] uint8 + `manifest_lld.csv`. Chạy SAU khi QC (qc_lld.py)
xác nhận bbox/trục đúng. Nhãn 7 lớp theo `category`; kèm cột `malignant` (nhị phân).

Ví dụ (Kaggle):
    python scripts/build_lld.py --config configs/data/lld.yaml --out-dir /kaggle/working
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.lld_ingest import (  # noqa: E402
    crop_roi, entry_by_phase, find_image, find_mask, load_annotation, load_slice,
    mask_roi, patient_category, representative_box,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/lld.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-dir", default="/kaggle/working")
    ap.add_argument("--limit", type=int, default=None, help="chỉ N bệnh nhân đầu (smoke test)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    root = args.data_root or cfg["dataset"]["data_root"]
    pp, paths = cfg["preprocess"], cfg["paths"]
    size, margin = pp["size"], pp["roi_margin"]
    transpose = pp.get("bbox_transpose", False)
    phases = cfg["phases"]
    names = cfg["classes"]["names"]
    malignant = set(cfg["classes"]["malignant"])
    images_dir = os.path.join(root, cfg["dataset"]["images_dir"])
    labels_dir = os.path.join(root, cfg["dataset"]["labels_dir"])
    os.makedirs(args.out_dir, exist_ok=True)

    ann, _, _ = load_annotation(root, cfg["dataset"]["annotation"])
    patients = sorted(ann.keys())
    if args.limit:
        patients = patients[: args.limit]
    print(f"LLD-MMRI: {len(patients)} bệnh nhân dưới {root}")

    images: list[np.ndarray] = []
    rows = []
    miss, n_bbox = 0, 0
    for pi, pid in enumerate(patients):
        entries = ann[pid]
        cat = patient_category(entries)
        ebp = entry_by_phase(entries)
        for ph in phases:
            img = find_image(images_dir, pid, ph)
            msk = find_mask(labels_dir, pid, ph)
            if img is None:
                miss += 1
                continue
            try:
                res = mask_roi(img, msk, margin, size) if msk else None
                if res is not None:                       # ROI từ mask (chính)
                    roi, zi, _, area = res
                else:                                     # dự phòng: bbox JSON
                    entry = ebp.get(ph)
                    box = representative_box(entry) if entry else None
                    if box is None:
                        miss += 1; continue
                    roi = crop_roi(load_slice(img, box[0]), box, margin, size, transpose)
                    zi, area = box[0], -1; n_bbox += 1
            except Exception as e:  # noqa: BLE001
                print(f"[SKIP] {pid} {ph}: {e}"); miss += 1; continue
            rows.append(dict(
                patient_id=pid, category=cat, class_name=names[cat],
                malignant=int(cat in malignant), phase=ph, slice_idx=zi,
                mask_area=area, roi_src=("mask" if area >= 0 else "bbox"),
                row=len(images), image_file=paths["image_file"],
            ))
            images.append(roi)
        if (pi + 1) % 50 == 0 or pi == len(patients) - 1:
            print(f"[{pi + 1}/{len(patients)}] tổng ROI={len(images)} (miss={miss}, bbox-fallback={n_bbox})")

    if not images:
        raise SystemExit("Không tạo được ROI nào — kiểm tra data_root / annotation.")

    arr = np.stack(images).astype(np.uint8)
    np.save(os.path.join(args.out_dir, paths["image_file"]), arr)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, paths["manifest"]), index=False)

    print(f"\nẢnh gộp: {paths['image_file']} shape={arr.shape} ({arr.nbytes / 1e6:.0f} MB)")
    print(f"Manifest: {paths['manifest']} rows={len(df)} patients={df.patient_id.nunique()} miss={miss}")
    ppat = df.drop_duplicates("patient_id")
    print("Phân bố lớp (mức bệnh nhân):")
    for c in sorted(ppat.category.unique()):
        n = int((ppat.category == c).sum())
        print(f"  {c} {names[c]:35s}: {n}")
    print(f"Nhị phân (bệnh nhân): ác={int(ppat.malignant.sum())} lành={int((ppat.malignant == 0).sum())}")


if __name__ == "__main__":
    main()
