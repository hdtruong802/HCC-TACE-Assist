"""QC LLD-MMRI: xác nhận ROI lấy từ MASK MedSAM2 có trùm đúng tổn thương.
Hiển thị lát có mask lớn nhất: overlay mask (đỏ) + bbox ROI (vàng) + ảnh ROI cắt.

Ví dụ (Kaggle):
    python scripts/qc_lld.py --config configs/data/lld.yaml --data-root <LLD_ROOT> \
        --n 7 --phase "C+A" --out /kaggle/working/qc_lld.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.lld_ingest import (  # noqa: E402
    entry_by_phase, find_image, find_mask, load_annotation, load_slice,
    mask_roi, patient_category, _robust_u8,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/lld.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--phase", default="C+A")
    ap.add_argument("--out", default="/kaggle/working/qc_lld.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    root = args.data_root or cfg["dataset"]["data_root"]
    pp = cfg["preprocess"]
    size, margin = pp["size"], pp["roi_margin"]
    names = cfg["classes"]["names"]
    malignant = set(cfg["classes"]["malignant"])
    images_dir = os.path.join(root, cfg["dataset"]["images_dir"])
    labels_dir = os.path.join(root, cfg["dataset"]["labels_dir"])

    ann, _, _ = load_annotation(root, cfg["dataset"]["annotation"])
    by_cat: dict[int, str] = {}
    for pid, entries in ann.items():
        by_cat.setdefault(patient_category(entries), pid)
    picks = [by_cat[c] for c in sorted(by_cat)][: args.n]

    ph = args.phase.replace(" ", "")
    fig, axes = plt.subplots(len(picks), 2, figsize=(6, 3 * len(picks)), squeeze=False)
    for ri, pid in enumerate(picks):
        c = patient_category(ann[pid])
        axL, axR = axes[ri]
        img = find_image(images_dir, pid, ph)
        msk = find_mask(labels_dir, pid, ph)
        res = mask_roi(img, msk, margin, size) if (img and msk) else None
        if res is None:
            axL.set_title(f"{pid}: thiếu ảnh/mask/ROI"); axL.axis("off"); axR.axis("off"); continue
        roi, zi, (x0, y0, x1, y1), area = res
        i2 = load_slice(img, zi); m2 = load_slice(msk, zi) > 0
        axL.imshow(_robust_u8(i2), cmap="gray")
        axL.imshow(np.ma.masked_where(~m2, m2), cmap="autumn", alpha=0.45)
        axL.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="yellow", lw=1.5))
        mal = "ÁC" if c in malignant else "lành"
        axL.set_title(f"{pid} · {names[c]} ({mal}) · z={zi}", fontsize=7); axL.axis("off")
        axR.imshow(roi, cmap="gray"); axR.set_title("ROI cắt", fontsize=7); axR.axis("off")

    fig.suptitle(f"QC LLD-MMRI (ROI từ mask) · phase={args.phase}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120); plt.close(fig)
    print(f"→ {args.out}  (kiểm tra: mask đỏ trùm tổn thương, ROI là ổ tổn thương)")


if __name__ == "__main__":
    main()
