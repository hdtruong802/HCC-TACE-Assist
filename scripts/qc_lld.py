"""QC LLD-MMRI: overlay bbox lên lát đại diện + ROI cắt ra, để mắt thường xác nhận
tọa độ/trục ĐÚNG trước khi build cache. Chạy cho vài bệnh nhân trải đủ 7 lớp.

Nếu hộp/ROI lệch khỏi tổn thương → thử lại với --transpose (rồi set bbox_transpose trong config).

Ví dụ (Kaggle):
    python scripts/qc_lld.py --config configs/data/lld.yaml --n 7 --phase "C+A" \
        --out /kaggle/working/qc_lld.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.lld_ingest import (  # noqa: E402
    crop_roi, entry_by_phase, find_image, load_annotation, load_slice,
    patient_category, representative_box, _robust_u8,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/lld.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--n", type=int, default=7, help="số bệnh nhân (cố trải đủ lớp)")
    ap.add_argument("--phase", default="C+A", help="thì dùng để QC")
    ap.add_argument("--transpose", action="store_true", help="ghi đè bbox_transpose để thử")
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
    transpose = args.transpose or pp.get("bbox_transpose", False)
    names = cfg["classes"]["names"]
    images_dir = os.path.join(root, cfg["dataset"]["images_dir"])

    ann, id2name, malignant = load_annotation(root, cfg["dataset"]["annotation"])

    # chọn 1 bệnh nhân đại diện cho mỗi lớp (tới n lớp)
    by_cat: dict[int, str] = {}
    for pid, entries in ann.items():
        c = patient_category(entries)
        by_cat.setdefault(c, pid)
    picks = [by_cat[c] for c in sorted(by_cat)][: args.n]

    fig, axes = plt.subplots(len(picks), 2, figsize=(6, 3 * len(picks)), squeeze=False)
    ph = args.phase.replace(" ", "")
    for ri, pid in enumerate(picks):
        entries = ann[pid]
        c = patient_category(entries)
        ebp = entry_by_phase(entries)
        entry = ebp.get(ph) or next(iter(ebp.values()))
        box = representative_box(entry)
        img = find_image(images_dir, pid, ph) or find_image(images_dir, pid, entry["phase"].replace(" ", ""))
        axL, axR = axes[ri]
        if box is None or img is None:
            axL.set_title(f"{pid}: thiếu box/ảnh"); axL.axis("off"); axR.axis("off"); continue
        sl = load_slice(img, box[0])
        disp = sl.T if transpose else sl
        axL.imshow(_robust_u8(disp), cmap="gray")
        _, x0, y0, x1, y1 = box
        axL.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="red", lw=1.5))
        mal = "ÁC" if c in malignant else "lành"
        axL.set_title(f"{pid} · {names[c]} ({mal})", fontsize=7); axL.axis("off")
        roi = crop_roi(sl, box, margin, size, transpose)
        axR.imshow(roi, cmap="gray"); axR.set_title("ROI cắt", fontsize=7); axR.axis("off")

    fig.suptitle(f"QC LLD-MMRI · phase={args.phase} · transpose={transpose}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=120); plt.close(fig)
    print(f"→ {args.out}  (kiểm tra: hộp đỏ trùm tổn thương, ROI phải là ổ tổn thương)")


if __name__ == "__main__":
    main()
