"""Data audit: phân bố lớp, histogram tumor_area (để chốt τ), QC montage."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


def class_distribution(df: pd.DataFrame) -> dict:
    """Phân bố lớp ở slice-level và patient-level."""
    ppat = df.groupby("patient_id").label.max()
    n = len(df)
    return {
        "n_slices": int(n),
        "n_patients": int(df.patient_id.nunique()),
        "slice_pos": int((df.label == 1).sum()),
        "slice_neg": int((df.label == 0).sum()),
        "slice_pos_ratio": float((df.label == 1).mean()) if n else 0.0,
        "patient_pos": int((ppat == 1).sum()),
        "patient_neg": int((ppat == 0).sum()),
        "slices_per_patient_median": float(df.groupby("patient_id").size().median()) if n else 0.0,
    }


def tumor_area_hist(df: pd.DataFrame, out_png: str = "reports/tumor_area_hist.png", bins: int = 60) -> dict:
    """Histogram log10(tumor_area) trên các slice có u → giúp CHỐT tau_area."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = df.loc[df.tumor_area_px > 0, "tumor_area_px"].values.astype(float)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.figure(figsize=(7, 4))
    if len(pos):
        plt.hist(np.log10(pos + 1), bins=bins, color="#0d7d84")
    plt.xlabel("log10(tumor_area_px + 1)")
    plt.ylabel("số slice")
    plt.title(f"Phân bố diện tích u (n={len(pos)} slice có u)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    plt.close()
    qs = np.percentile(pos, [1, 5, 25, 50, 75, 95]) if len(pos) else []
    return {"png": out_png, "n_tumor_slices": int(len(pos)),
            "percentiles_1_5_25_50_75_95": [float(q) for q in qs]}


def qc_montage(df: pd.DataFrame, cache_root: str, n: int = 24,
               out_png: str = "reports/qc_montage.png", seed: int = 42) -> str:
    """Lưới ảnh mẫu (từ mảng gộp images_u8.npy, memmap) để mắt thường kiểm nhãn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_file = df.image_file.iloc[0] if "image_file" in df.columns else "images_u8.npy"
    arr = np.load(os.path.join(cache_root, image_file), mmap_mode="r")
    sample = df.sample(min(n, len(df)), random_state=seed)
    cols = 6
    rows = int(np.ceil(len(sample) / cols))
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    flat = np.array(axes).ravel()
    for ax, (_, r) in zip(flat, sample.iterrows()):
        img = arr[int(r.row)]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{r.patient_id} L{int(r.label)}", fontsize=7)
        ax.axis("off")
    for ax in flat[len(sample):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=110)
    plt.close()
    return out_png


def write_report(dist: dict, hist: dict, out_md: str = "reports/data_audit_report.md") -> str:
    os.makedirs(os.path.dirname(out_md) or ".", exist_ok=True)
    lines = [
        "# Data Audit Report — LiTS (W1)", "",
        "## Phân bố lớp", "",
        f"- Slices: **{dist['n_slices']}** (pos={dist['slice_pos']}, neg={dist['slice_neg']}, "
        f"pos_ratio={dist['slice_pos_ratio']:.3f})",
        f"- Patients: **{dist['n_patients']}** (pos={dist['patient_pos']}, neg={dist['patient_neg']})",
        f"- Slices/bệnh nhân (median): {dist['slices_per_patient_median']:.0f}", "",
        "## Diện tích u (chốt τ_area)", "",
        f"- Số slice có u: {hist['n_tumor_slices']}",
        f"- Percentiles [1,5,25,50,75,95] px: {hist['percentiles_1_5_25_50_75_95']}",
        f"- Histogram: `{hist['png']}`", "",
        "> Dựa vào percentiles + histogram để CHỐT `tau_area` trong configs/data/lits.yaml.", "",
    ]
    open(out_md, "w", encoding="utf-8").write("\n".join(lines))
    return out_md