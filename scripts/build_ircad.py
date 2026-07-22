"""Sinh cache ảnh gộp + manifest cho 3D-IRCADb-01 (external test), CÙNG FORMAT với LiTS.

Tái dùng nguyên preprocess/label-transfer của W1 → chỉ khác lớp đọc (DICOM). Xuất
`images_u8_ircad.npy` [N,256,256] uint8 + `manifest_ircad.csv` (cột `row` trỏ vào mảng)
để eval_external nạp qua đúng LitsSliceDataset.

Ví dụ (Kaggle, CPU):
    python scripts/build_ircad.py --data-root /kaggle/input --out-dir /kaggle/working
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
from src.data.ircad_ingest import discover_patients, load_patient  # noqa: E402
from src.data.label_transfer import EXCLUDE, make_label, slice_areas  # noqa: E402
from src.data.preprocess import crop_resize, liver_bbox, window_ct  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/ircad.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out-dir", default="/kaggle/working")
    ap.add_argument("--limit", type=int, default=None, help="chỉ xử lý N bệnh nhân đầu (smoke test)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data_root = args.data_root or os.environ.get("IRCAD_ROOT") or os.environ.get("DATA_ROOT") \
        or cfg["dataset"]["data_root"]
    pp, lt, paths = cfg["preprocess"], cfg["label_transfer"], cfg["paths"]
    size, wl, ww = pp["size"], pp["window"]["wl"], pp["window"]["ww"]
    margin, do_crop = pp["crop_margin"], pp["liver_crop"]
    tau_area, tau_liver = lt["tau_area"], lt["tau_liver"]
    ll, tl = lt["liver_label"], lt["tumor_label"]

    os.makedirs(args.out_dir, exist_ok=True)
    patients = discover_patients(data_root)
    if args.limit:
        patients = patients[: args.limit]
    print(f"Tìm thấy {len(patients)} bệnh nhân IRCADb dưới {data_root}")
    if not patients:
        raise SystemExit("Không thấy PATIENT_DICOM nào — kiểm tra IRCAD_ROOT / cấu trúc dataset.")

    images: list[np.ndarray] = []
    rows = []
    n_neg_pat = 0
    for pi, p in enumerate(patients):
        try:
            vol, seg, spacing, has_tumor = load_patient(p)
        except Exception as e:  # noqa: BLE001
            print(f"[SKIP] {p.patient_id}: lỗi đọc {e}")
            continue
        if not has_tumor:
            n_neg_pat += 1
        bbox = liver_bbox(seg, ll, tl, margin) if do_crop else None
        kept = 0
        for z in range(vol.shape[2]):
            liver_area, tumor_area = slice_areas(seg[:, :, z], ll, tl)
            label = make_label(liver_area, tumor_area, tau_area, tau_liver)
            if label == EXCLUDE:
                continue
            img = crop_resize(window_ct(vol[:, :, z], wl, ww), bbox, size)
            rows.append(dict(
                patient_id=p.patient_id, slice_idx=z, row=len(images),
                image_file=paths["image_file"], liver_area_px=liver_area, tumor_area_px=tumor_area,
                has_liver=1, label=label,
                spacing=round(spacing[0], 4), thickness=round(spacing[2], 4),
            ))
            images.append(img)
            kept += 1
        tag = "" if has_tumor else "  [KHÔNG có folder u → ca ÂM]"
        print(f"[{pi + 1}/{len(patients)}] {p.patient_id}: {kept} slice có gan (tổng {len(images)}){tag}")

    if not images:
        raise SystemExit("Không giữ được slice nào — kiểm tra mask gan / τ.")

    arr = np.stack(images).astype(np.uint8)
    np.save(os.path.join(args.out_dir, paths["image_file"]), arr)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, paths["manifest"]), index=False)

    pos = int((df.label == 1).sum())
    ppat = df.groupby("patient_id").label.max()
    print(f"\nẢnh gộp: {paths['image_file']}  shape={arr.shape}  ({arr.nbytes / 1e6:.0f} MB)")
    print(f"Manifest: {paths['manifest']}  rows={len(df)}  patients={df.patient_id.nunique()}")
    print(f"Slice: positive={pos}  negative={len(df) - pos}  pos_ratio={pos / max(1, len(df)):.3f}")
    print(f"Patient: pos={int((ppat == 1).sum())}  neg={int((ppat == 0).sum())} "
          f"(trong đó {n_neg_pat} ca không có folder u)")


if __name__ == "__main__":
    main()
