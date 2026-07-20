"""Sinh cache slice (uint8) + manifest.csv cho LiTS. Chạy trên Kaggle (CPU).

Ví dụ:
    python scripts/build_manifest.py --data-root /kaggle/input/liver-tumor-segmentation \
        --out-dir /kaggle/working [--limit 3]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

try:  # in tiếng Việt/ký tự đặc biệt an toàn trên mọi console
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.io import discover_pairs, load_pair  # noqa: E402
from src.data.label_transfer import EXCLUDE, make_label, slice_areas  # noqa: E402
from src.data.preprocess import crop_resize, liver_bbox, window_ct  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/lits.yaml")
    ap.add_argument("--data-root", default=None, help="override DATA_ROOT")
    ap.add_argument("--out-dir", default="/kaggle/working")
    ap.add_argument("--limit", type=int, default=None, help="chỉ xử lý N volume đầu (smoke test)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    data_root = args.data_root or os.environ.get("DATA_ROOT") or cfg["dataset"]["data_root"]
    pp, lt, paths = cfg["preprocess"], cfg["label_transfer"], cfg["paths"]
    size, wl, ww = pp["size"], pp["window"]["wl"], pp["window"]["ww"]
    margin, do_crop = pp["crop_margin"], pp["liver_crop"]
    tau_area, tau_liver = lt["tau_area"], lt["tau_liver"]
    ll, tl = lt["liver_label"], lt["tumor_label"]

    cache_dir = os.path.join(args.out_dir, paths["cache_dir"])
    os.makedirs(cache_dir, exist_ok=True)

    pairs = discover_pairs(data_root, cfg["dataset"]["volume_glob"], cfg["dataset"]["seg_glob"])
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"Tìm thấy {len(pairs)} cặp volume/seg dưới {data_root}")
    if not pairs:
        raise SystemExit("Không tìm thấy cặp nào — kiểm tra DATA_ROOT / glob trong config.")

    rows = []
    for pi, pair in enumerate(pairs):
        try:
            vol, seg, spacing = load_pair(pair)
        except Exception as e:  # noqa: BLE001
            print(f"[SKIP] {pair.patient_id}: lỗi đọc {e}")
            continue
        bbox = liver_bbox(seg, ll, tl, margin) if do_crop else None
        pdir = os.path.join(cache_dir, pair.patient_id)
        os.makedirs(pdir, exist_ok=True)
        kept = 0
        for z in range(vol.shape[2]):
            liver_area, tumor_area = slice_areas(seg[:, :, z], ll, tl)
            label = make_label(liver_area, tumor_area, tau_area, tau_liver)
            if label == EXCLUDE:
                continue
            img = crop_resize(window_ct(vol[:, :, z], wl, ww), bbox, size)
            rel = os.path.join(paths["cache_dir"], pair.patient_id, f"{z:04d}.npy")
            np.save(os.path.join(args.out_dir, rel), img)
            rows.append(
                dict(
                    patient_id=pair.patient_id, slice_idx=z, cache_path=rel,
                    liver_area_px=liver_area, tumor_area_px=tumor_area,
                    has_liver=1, label=label,
                    spacing=round(spacing[0], 4), thickness=round(spacing[2], 4),
                )
            )
            kept += 1
        print(f"[{pi + 1}/{len(pairs)}] {pair.patient_id}: {kept} slice có gan")

    df = pd.DataFrame(rows)
    man_path = os.path.join(args.out_dir, paths["manifest"])
    df.to_csv(man_path, index=False)

    pos = int((df.label == 1).sum())
    print(f"\nManifest: {man_path}  rows={len(df)}  patients={df.patient_id.nunique()}")
    print(f"Slice: positive={pos}  negative={len(df) - pos}  pos_ratio={pos / max(1, len(df)):.3f}")
    if len(df):
        ppat = df.groupby("patient_id").label.max()
        print(f"Patient: pos={int((ppat == 1).sum())}/{len(ppat)}  neg={int((ppat == 0).sum())}")


if __name__ == "__main__":
    main()