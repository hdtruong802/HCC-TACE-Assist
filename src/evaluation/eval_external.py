"""Đánh giá GENERALIZATION trên external test 3D-IRCADb-01 (chạm 1 LẦN).

Nạp checkpoint đã chọn trên LiTS-val → suy luận trên toàn bộ IRCADb → báo cáo
slice/patient AUROC + PR + Sens/Spec tại **threshold ĐÃ KHÓA trên val** (không tinh chỉnh).
IRCADb có ca ÂM thật → Specificity ở đây mới có ý nghĩa.

Ví dụ (Kaggle):
    python -m src.evaluation.eval_external --config configs/train/base.yaml \
        --ircad-config configs/data/ircad.yaml \
        --ckpt /kaggle/working/outputs/convnextv2_nano_fold0/best.ckpt \
        --data-root /kaggle/working        # nơi có manifest_ircad.csv + images_u8_ircad.npy
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.dataset import LitsSliceDataset, build_transforms, resolve  # noqa: E402
from src.evaluation.evaluate import _curves, _slice_curves  # noqa: E402
from src.evaluation.metrics import aggregate_patient, full_report  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.training.train import predict  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train/base.yaml")
    ap.add_argument("--ircad-config", default="configs/data/ircad.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default=None, help="nơi chứa manifest_ircad.csv + images_u8_ircad.npy")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    icfg = yaml.safe_load(open(args.ircad_config, encoding="utf-8"))
    paths = icfg["paths"]
    root = args.data_root or os.environ.get("IRCAD_ROOT") or os.environ.get("DATA_ROOT") or "/kaggle/working"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location=device)
    model = build_model(ck["arch"], pretrained=False, drop_rate=cfg["model"]["drop_rate"]).to(device)
    model.load_state_dict(ck["model"])

    manifest = pd.read_csv(resolve(root, paths["manifest"]))
    img_path = resolve(root, paths["image_file"])
    patients = sorted(manifest.patient_id.unique())
    ds = LitsSliceDataset(manifest, img_path, patients, build_transforms(cfg["data"]["size"], False))
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                    num_workers=cfg["train"]["num_workers"], pin_memory=True)

    # ---- ngưỡng KHÓA trên val (từ ckpt) ----
    val = ck.get("val", {})
    thr_pat = val.get("threshold")
    thr_sl = val.get("slice_threshold")
    ppat = manifest.groupby("patient_id").label.max()
    print(f"IRCADb: {len(ds)} slice / {len(patients)} bệnh nhân "
          f"(pos={int((ppat == 1).sum())} neg={int((ppat == 0).sum())})")
    print(f"Ngưỡng khóa từ val → patient={thr_pat} slice={thr_sl}")

    p, lb, pid = predict(model, dl, device)
    rep = full_report(pid, p, lb, threshold=thr_pat, cfg=cfg["eval"],
                      slice_bootstrap=True, slice_threshold=thr_sl)
    rep["_locked_threshold_patient"] = thr_pat
    rep["_locked_threshold_slice"] = thr_sl
    rep["_ckpt"] = os.path.abspath(args.ckpt)
    rep["_arch"] = ck["arch"]

    out_dir = os.path.join(os.path.dirname(args.ckpt), "eval_ircad")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(rep, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    pdf = aggregate_patient(pid, p, lb, cfg["eval"]["patient_agg"], cfg["eval"]["topk"])
    _curves(pdf, thr_pat if thr_pat is not None else rep["threshold"], out_dir)
    _slice_curves(lb, p, out_dir)

    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in rep.items()}, ensure_ascii=False, indent=2))
    print(f"→ {out_dir}")


if __name__ == "__main__":
    main()
