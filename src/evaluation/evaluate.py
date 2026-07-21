"""Đánh giá checkpoint → metric patient-level + ROC/PR/confusion/calibration curves.

Ví dụ:
    python -m src.evaluation.evaluate --config configs/train/base.yaml \
        --data-root /kaggle/input/lits-processed --ckpt /kaggle/working/outputs/resnet50_fold0/best.ckpt \
        --split val --fold 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.dataset import (  # noqa: E402
    LitsSliceDataset, build_transforms, fold_patients, load_split, resolve,
)
from src.evaluation.metrics import aggregate_patient, full_report  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.training.train import predict  # noqa: E402


def _curves(pdf, threshold, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix

    os.makedirs(out_dir, exist_ok=True)
    y, s = pdf.label.values, pdf.score.values
    if len(np.unique(y)) < 2:
        print("[WARN] chỉ 1 lớp ở tập này → bỏ vẽ ROC/PR"); return
    RocCurveDisplay.from_predictions(y, s); plt.title("ROC (patient)")
    plt.savefig(os.path.join(out_dir, "roc.png"), dpi=110); plt.close()
    PrecisionRecallDisplay.from_predictions(y, s); plt.title("PR (patient)")
    plt.savefig(os.path.join(out_dir, "pr.png"), dpi=110); plt.close()
    cm = confusion_matrix(y, (s >= threshold).astype(int))
    plt.figure(figsize=(3.2, 3)); plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xticks([0, 1], ["neg", "pos"]); plt.yticks([0, 1], ["neg", "pos"])
    plt.xlabel("pred"); plt.ylabel("true"); plt.title(f"Confusion @thr={threshold:.2f}")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "confusion.png"), dpi=110); plt.close()
    try:
        frac, mean_pred = calibration_curve(y, s, n_bins=8, strategy="quantile")
        plt.figure(figsize=(4, 4)); plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.plot(mean_pred, frac, "o-"); plt.xlabel("dự đoán"); plt.ylabel("thực tế")
        plt.title("Calibration"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "calibration.png"), dpi=110); plt.close()
    except Exception as e:  # noqa: BLE001
        print("calibration skip:", e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train/base.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    root = args.data_root or os.environ.get("DATA_ROOT") or cfg["data"]["processed_root"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location=device)
    model = build_model(ck["arch"], pretrained=False, drop_rate=cfg["model"]["drop_rate"]).to(device)
    model.load_state_dict(ck["model"])

    manifest = pd.read_csv(resolve(root, cfg["data"]["manifest"]))
    split = load_split(resolve(root, cfg["data"]["split"]))
    train_p, val_p, test_p = fold_patients(split, args.fold)
    patients = val_p if args.split == "val" else test_p
    ds = LitsSliceDataset(manifest, resolve(root, cfg["data"]["image_file"]), patients,
                          build_transforms(cfg["data"]["size"], False))
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                    num_workers=cfg["train"]["num_workers"], pin_memory=True)
    print(f"{args.split}: {len(ds)} slices / {len(patients)} bệnh nhân")

    p, lb, pid = predict(model, dl, device)
    # threshold: test dùng ngưỡng đã khóa trên val (từ ckpt), trừ khi --threshold
    threshold = args.threshold
    if threshold is None and args.split == "test":
        threshold = ck.get("val", {}).get("threshold")
    rep = full_report(pid, p, lb, threshold=threshold, cfg=cfg["eval"])
    out_dir = os.path.join(os.path.dirname(args.ckpt), f"eval_{args.split}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(rep, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    pdf = aggregate_patient(pid, p, lb, cfg["eval"]["patient_agg"], cfg["eval"]["topk"])
    _curves(pdf, rep["threshold"], out_dir)

    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in rep.items()}, ensure_ascii=False, indent=2))
    print(f"→ {out_dir}")


if __name__ == "__main__":
    main()
