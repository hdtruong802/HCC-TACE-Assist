"""Đánh giá Tầng 2 (LLD-MMRI) — macro-F1 + per-class + confusion + malignant-AUC (có CI).

Ví dụ:
    python -m src.evaluation.evaluate_cls --config configs/train/lld_cls.yaml \
        --ckpt /kaggle/working/outputs_cls/convnextv2_nano_fold0/best.ckpt --split val --fold 0
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
from src.data.dataset import fold_patients, load_split, resolve  # noqa: E402
from src.data.lld_dataset import LldDataset  # noqa: E402
from src.evaluation.metrics_cls import multiclass_report  # noqa: E402
from src.models.factory import build_classifier  # noqa: E402
from src.training.train_cls import predict  # noqa: E402


def _confusion_png(cm, names, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    short = [n[:10] for n in names]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xlabel("dự đoán"); ax.set_ylabel("thực tế"); ax.set_title("Confusion (7 lớp)")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train/lld_cls.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    root = args.data_root or os.environ.get("DATA_ROOT") or cfg["data"]["processed_root"]
    dc = cfg["data"]; phases = dc["phases"]; ncls = cfg["model"]["num_classes"]
    names = yaml.safe_load(open("configs/data/lld.yaml", encoding="utf-8"))["classes"]["names"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location=device)
    model = build_classifier(ck["arch"], ncls, in_chans=len(phases), pretrained=False,
                             drop_rate=cfg["model"]["drop_rate"]).to(device)
    model.load_state_dict(ck["model"])

    manifest = pd.read_csv(resolve(root, dc["manifest"]))
    split = load_split(resolve(root, dc["split"]))
    train_p, val_p, test_p = fold_patients(split, args.fold)
    patients = val_p if args.split == "val" else test_p
    ds = LldDataset(manifest, resolve(root, dc["image_file"]), patients, phases, dc["size"], False)
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                    num_workers=cfg["train"]["num_workers"], pin_memory=True)
    print(f"{args.split}: {len(ds)} bệnh nhân")

    probs, labels = predict(model, dl, device)
    rep = multiclass_report(probs, labels, cfg["eval"]["malignant"], n_boot=cfg["eval"]["bootstrap_n"])
    rep["class_names"] = names
    out_dir = os.path.join(os.path.dirname(args.ckpt), f"eval_{args.split}")
    os.makedirs(out_dir, exist_ok=True)
    json.dump(rep, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
    _confusion_png(rep["confusion"], names, os.path.join(out_dir, "confusion.png"))

    print(json.dumps({k: v for k, v in rep.items() if k != "confusion"}, ensure_ascii=False, indent=2))
    print(f"→ {out_dir}")


if __name__ == "__main__":
    main()
