"""Train Tầng 2 — phân loại 7 lớp tổn thương gan trên LLD-MMRI (MRI đa thì).

Đầu vào [K,H,W] (K thì) → timm in_chans=K, 7 logit. CE + class-weight (mất cân bằng),
AdamW discriminative LR + cosine/warmup + gradual unfreeze + AMP. Chọn best theo macro-F1 (val).

Ví dụ (Kaggle GPU):
    python -m src.training.train_cls --config configs/train/lld_cls.yaml \
        --data-root /kaggle/working --fold 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.dataset import fold_patients, load_split, resolve  # noqa: E402
from src.data.lld_dataset import LldDataset  # noqa: E402
from src.evaluation.metrics_cls import multiclass_report  # noqa: E402
from src.models.factory import build_classifier, param_groups, set_backbone_requires_grad  # noqa: E402
from src.training.train import seed_everything  # noqa: E402


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logit = model(x)
        probs.append(torch.softmax(logit.float(), 1).cpu().numpy())
        labels.append(np.asarray(y))
    return np.concatenate(probs), np.concatenate(labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train/lld_cls.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    root = args.data_root or os.environ.get("DATA_ROOT") or cfg["data"]["processed_root"]
    arch = args.arch or cfg["model"]["arch"]
    tr, dc = cfg["train"], cfg["data"]
    fold = args.fold if args.fold is not None else tr["fold"]
    epochs = args.epochs or tr["epochs"]
    phases = dc["phases"]; K = len(phases); ncls = cfg["model"]["num_classes"]
    seed_everything(tr["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"arch={arch} in_chans={K} classes={ncls} fold={fold} device={device}")

    manifest = pd.read_csv(resolve(root, dc["manifest"]))
    split = load_split(resolve(root, dc["split"]))
    train_p, val_p, _ = fold_patients(split, fold)
    img_path = resolve(root, dc["image_file"]); size = dc["size"]
    ds_tr = LldDataset(manifest, img_path, train_p, phases, size, True)
    ds_va = LldDataset(manifest, img_path, val_p, phases, size, False)
    nw = tr["num_workers"]
    dl_tr = DataLoader(ds_tr, batch_size=tr["batch_size"], shuffle=True, num_workers=nw,
                       pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=tr["batch_size"], shuffle=False, num_workers=nw, pin_memory=True)
    print(f"train={len(ds_tr)} bn | val={len(ds_va)} bn | phân bố train={ds_tr.class_counts(ncls).tolist()}")

    model = build_classifier(arch, ncls, in_chans=K, pretrained=cfg["model"]["pretrained"],
                             drop_rate=cfg["model"]["drop_rate"]).to(device)
    groups, wd = param_groups(model, tr["lr_head"], tr["lr_backbone"], tr["weight_decay"])
    opt = torch.optim.AdamW(groups, weight_decay=wd)
    warm = tr["warmup_epochs"]

    def lr_scale(ep):
        if ep < warm:
            return (ep + 1) / max(1, warm)
        t = (ep - warm) / max(1, epochs - warm)
        return 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_scale)
    amp_on = bool(tr["amp"]) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_on)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_on)

    if tr.get("class_weight") == "auto":
        cnt = ds_tr.class_counts(ncls).astype(np.float32)
        w = cnt.sum() / (ncls * np.clip(cnt, 1, None))
        class_w = torch.tensor(w, dtype=torch.float32, device=device)
    else:
        class_w = None
    crit = nn.CrossEntropyLoss(weight=class_w,
                               label_smoothing=tr.get("label_smoothing", 0.0))
    print(f"class_weight={None if class_w is None else [round(float(v),2) for v in class_w]}")

    out_dir = os.path.join(cfg["output"]["dir"], f"{arch}_fold{fold}")
    os.makedirs(out_dir, exist_ok=True)
    sel = tr.get("select_metric", "macro_f1")
    best, best_ep, patience = -1.0, -1, tr["early_stopping"]["patience"]

    for ep in range(epochs):
        set_backbone_requires_grad(model, ep >= tr["freeze_backbone_epochs"])
        frozen = " [frozen]" if ep < tr["freeze_backbone_epochs"] else ""
        model.train(); run, seen = 0.0, 0
        pbar = tqdm(dl_tr, desc=f"Epoch {ep + 1}/{epochs}{frozen}", ncols=100, leave=True)
        for x, y, _ in pbar:
            x = x.to(device, non_blocking=True); y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_on):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update()
            run += loss.item() * x.size(0); seen += x.size(0)
            pbar.set_postfix(loss=f"{run / seen:.4f}")
        pbar.close(); sched.step()

        probs, labels = predict(model, dl_va, device)
        rep = multiclass_report(probs, labels, cfg["eval"]["malignant"], n_boot=0)
        val_metric = rep.get(sel, rep["macro_f1"])
        print(f"  Epoch {ep + 1}/{epochs} | loss={run / max(1, seen):.4f} | "
              f"macro_F1={rep['macro_f1']:.4f} | bal_acc={rep['balanced_acc']:.4f} | "
              f"acc={rep['accuracy']:.4f} | malignant_AUC={rep['malignant_auc']:.4f}", flush=True)

        if val_metric > best:
            best, best_ep = val_metric, ep
            torch.save({"model": model.state_dict(), "arch": arch, "cfg": cfg,
                        "in_chans": K, "val": rep, "epoch": ep}, os.path.join(out_dir, "best.ckpt"))
            json.dump(rep, open(os.path.join(out_dir, "val_metrics.json"), "w"), indent=2)
        if best_ep >= 0 and ep - best_ep >= patience:
            print(f"early stop @ Epoch {ep + 1} (best Epoch {best_ep + 1}, {sel}={best:.4f})"); break

    print(f"DONE best {sel}={best:.4f} @ Epoch {best_ep + 1} → {out_dir}/best.ckpt")


if __name__ == "__main__":
    main()
