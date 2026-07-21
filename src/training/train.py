"""Train loop: AMP + discriminative LR + gradual unfreeze + early stopping + patient AUROC + MLflow.

Ví dụ (Kaggle GPU):
    python -m src.training.train --config configs/train/base.yaml \
        --data-root /kaggle/input/lits-processed --arch resnet50 --fold 0
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.dataset import (  # noqa: E402
    LitsSliceDataset, build_transforms, fold_patients, load_split, resolve,
)
from src.evaluation.metrics import full_report  # noqa: E402
from src.models.factory import build_model, param_groups, set_backbone_requires_grad  # noqa: E402


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def focal_loss(logits, targets, gamma, pos_weight):
    ce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)
    return ((1 - pt) ** gamma * ce).mean()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, labels, pids = [], [], []
    for img, y, pid in loader:
        img = img.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            logit = model(img).squeeze(1)
        probs.append(torch.sigmoid(logit).float().cpu().numpy())
        labels.append(y.numpy()); pids.extend(pid)
    return np.concatenate(probs), np.concatenate(labels), np.array(pids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train/base.yaml")
    ap.add_argument("--data-root", default=None, help="override data.processed_root")
    ap.add_argument("--arch", default=None, help="override model.arch")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    root = args.data_root or os.environ.get("DATA_ROOT") or cfg["data"]["processed_root"]
    arch = args.arch or cfg["model"]["arch"]
    tr = cfg["train"]
    fold = args.fold if args.fold is not None else tr["fold"]
    epochs = args.epochs or tr["epochs"]
    seed_everything(tr["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ngpu = torch.cuda.device_count()
    print(f"arch={arch} fold={fold} epochs={epochs} device={device} GPUs={ngpu} "
          f"(dùng cuda:0; xem hướng dẫn dùng cả 2 GPU trong docs/W2_training_plan.md)")

    # ---- data ----
    manifest = pd.read_csv(resolve(root, cfg["data"]["manifest"]))
    split = load_split(resolve(root, cfg["data"]["split"]))
    train_p, val_p, _ = fold_patients(split, fold)
    size = cfg["data"]["size"]
    img_path = resolve(root, cfg["data"]["image_file"])
    ds_tr = LitsSliceDataset(manifest, img_path, train_p, build_transforms(size, True))
    ds_va = LitsSliceDataset(manifest, img_path, val_p, build_transforms(size, False))
    nw = tr["num_workers"]
    dl_tr = DataLoader(ds_tr, batch_size=tr["batch_size"], shuffle=True, num_workers=nw,
                       pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=tr["batch_size"], shuffle=False, num_workers=nw, pin_memory=True)
    print(f"train slices={len(ds_tr)} ({len(train_p)} bn) | val slices={len(ds_va)} ({len(val_p)} bn)")

    # ---- model / optim ----
    model = build_model(arch, cfg["model"]["pretrained"], cfg["model"]["drop_rate"]).to(device)
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
        scaler = torch.amp.GradScaler("cuda", enabled=amp_on)   # API mới (torch>=2.3)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_on)
    pw = ds_tr.pos_weight() if tr["pos_weight"] == "auto" else float(tr["pos_weight"])
    pos_weight = torch.tensor([pw], device=device)
    print(f"pos_weight={pw:.2f} loss={tr['loss']}")

    use_mlflow = cfg["output"].get("mlflow", False)
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_experiment("liver-cancer-ai")
            mlflow.start_run(run_name=f"{arch}_fold{fold}")
            mlflow.log_params({"arch": arch, "fold": fold, "epochs": epochs, "pos_weight": pw,
                               "loss": tr["loss"], "batch_size": tr["batch_size"]})
        except Exception as e:  # noqa: BLE001
            print("mlflow off:", e); use_mlflow = False

    out_dir = os.path.join(cfg["output"]["dir"], f"{arch}_fold{fold}")
    os.makedirs(out_dir, exist_ok=True)
    best_auc, best_ep, patience = -1.0, -1, tr["early_stopping"]["patience"]

    for ep in range(epochs):
        set_backbone_requires_grad(model, ep >= tr["freeze_backbone_epochs"])
        frozen = " [frozen]" if ep < tr["freeze_backbone_epochs"] else ""
        model.train()
        run_loss, seen = 0.0, 0
        pbar = tqdm(dl_tr, desc=f"Epoch {ep + 1}/{epochs}{frozen}", ncols=100, leave=True)
        for img, y, _ in pbar:
            img = img.to(device, non_blocking=True); y = y.to(device)
            if tr.get("label_smoothing", 0):
                eps = tr["label_smoothing"]; y = y * (1 - eps) + 0.5 * eps
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logit = model(img).squeeze(1)
                if tr["loss"] == "focal":
                    loss = focal_loss(logit, y, tr["focal_gamma"], pos_weight)
                else:
                    loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)   # chống inf/nan
            scaler.step(opt); scaler.update()
            run_loss += loss.item() * img.size(0); seen += img.size(0)
            pbar.set_postfix(loss=f"{run_loss / seen:.4f}")
        pbar.close()
        sched.step()
        train_loss = run_loss / max(1, seen)

        # ---- validate (patient-level) + in TOÀN BỘ metric cuối epoch ----
        if len(ds_va):
            p, lb, pid = predict(model, dl_va, device)
            rep = full_report(pid, p, lb, threshold=None, cfg=cfg["eval"])
            val_auc = rep["auroc"]
            msg = (f"  Epoch {ep + 1}/{epochs} | train_loss={train_loss:.4f} | "
                   f"val_auroc={val_auc:.4f} CI[{rep['ci_low']:.3f},{rep['ci_high']:.3f}] | "
                   f"pr_auc={rep['pr_auc']:.3f} | sens@spec90={rep['sens_at_spec90']:.3f} | "
                   f"sens={rep['sensitivity']:.3f} spec={rep['specificity']:.3f} "
                   f"f1={rep['f1']:.3f} acc={rep['accuracy']:.3f} @thr={rep['threshold']:.2f}")
        else:
            val_auc, rep, msg = float("nan"), {}, f"  Epoch {ep + 1}/{epochs} | train_loss={train_loss:.4f}"
        print(msg, flush=True)
        if use_mlflow:
            import mlflow
            mlflow.log_metrics({"val_auroc": val_auc, "train_loss": train_loss}, step=ep)

        if val_auc > best_auc:
            best_auc, best_ep = val_auc, ep
            torch.save({"model": model.state_dict(), "arch": arch, "cfg": cfg,
                        "val": rep, "epoch": ep}, os.path.join(out_dir, "best.ckpt"))
            json.dump(rep, open(os.path.join(out_dir, "val_metrics.json"), "w"), indent=2)
        if best_ep >= 0 and ep - best_ep >= patience:
            print(f"early stop @ ep{ep} (best ep{best_ep} auroc={best_auc:.4f})"); break

    print(f"DONE best val_auroc={best_auc:.4f} @ ep{best_ep} → {out_dir}/best.ckpt")
    if use_mlflow:
        import mlflow
        mlflow.log_metric("best_val_auroc", best_auc); mlflow.end_run()


if __name__ == "__main__":
    main()
