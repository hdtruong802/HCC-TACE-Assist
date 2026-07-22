"""Grad-CAM sanity check — model có bám vùng gan/tổn thương hay "shortcut"?

Generic cho mọi backbone timm (resnet50 / convnextv2_nano / fastvit_sa12 /
efficientnet_b0 ...) nhờ dùng `forward_features` → `forward_head`
(feature map [B,C,H,W] trước global-pool làm target Grad-CAM).

Xuất montage overlay theo nhóm TP / FP / FN để soi định tính:
  - TP (đúng, tự tin): heat phải nằm trên tổn thương.
  - FP (âm nhưng model kêu dương): model bám vào cái gì? (artifact/rìa/marker → shortcut).
  - FN (u nhưng model bỏ sót): tổn thương nào bị lơ.

Lưu ý: cache đã liver-crop nên gần như cả ảnh = vùng gan → check ở đây là
"heat có tập trung đúng ổ tổn thương trong gan" chứ không phải "trong/ngoài gan".
Không có mask u trong cache nên đây là kiểm chứng ĐỊNH TÍNH (xem mắt thường).

Ví dụ (Kaggle):
    python -m src.interpretability.gradcam --config configs/train/base.yaml \
        --data-root /kaggle/input/lits-processed \
        --ckpt /kaggle/working/outputs/convnextv2_nano_fold0/best.ckpt \
        --split val --fold 0 --n-per 6 \
        --out /kaggle/working/outputs/convnextv2_nano_fold0/gradcam
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.dataset import (  # noqa: E402
    IMAGENET_MEAN, IMAGENET_STD, LitsSliceDataset, build_transforms,
    fold_patients, load_split, resolve,
)
from src.models.factory import build_model  # noqa: E402
from src.training.train import predict  # noqa: E402


def gradcam_map(model, x, size):
    """Grad-CAM cho 1 ảnh x=[1,3,H,W] (fp32, đã normalize). Trả cam [size,size] in [0,1].

    Dùng forward_features (feature map spatial) + forward_head (pool+classifier);
    weight = grad trung bình theo không gian, tổ hợp kênh → ReLU → chuẩn hoá.
    """
    model.zero_grad(set_to_none=True)
    feat = model.forward_features(x)          # [1,C,h,w]
    if feat.dim() != 4:
        raise RuntimeError(f"forward_features trả tensor {feat.shape}, cần 4D [B,C,H,W]")
    feat.retain_grad()
    logit = model.forward_head(feat)          # [1,1]
    logit.squeeze().backward()
    grad = feat.grad                          # [1,C,h,w]
    weight = grad.mean(dim=(2, 3), keepdim=True)          # [1,C,1,1]
    cam = (weight * feat).sum(dim=1, keepdim=True).relu()  # [1,1,h,w]
    cam = F.interpolate(cam, size=(size, size), mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().float().cpu().numpy()
    cam -= cam.min()
    m = cam.max()
    return cam / m if m > 1e-8 else cam, float(torch.sigmoid(logit).item())


def _diversify(cands, pids, n_per, cap):
    """Lấy tối đa `cap` slice/bệnh nhân theo thứ tự đã sort; thiếu thì bù bằng phần còn lại."""
    seen, chosen, chosen_set = {}, [], set()
    for i in cands:
        if seen.get(pids[i], 0) < cap:
            chosen.append(i); chosen_set.add(i); seen[pids[i]] = seen.get(pids[i], 0) + 1
        if len(chosen) >= n_per:
            return np.array(chosen, dtype=int)
    for i in cands:                                  # bù nếu không đủ bệnh nhân khác nhau
        if i not in chosen_set:
            chosen.append(i)
            if len(chosen) >= n_per:
                break
    return np.array(chosen[:n_per], dtype=int)


def _pick(df, probs, thr, n_per, cap=1):
    """Chọn chỉ số slice theo nhóm TP / FP / FN, đa dạng theo bệnh nhân (cap slice/bn)."""
    y = df.label.values.astype(int)
    pids = df.patient_id.values
    tp = np.where((y == 1) & (probs >= thr))[0]
    fp = np.where((y == 0) & (probs >= thr))[0]
    fn = np.where((y == 1) & (probs < thr))[0]
    tp = tp[np.argsort(-probs[tp])]                  # dương đúng, tự tin nhất
    fp = fp[np.argsort(-probs[fp])]                  # âm bị kêu dương, tự tin nhất
    fn = fn[np.argsort(probs[fn])]                   # u bị bỏ sót "chắc" nhất
    return {"TP": _diversify(tp, pids, n_per, cap),
            "FP": _diversify(fp, pids, n_per, cap),
            "FN": _diversify(fn, pids, n_per, cap)}


def _overlay(ax, raw_u8, cam, title):
    ax.imshow(raw_u8, cmap="gray", vmin=0, vmax=255)
    ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train/base.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-per", type=int, default=6, help="số ví dụ mỗi nhóm TP/FP/FN")
    ap.add_argument("--cap", type=int, default=1, help="tối đa số slice/bệnh nhân mỗi nhóm (đa dạng hoá)")
    ap.add_argument("--threshold", type=float, default=None, help="mặc định: slice_threshold từ ckpt")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    root = args.data_root or os.environ.get("DATA_ROOT") or cfg["data"]["processed_root"]
    size = cfg["data"]["size"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location=device)
    model = build_model(ck["arch"], pretrained=False, drop_rate=cfg["model"]["drop_rate"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    manifest = pd.read_csv(resolve(root, cfg["data"]["manifest"]))
    split = load_split(resolve(root, cfg["data"]["split"]))
    _, val_p, test_p = fold_patients(split, args.fold)
    patients = val_p if args.split == "val" else test_p
    img_path = resolve(root, cfg["data"]["image_file"])
    ds = LitsSliceDataset(manifest, img_path, patients, build_transforms(size, False))
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
                    num_workers=cfg["train"]["num_workers"], pin_memory=True)

    probs, labels, _ = predict(model, dl, device)      # aligned với ds.df
    thr = args.threshold
    if thr is None:
        thr = ck.get("val", {}).get("slice_threshold", 0.5)
    groups = _pick(ds.df, probs, thr, args.n_per, args.cap)
    print(f"{args.split}: {len(ds)} slice | thr={thr:.3f} | "
          f"TP={len(groups['TP'])} FP={len(groups['FP'])} FN={len(groups['FN'])}")

    out = args.out or os.path.join(os.path.dirname(args.ckpt), "gradcam")
    os.makedirs(out, exist_ok=True)
    mean = np.array(IMAGENET_MEAN); std = np.array(IMAGENET_STD)

    rows = [g for g in ("TP", "FP", "FN") if len(groups[g])]
    ncol = args.n_per
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.0 * ncol, 2.2 * len(rows)),
                             squeeze=False)
    for ri, gname in enumerate(rows):
        for ci in range(ncol):
            ax = axes[ri][ci]
            idxs = groups[gname]
            if ci >= len(idxs):
                ax.axis("off"); continue
            i = int(idxs[ci])
            r = ds.df.iloc[i]
            raw = np.asarray(ds.arr[int(r.row)], dtype=np.uint8)      # [H,W] 256
            img3 = np.repeat(raw[:, :, None], 3, axis=2).astype(np.float32) / 255.0
            img3 = (img3 - mean) / std
            x = torch.from_numpy(img3).permute(2, 0, 1).unsqueeze(0).float().to(device)
            x.requires_grad_(True)
            cam, p = gradcam_map(model, x, size)
            _overlay(ax, raw, cam, f"{gname} p={p:.2f} y={int(r.label)}\n{r.patient_id}")
    fig.suptitle(f"Grad-CAM {ck['arch']} · {args.split} fold{args.fold} · thr={thr:.2f}", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    montage = os.path.join(out, f"gradcam_{args.split}.png")
    fig.savefig(montage, dpi=120); plt.close(fig)
    print(f"→ {montage}")


if __name__ == "__main__":
    main()
