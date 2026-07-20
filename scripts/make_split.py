"""Chia patient-level (StratifiedGroupKFold + hold-out) → split.json.

Ví dụ:
    python scripts/make_split.py --manifest /kaggle/working/manifest.csv \
        --out /kaggle/working/splits/lits_v1.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import sys

import pandas as pd
import yaml
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

try:  # in ký tự đặc biệt an toàn trên mọi console
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/lits.yaml")
    ap.add_argument("--manifest", default="/kaggle/working/manifest.csv")
    ap.add_argument("--out", default="/kaggle/working/splits/lits_v1.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))["split"]
    seed, k, test_frac = cfg["seed"], cfg["k_folds"], cfg["test_frac"]

    df = pd.read_csv(args.manifest)
    pat = df.groupby("patient_id").label.max().reset_index().rename(columns={"label": "y"})
    print(f"Patients={len(pat)}  pos={int(pat.y.sum())}  neg={int((pat.y == 0).sum())}")

    strat = pat.y if pat.y.nunique() > 1 else None
    if strat is None:
        print("[WARN] chỉ 1 lớp ở patient-level → không stratify được (đặc thù LiTS ít ca 'bình thường').")

    trainval, test = train_test_split(pat, test_size=test_frac, stratify=strat, random_state=seed)

    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    X = trainval.patient_id.values
    y = trainval.y.values
    folds = []
    for tr, va in sgkf.split(X, y, groups=X):
        folds.append({"train": sorted(X[tr].tolist()), "val": sorted(X[va].tolist())})

    out = {
        "seed": seed, "k_folds": k, "test_frac": test_frac,
        "test": sorted(test.patient_id.tolist()),
        "folds": folds,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    js = json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True)
    open(args.out, "w", encoding="utf-8").write(js)
    h = hashlib.sha256(js.encode("utf-8")).hexdigest()[:12]
    print(f"Split → {args.out}\ntest={len(out['test'])} patients, {k} folds, seed={seed}, hash={h}")


if __name__ == "__main__":
    main()