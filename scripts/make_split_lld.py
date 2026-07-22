"""Split PATIENT-LEVEL phân tầng theo 7 lớp cho LLD-MMRI (Tầng 2).

Mỗi bệnh nhân 1 nhãn → StratifiedKFold trên bệnh nhân + hold-out test phân tầng.
Xuất `splits/lld_v1.json` ({test, folds:[{train,val}], hash}). Leakage: tập bệnh nhân rời nhau.

Ví dụ:
    python scripts/make_split_lld.py --config configs/data/lld.yaml \
        --manifest /kaggle/working/manifest_lld.csv --out-dir /kaggle/working
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/lld.yaml")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", default="/kaggle/working")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    sp = cfg["split"]
    seed, k, test_frac = sp["seed"], sp["k_folds"], sp["test_frac"]

    df = pd.read_csv(args.manifest).drop_duplicates("patient_id")[["patient_id", "category"]]
    pids = df.patient_id.to_numpy()
    y = df.category.to_numpy()

    trainval_p, test_p, y_tv, _ = train_test_split(
        pids, y, test_size=test_frac, random_state=seed, stratify=y)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    for tr_idx, va_idx in skf.split(trainval_p, y_tv):
        folds.append({"train": sorted(trainval_p[tr_idx].tolist()),
                      "val": sorted(trainval_p[va_idx].tolist())})

    split = {"test": sorted(test_p.tolist()), "folds": folds, "seed": seed, "k_folds": k}
    blob = json.dumps(split, sort_keys=True).encode()
    split["hash"] = hashlib.sha1(blob).hexdigest()[:7]

    # ---- leakage check ----
    test_set = set(split["test"])
    for i, f in enumerate(folds):
        assert set(f["train"]).isdisjoint(f["val"]), f"fold{i}: train∩val ≠ ∅"
        assert test_set.isdisjoint(f["train"]) and test_set.isdisjoint(f["val"]), f"fold{i}: test rò rỉ"

    out = os.path.join(args.out_dir, "splits")
    os.makedirs(out, exist_ok=True)
    json.dump(split, open(os.path.join(out, "lld_v1.json"), "w"), indent=2)
    print(f"hash={split['hash']} | test={len(test_p)} | trainval={len(trainval_p)} | {k} folds")
    print(f"phân bố test theo lớp: {df[df.patient_id.isin(test_set)].category.value_counts().sort_index().to_dict()}")
    print(f"→ {os.path.join(out, 'lld_v1.json')} (leakage PASS)")


if __name__ == "__main__":
    main()
