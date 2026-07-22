"""Metric phân loại đa lớp (Tầng 2) — mức bệnh nhân, kèm nhánh nhị phân lành/ác.

Chính: macro-F1 (mất cân bằng 7 lớp). Phụ: balanced-accuracy, accuracy, per-class F1,
confusion, và **malignant AUC** (nhị phân từ tổng xác suất các lớp ác). CI = bootstrap
resample BỆNH NHÂN (mỗi mẫu = 1 bệnh nhân ở suy luận).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score,
)


def multiclass_report(probs, labels, malignant_ids, n_boot: int = 1000, seed: int = 42) -> dict:
    probs = np.asarray(probs); labels = np.asarray(labels).astype(int)
    preds = probs.argmax(1)
    mal = np.array(sorted(malignant_ids))
    mal_true = np.isin(labels, mal).astype(int)
    mal_score = probs[:, mal].sum(1)

    def _mal_auc(yt, ys):
        return float(roc_auc_score(yt, ys)) if len(np.unique(yt)) > 1 else float("nan")

    rep = {
        "n": int(len(labels)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "balanced_acc": float(balanced_accuracy_score(labels, preds)),
        "accuracy": float(accuracy_score(labels, preds)),
        "per_class_f1": [round(float(v), 4) for v in
                         f1_score(labels, preds, average=None,
                                  labels=list(range(probs.shape[1])), zero_division=0)],
        "confusion": confusion_matrix(labels, preds, labels=list(range(probs.shape[1]))).tolist(),
        "malignant_auc": _mal_auc(mal_true, mal_score),
    }
    # ---- bootstrap CI (resample bệnh nhân) ----
    rng = np.random.default_rng(seed); N = len(labels)
    f1s, aucs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        f1s.append(f1_score(labels[idx], preds[idx], average="macro"))
        yt = mal_true[idx]
        if len(np.unique(yt)) > 1:
            aucs.append(roc_auc_score(yt, mal_score[idx]))
    if f1s:
        lo, hi = np.percentile(f1s, [2.5, 97.5])
        rep["macro_f1_ci"] = [round(float(lo), 4), round(float(hi), 4)]
    if aucs:
        lo, hi = np.percentile(aucs, [2.5, 97.5])
        rep["malignant_auc_ci"] = [round(float(lo), 4), round(float(hi), 4)]
    return rep
