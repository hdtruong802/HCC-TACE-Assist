"""Đánh giá: gộp slice→patient, AUROC/PR, Sens/Spec, threshold, bootstrap CI (mức bệnh nhân)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def aggregate_patient(patient_ids, probs, labels, method: str = "mean_topk", topk: int = 3) -> pd.DataFrame:
    """Gộp xác suất slice → điểm mức bệnh nhân. Nhãn bệnh nhân = max nhãn slice."""
    df = pd.DataFrame({"patient_id": patient_ids, "prob": probs, "label": labels})
    g = df.groupby("patient_id")
    label = g["label"].max()
    if method == "max":
        score = g["prob"].max()
    elif method == "mean":
        score = g["prob"].mean()
    else:  # mean_topk
        k = max(1, topk)
        score = g["prob"].apply(lambda s: np.sort(s.to_numpy())[::-1][:k].mean())
    return pd.DataFrame({
        "patient_id": label.index,
        "score": score.reindex(label.index).to_numpy(dtype=float),
        "label": label.to_numpy().astype(int),
    }).reset_index(drop=True)


def _safe_auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan")


def choose_threshold(y, s, strategy: str = "youden", target_sens: float = 0.90):
    """Chọn ngưỡng trên VALIDATION. youden: max(Sens+Spec-1); sens_priority: Sens>=target."""
    fpr, tpr, thr = roc_curve(y, s)
    if strategy == "sens_priority":
        ok = np.where(tpr >= target_sens)[0]
        idx = ok[np.argmin(fpr[ok])] if len(ok) else int(np.argmax(tpr - fpr))
    else:
        idx = int(np.argmax(tpr - fpr))
    return float(thr[idx])


def point_metrics(y, s, threshold: float) -> dict:
    y = np.asarray(y); s = np.asarray(s)
    pred = (s >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(1, tp + fn); spec = tn / max(1, tn + fp)
    prec = tp / max(1, tp + fp)
    return {
        "threshold": float(threshold), "sensitivity": sens, "specificity": spec,
        "precision": prec, "f1": 2 * prec * sens / max(1e-9, prec + sens),
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def bootstrap_auc(y, s, n: int = 2000, seed: int = 42) -> dict:
    """95% CI cho AUROC bằng resample BỆNH NHÂN (không phải slice)."""
    y = np.asarray(y); s = np.asarray(s); rng = np.random.default_rng(seed); N = len(y)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    if not vals:
        return {"auroc_mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"auroc_mean": float(np.mean(vals)), "ci_low": float(lo), "ci_high": float(hi)}


def sens_at_spec(y, s, target_spec: float = 0.90) -> float:
    """Sensitivity tại mức Specificity mục tiêu."""
    fpr, tpr, _ = roc_curve(y, s)
    ok = np.where((1 - fpr) >= target_spec)[0]
    return float(tpr[ok].max()) if len(ok) else float("nan")


def bootstrap_slice_auc(patient_ids, probs, labels, n: int = 2000, seed: int = 42) -> dict:
    """95% CI cho slice-level AUROC bằng CLUSTER bootstrap (resample BỆNH NHÂN, giữ cả slice của họ).

    Không resample slice độc lập (sẽ phóng đại độ chắc chắn vì slice cùng bn tương quan).
    """
    patient_ids = np.asarray(patient_ids); probs = np.asarray(probs); labels = np.asarray(labels)
    uniq = np.unique(patient_ids)
    idx_by_pid = {pid: np.where(patient_ids == pid)[0] for pid in uniq}
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_pid[p] for p in samp])
        yy = labels[idx]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(roc_auc_score(yy, probs[idx]))
    if not vals:
        return {"slice_auroc_mean": float("nan"), "slice_ci_low": float("nan"), "slice_ci_high": float("nan")}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"slice_auroc_mean": float(np.mean(vals)), "slice_ci_low": float(lo), "slice_ci_high": float(hi)}


def full_report(patient_ids, probs, labels, threshold=None, cfg=None, slice_bootstrap=False) -> dict:
    """Metric cả PATIENT-level và SLICE-level. threshold=None → tự chọn (Youden) cho patient."""
    cfg = cfg or {}
    target = cfg.get("target_spec", 0.90)
    boot = cfg.get("bootstrap_n", 2000)

    # ---- patient-level (đơn vị báo cáo chính, nhưng ít bệnh nhân → nhiễu) ----
    pdf = aggregate_patient(patient_ids, probs, labels,
                            cfg.get("patient_agg", "mean_topk"), cfg.get("topk", 3))
    y, s = pdf.label.values, pdf.score.values
    thr_pat = threshold if threshold is not None else choose_threshold(y, s, "youden")
    rep = {
        "n_patients": int(len(pdf)), "n_pos": int((y == 1).sum()),
        "auroc": _safe_auc(y, s),
        "pr_auc": float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else float("nan"),
        "sens_at_spec90": sens_at_spec(y, s, target),
        **point_metrics(y, s, thr_pat),
        **bootstrap_auc(y, s, boot),
    }

    # ---- slice-level (ổn định hơn: nhiều slice âm; CI theo cluster-bootstrap bệnh nhân) ----
    probs = np.asarray(probs); labels = np.asarray(labels)
    thr_sl = choose_threshold(labels, probs, "youden") if len(np.unique(labels)) > 1 else 0.5
    sm = point_metrics(labels, probs, thr_sl)
    rep.update({
        "slice_n": int(len(labels)), "slice_pos": int((labels == 1).sum()),
        "slice_auroc": _safe_auc(labels, probs),
        "slice_pr_auc": float(average_precision_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan"),
        "slice_sens_at_spec90": sens_at_spec(labels, probs, target),
        "slice_threshold": sm["threshold"], "slice_sensitivity": sm["sensitivity"],
        "slice_specificity": sm["specificity"], "slice_f1": sm["f1"], "slice_accuracy": sm["accuracy"],
    })
    if slice_bootstrap:
        rep.update(bootstrap_slice_auc(patient_ids, probs, labels, boot))
    return rep
