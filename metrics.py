# -*- coding: utf-8 -*-
from typing import Dict, Optional
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score

def _safe_auroc(y_true, y_prob) -> Optional[float]:
    y = y_true.astype(int)
    if y.min() == y.max(): return None
    try: return float(roc_auc_score(y, y_prob))
    except Exception: return None

def _safe_auprc(y_true, y_prob) -> Optional[float]:
    y = y_true.astype(int)
    if y.min() == y.max(): return None
    try: return float(average_precision_score(y, y_prob))
    except Exception: return None

def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
    auroc = _safe_auroc(y_true, y_prob)
    auprc = _safe_auprc(y_true, y_prob)
    thr_best = f1_best = None
    try:
        p, r, t = precision_recall_curve(y_true.astype(int), y_prob)
        if len(t) > 0:
            f1s = 2*(p[1:]*r[1:])/(p[1:]+r[1:]+1e-12)
            idx = int(np.nanargmax(f1s))
            f1_best, thr_best = float(f1s[idx]), float(t[idx])
    except Exception:
        pass
    return {"auroc": auroc, "auprc": auprc, "best_f1": f1_best, "best_thr": thr_best}

def multilabel_metrics(Y_true: np.ndarray, Y_prob: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> Dict:
    assert Y_true.shape == Y_prob.shape
    N, L = Y_true.shape
    if valid_mask is None:
        valid_mask = np.ones(L, dtype=bool)

    per_label = []
    aurocs, auprcs = [], []
    for j in range(L):
        yj = Y_true[:, j]; pj = Y_prob[:, j]
        aj = _safe_auroc(yj, pj) if valid_mask[j] else None
        bj = _safe_auprc(yj, pj) if valid_mask[j] else None
        per_label.append({"auroc": aj, "auprc": bj})
        if aj is not None: aurocs.append(aj)
        if bj is not None: auprcs.append(bj)

    macro_auroc = float(np.mean(aurocs)) if len(aurocs) else None
    macro_auprc = float(np.mean(auprcs)) if len(auprcs) else None

    # micro over only valid heads
    if valid_mask.any():
        y_flat = Y_true[:, valid_mask].reshape(-1)
        p_flat = Y_prob[:, valid_mask].reshape(-1)
        micro_auroc = _safe_auroc(y_flat, p_flat)
        micro_auprc = _safe_auprc(y_flat, p_flat)
    else:
        micro_auroc = micro_auprc = None

    return {
        "per_label": per_label,
        "macro_auroc": macro_auroc,
        "macro_auprc": macro_auprc,
        "micro_auroc": micro_auroc,
        "micro_auprc": micro_auprc,
    }

def compute_metrics_dict(mode: str, y_true, y_prob, valid_mask: Optional[np.ndarray] = None) -> Dict:
    m = mode.lower()
    if m == "binary":
        return binary_metrics(y_true, y_prob)
    else:
        return multilabel_metrics(y_true, y_prob, valid_mask=valid_mask)
