"""Classification metrics, bootstrap confidence intervals, McNemar test, confusion analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    labels = np.arange(num_classes)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "per_class": {"precision": precision, "recall": recall, "f1": f1, "support": support},
    }


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def bootstrap_ci(y_true, y_pred, num_classes: int, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI over test samples for accuracy and macro-F1."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n, k = len(y_true), num_classes
    rng = np.random.default_rng(seed)
    pair = y_true * k + y_pred
    accs = np.empty(n_boot)
    f1s = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        cm = np.bincount(pair[idx], minlength=k * k).reshape(k, k)
        accs[b] = np.trace(cm) / n
        f1s[b] = macro_f1_from_confusion(cm)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "accuracy": [float(np.percentile(accs, lo)), float(np.percentile(accs, hi))],
        "macro_f1": [float(np.percentile(f1s, lo)), float(np.percentile(f1s, hi))],
        "n_boot": int(n_boot),
        "confidence": 1 - alpha,
    }


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """Paired test on the same test samples. b = A right & B wrong, c = A wrong & B right."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok = pred_a == y_true
    b_ok = pred_b == y_true
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0, "method": "no discordant pairs"}
    if b + c < 25:
        p = min(1.0, 2 * stats.binom.cdf(min(b, c), b + c, 0.5))
        return {"b": b, "c": c, "statistic": float(min(b, c)), "p_value": float(p), "method": "exact binomial"}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b": b, "c": c, "statistic": float(stat), "p_value": float(stats.chi2.sf(stat, 1)),
            "method": "chi2 with continuity correction"}


def top_confusions(cm: np.ndarray, classes, top_k: int = 15) -> pd.DataFrame:
    n_errors = int(cm.sum() - np.trace(cm))
    rows = []
    for i in range(len(classes)):
        row_total = cm[i].sum()
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                rows.append({
                    "true": classes[i],
                    "predicted": classes[j],
                    "count": int(cm[i, j]),
                    "rate_within_true_class": float(cm[i, j] / row_total) if row_total else 0.0,
                    "share_of_all_errors": float(cm[i, j] / n_errors) if n_errors else 0.0,
                })
    df = pd.DataFrame(rows, columns=["true", "predicted", "count", "rate_within_true_class", "share_of_all_errors"])
    return df.sort_values("count", ascending=False).head(top_k).reset_index(drop=True)
