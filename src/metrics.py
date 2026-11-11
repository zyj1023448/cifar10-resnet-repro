"""Evaluation metrics implemented with numpy/torch only (no sklearn dependency)."""
from __future__ import annotations

import numpy as np


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float((logits.argmax(1) == labels).mean())


def confusion_matrix(logits: np.ndarray, labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    pred = logits.argmax(1)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels, pred):
        cm[t, p] += 1
    return cm


def macro_f1(logits: np.ndarray, labels: np.ndarray, num_classes: int = 10) -> float:
    cm = confusion_matrix(logits, labels, num_classes)
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = 2 * tp + fp + fn
        f1s.append(0.0 if denom == 0 else 2 * tp / denom)
    return float(np.mean(f1s))


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(logits: np.ndarray, labels: np.ndarray) -> float:
    p = softmax(logits)
    return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, None)).mean())


def ece(logits: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins."""
    p = softmax(logits)
    conf = p.max(1)
    pred = p.argmax(1)
    correct = (pred == labels).astype(np.float64)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def brier(logits: np.ndarray, labels: np.ndarray, num_classes: int = 10) -> float:
    p = softmax(logits)
    onehot = np.eye(num_classes)[labels]
    return float(((p - onehot) ** 2).sum(1).mean())


def per_class_accuracy(logits: np.ndarray, labels: np.ndarray, num_classes: int = 10) -> list[float]:
    cm = confusion_matrix(logits, labels, num_classes)
    return [float(cm[c, c] / max(cm[c].sum(), 1)) for c in range(num_classes)]


def mcnemar(logits_a: np.ndarray, logits_b: np.ndarray, labels: np.ndarray) -> dict:
    """McNemar test between two classifiers on the same test set (continuity corrected)."""
    ca = logits_a.argmax(1) == labels
    cb = logits_b.argmax(1) == labels
    n01 = int(np.sum(ca & ~cb))  # A right, B wrong
    n10 = int(np.sum(~ca & cb))  # A wrong, B right
    stat = (abs(n01 - n10) - 1) ** 2 / max(n01 + n10, 1)
    # survival function of chi-square with 1 dof = erfc(sqrt(stat/2))
    from math import erfc, sqrt
    p = erfc(sqrt(stat / 2.0)) if n01 + n10 > 0 else 1.0
    return {"n01": n01, "n10": n10, "chi2": float(stat), "p_value": float(p)}
