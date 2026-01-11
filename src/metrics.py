from __future__ import annotations

from typing import Dict, Iterable, List, Tuple
import numpy as np


def precision_recall_lift_at_k(y_true: np.ndarray, y_score: np.ndarray, k_frac: float) -> Dict[str, float]:
    """Baseline, Precision@k, Recall@k, Lift@k. k_frac in (0,1]."""
    n = len(y_true)
    k = max(1, int(np.ceil(k_frac * n)))
    order = np.argsort(-y_score)
    topk = order[:k]

    baseline = float(np.mean(y_true))
    precision_k = float(np.mean(y_true[topk])) if k > 0 else 0.0
    positives = float(np.sum(y_true))
    recall_k = float(np.sum(y_true[topk]) / positives) if positives > 0 else 0.0
    lift_k = float(precision_k / baseline) if baseline > 0 else 0.0

    return {
        "baseline": baseline,
        "precision_k": precision_k,
        "recall_k": recall_k,
        "lift_k": lift_k,
        "k": k,
    }


def lift_curve(y_true: np.ndarray, y_score: np.ndarray, ks: Iterable[float]) -> List[Tuple[float, float]]:
    """Return list of (k_frac, lift@k)."""
    out = []
    for k in ks:
        m = precision_recall_lift_at_k(y_true, y_score, float(k))
        out.append((float(k), float(m["lift_k"])))
    return out
