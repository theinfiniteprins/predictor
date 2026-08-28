"""Scoring focused on what this system is for: precision on the rare fired calls.

Blended accuracy is deliberately not the headline - a model that always predicts
"timeout" scores ~90% accuracy here and is useless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CLASSES = (-1, 0, 1)


def per_class_report(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = {}
    for c in CLASSES:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) else np.nan
        rec = tp / (tp + fn) if (tp + fn) else np.nan
        rows[c] = {
            "precision": prec,
            "recall": rec,
            "support": int(np.sum(y_true == c)),
            "predicted": int(np.sum(y_pred == c)),
        }
    return pd.DataFrame(rows).T


def directional_precision(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Among rows where the model predicted a direction (+/-1), how often right."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fired = y_pred != 0
    if not fired.any():
        return {"n_fired": 0, "precision": np.nan}
    correct = y_pred[fired] == y_true[fired]
    return {"n_fired": int(fired.sum()), "precision": float(correct.mean())}


def high_confidence_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: np.ndarray,
    top_fraction: float,
    n_days: int | None = None,
) -> dict:
    """Precision on the top-``top_fraction`` most-confident *directional* calls.

    ``score`` is the meta-model's P(primary is right). This is THE headline number.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    score = np.asarray(score, dtype="float64")

    cand = np.where((y_pred != 0) & np.isfinite(score))[0]
    if cand.size == 0:
        return {"n_fired": 0, "precision": np.nan, "threshold": np.nan, "fires_per_day": 0.0}

    k = max(1, int(np.ceil(cand.size * top_fraction)))
    order = cand[np.argsort(score[cand])[::-1]]
    fired = order[:k]
    thr = float(score[order[k - 1]])
    correct = y_pred[fired] == y_true[fired]
    out = {
        "n_fired": int(k),
        "precision": float(correct.mean()),
        "threshold": thr,
        "hit_up": int(np.sum((y_pred[fired] == 1) & correct)),
        "hit_down": int(np.sum((y_pred[fired] == -1) & correct)),
    }
    if n_days:
        out["fires_per_day"] = k / n_days
    return out


def summarize(oof: pd.DataFrame, top_fraction: float) -> dict:
    """`oof` has columns: label, primary_pred, meta_score, day."""
    valid = oof.dropna(subset=["label", "primary_pred"])
    y_true = valid["label"].to_numpy()
    y_pred = valid["primary_pred"].to_numpy()
    n_days = valid["day"].nunique() if "day" in valid else None
    return {
        "n_samples": len(valid),
        "n_days": n_days,
        "per_class": per_class_report(y_true, y_pred).to_dict(),
        "directional": directional_precision(y_true, y_pred),
        "high_confidence": high_confidence_precision(
            y_true, y_pred, valid.get("meta_score", pd.Series(np.nan, index=valid.index)).to_numpy(),
            top_fraction, n_days,
        ),
    }
