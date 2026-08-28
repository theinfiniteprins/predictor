"""Primary model - LightGBM multiclass over {-1, 0, +1}.

Produces out-of-fold predictions via the purged walk-forward splitter. Those OOF
predictions (never seen during their own training) are what the meta-model and all
reported metrics are built on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from ..logging_setup import get_logger
from ..validation.purged_cv import PurgedWalkForwardCV

log = get_logger("models.primary")

_LABELS = np.array([-1, 0, 1])
_TO_IDX = {-1: 0, 0: 1, 1: 2}
PROBA_COLS = ["proba_dn", "proba_to", "proba_up"]


def default_params() -> dict:
    return dict(
        objective="multiclass",
        num_class=3,
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        max_depth=-1,
        n_jobs=-1,
        verbosity=-1,
    )


def _balanced_weights(y: np.ndarray) -> np.ndarray:
    w = np.ones(len(y), dtype="float64")
    for c in np.unique(y):
        w[y == c] = len(y) / (len(np.unique(y)) * np.sum(y == c))
    return w


@dataclass
class PrimaryResult:
    oof: pd.DataFrame
    fold_metrics: list[dict] = field(default_factory=list)
    feature_importance: pd.Series | None = None


def fit_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    t_entry: pd.Series,
    t_end: pd.Series,
    cv: PurgedWalkForwardCV | None = None,
    params: dict | None = None,
) -> PrimaryResult:
    cv = cv or PurgedWalkForwardCV()
    params = {**default_params(), **(params or {})}

    y_idx = y.map(_TO_IDX).to_numpy()
    Xv = X.to_numpy(dtype="float64", na_value=np.nan)

    oof = pd.DataFrame(index=X.index, columns=PROBA_COLS + ["primary_pred", "fold"], dtype="float64")
    importances = np.zeros(X.shape[1])
    fold_metrics: list[dict] = []

    n = 0
    for fold, (tr, te) in enumerate(cv.split(t_entry, t_end)):
        model = LGBMClassifier(**params)
        model.fit(Xv[tr], y_idx[tr], sample_weight=_balanced_weights(y_idx[tr]))
        proba = model.predict_proba(Xv[te])
        pred = _LABELS[np.argmax(proba, axis=1)]

        oof.iloc[te, oof.columns.get_indexer(PROBA_COLS)] = proba
        oof.iloc[te, oof.columns.get_loc("primary_pred")] = pred
        oof.iloc[te, oof.columns.get_loc("fold")] = fold
        importances += model.feature_importances_

        y_te = _LABELS[y_idx[te]]
        fired = pred != 0
        fold_metrics.append({
            "fold": fold, "n_train": len(tr), "n_test": len(te),
            "n_fired": int(fired.sum()),
            "dir_precision": float((pred[fired] == y_te[fired]).mean()) if fired.any() else np.nan,
        })
        n += 1

    log.info("primary: %d walk-forward folds, %d OOF predictions", n, oof["primary_pred"].notna().sum())
    return PrimaryResult(
        oof=oof,
        fold_metrics=fold_metrics,
        feature_importance=pd.Series(importances, index=X.columns).sort_values(ascending=False),
    )


def fit_final(X: pd.DataFrame, y: pd.Series, params: dict | None = None) -> LGBMClassifier:
    """Fit on ALL data - the model used for live prediction."""
    params = {**default_params(), **(params or {})}
    y_idx = y.map(_TO_IDX).to_numpy()
    model = LGBMClassifier(**params)
    model.fit(X.to_numpy(dtype="float64", na_value=np.nan), y_idx, sample_weight=_balanced_weights(y_idx))
    return model
