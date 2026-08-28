"""Meta-model - "should we trust this particular primary call?"

Trained only on entries where the primary predicted a direction (+/-1). Target is
binary: did the primary get it right. Features = the model features plus the
primary's own confidence signals (predicted-class probability, top-2 margin,
entropy). Its out-of-fold score is the confidence we rank on and fire the top
slice of.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from ..logging_setup import get_logger
from ..validation.purged_cv import PurgedWalkForwardCV
from .primary import PROBA_COLS

log = get_logger("models.meta")


def default_params() -> dict:
    return dict(
        objective="binary",
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.0,
        n_jobs=-1,
        verbosity=-1,
    )


def confidence_features(proba: pd.DataFrame) -> pd.DataFrame:
    p = proba[PROBA_COLS].to_numpy(dtype="float64")
    p = np.clip(p, 1e-9, 1.0)
    srt = np.sort(p, axis=1)
    return pd.DataFrame({
        "conf_max": srt[:, -1],
        "conf_margin": srt[:, -1] - srt[:, -2],
        "conf_entropy": -(p * np.log(p)).sum(axis=1),
        "conf_dir_vs_to": srt[:, -1] - p[:, 1],  # winning dir proba minus timeout proba
    }, index=proba.index)


def _meta_frame(X: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([X, confidence_features(oof)], axis=1)


def fit_walk_forward_meta(
    X: pd.DataFrame,
    y_true: pd.Series,
    primary_oof: pd.DataFrame,
    t_entry: pd.Series,
    t_end: pd.Series,
    cv: PurgedWalkForwardCV | None = None,
    params: dict | None = None,
    min_fold_train: int = 25,
) -> pd.Series:
    """Return a meta_score Series aligned to X (NaN where primary predicted timeout)."""
    cv = cv or PurgedWalkForwardCV()
    params = {**default_params(), **(params or {})}

    meta_X = _meta_frame(X, primary_oof)
    pred = primary_oof["primary_pred"]
    correct = (pred == y_true).astype("float64")
    directional = pred.fillna(0) != 0

    score = pd.Series(np.nan, index=X.index, name="meta_score")
    Xv = meta_X.to_numpy(dtype="float64", na_value=np.nan)
    yv = correct.to_numpy()
    dir_mask = directional.to_numpy()

    n = 0
    for tr, te in cv.split(t_entry, t_end):
        tr = tr[dir_mask[tr] & np.isfinite(yv[tr])]
        te = te[dir_mask[te]]
        if len(tr) < min_fold_train or len(te) == 0:
            continue
        if len(np.unique(yv[tr])) < 2:
            continue
        model = LGBMClassifier(**params)
        model.fit(Xv[tr], yv[tr])
        score.iloc[te] = model.predict_proba(Xv[te])[:, 1]
        n += 1

    log.info("meta: %d folds, %d scored directional entries", n, score.notna().sum())
    return score


def fit_final_meta(
    X: pd.DataFrame, y_true: pd.Series, primary_oof: pd.DataFrame, params: dict | None = None
) -> LGBMClassifier | None:
    params = {**default_params(), **(params or {})}
    meta_X = _meta_frame(X, primary_oof)
    pred = primary_oof["primary_pred"]
    mask = (pred.fillna(0) != 0) & pred.notna() & y_true.notna()
    y = (pred[mask] == y_true[mask]).astype(int)
    if y.nunique() < 2 or mask.sum() < 40:
        log.warning("not enough directional history to fit a final meta-model")
        return None
    model = LGBMClassifier(**params)
    model.fit(meta_X[mask].to_numpy(dtype="float64", na_value=np.nan), y.to_numpy())
    return model
