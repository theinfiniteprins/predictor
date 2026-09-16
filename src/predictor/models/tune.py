"""Hyperparameter search for the primary model (Optuna).

Objective: the day-block bootstrap lower bound on how far directional precision beats
the naive "always name the majority direction" baseline. Two things this deliberately
is NOT:

  * not raw precision - a model that fires twice and gets both right would win;
  * not a Wilson/binomial bound on rows - rolling entries overlap so heavily that
    ~1500 rows carry ~115 independent observations, so a row-wise interval is ~2-3x
    too narrow. Optimising it hands the search a metric it can win by exploiting the
    correlation structure, which is exactly how you tune your way into noise.

Beating the majority-direction baseline matters because barrier touches are
asymmetric: in a drifting market "always say down" scores well while predicting
nothing. Unlimited compute -> run many trials.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd

from ..logging_setup import get_logger
from ..validation.purged_cv import PurgedWalkForwardCV
from ..validation.significance import edge_report
from .primary import fit_walk_forward

log = get_logger("models.tune")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def wilson_lower_bound(k: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    return (centre - margin) / denom


def _objective(trial, X, y, t_entry, t_end, cv, min_fires, days, n_boot):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 200, 800, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 7, 63),
        min_child_samples=trial.suggest_int("min_child_samples", 15, 80),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 5.0, log=True),
    )
    res = fit_walk_forward(X, y, t_entry, t_end, cv=cv, params=params)
    oof = res.oof.dropna(subset=["primary_pred"])
    fired = pd.DataFrame({
        "primary_pred": oof["primary_pred"].to_numpy(),
        "label": y.loc[oof.index].to_numpy(),
        "day": days.loc[oof.index].to_numpy(),
    })
    n = int((fired["primary_pred"] != 0).sum())
    if n == 0:
        return -1.0
    rep = edge_report(fired, n_boot=n_boot, min_trades=min_fires, min_days=1)
    lo = rep.get("margin_ci", [np.nan])[0]
    if not np.isfinite(lo):
        return -1.0
    # a config that barely fires can't be trusted either - taper below min_fires
    return float(lo) * min(1.0, n / min_fires)


def tune_primary(
    X: pd.DataFrame,
    y: pd.Series,
    t_entry: pd.Series,
    t_end: pd.Series,
    n_trials: int = 100,
    cv: PurgedWalkForwardCV | None = None,
    min_fires: int = 30,
    days: pd.Series | None = None,
    n_boot: int = 400,
) -> dict:
    """``n_boot`` is deliberately lower than the reporting default - it runs inside
    every trial, and the search only needs to rank configurations, not publish a CI."""
    cv = cv or PurgedWalkForwardCV()
    if days is None:
        days = pd.Series(pd.DatetimeIndex(t_entry).normalize(), index=X.index)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: _objective(t, X, y, t_entry, t_end, cv, min_fires, days, n_boot),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    log.info("best objective %.4f with %s", study.best_value, study.best_params)
    if study.best_value <= 0:
        log.warning("no tuned configuration beat the naive baseline (best %.4f) - "
                    "keeping the search result, but treat it as unproven", study.best_value)
    return study.best_params
