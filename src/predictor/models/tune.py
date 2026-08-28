"""Hyperparameter search for the primary model (Optuna).

Objective: the Wilson lower bound of directional precision on the purged
walk-forward OOF predictions - this rewards a model that is both accurate on the
calls it makes *and* makes enough of them, instead of a model that fires twice and
gets both right. Unlimited compute -> run many trials.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd

from ..logging_setup import get_logger
from ..validation.purged_cv import PurgedWalkForwardCV
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


def _objective(trial, X, y, t_entry, t_end, cv, min_fires):
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
    pred = oof["primary_pred"].to_numpy()
    truth = y.loc[oof.index].to_numpy()
    fired = pred != 0
    n, k = int(fired.sum()), int(np.sum(pred[fired] == truth[fired]))
    if n < min_fires:
        return wilson_lower_bound(k, n) * (n / min_fires)
    return wilson_lower_bound(k, n)


def tune_primary(
    X: pd.DataFrame,
    y: pd.Series,
    t_entry: pd.Series,
    t_end: pd.Series,
    n_trials: int = 100,
    cv: PurgedWalkForwardCV | None = None,
    min_fires: int = 30,
) -> dict:
    cv = cv or PurgedWalkForwardCV()
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: _objective(t, X, y, t_entry, t_end, cv, min_fires),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    log.info("best objective %.4f with %s", study.best_value, study.best_params)
    return study.best_params
