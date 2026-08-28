"""End-to-end training/validation orchestration (Phases 4–6).

Kept out of the CLI script so it can be imported and tested.
"""

from __future__ import annotations

import json

import pandas as pd

from .config import CONFIG
from .dataset import load as load_dataset, split_xy
from .logging_setup import get_logger
from .models.meta import fit_final_meta, fit_walk_forward_meta
from .models.persist import save_bundle
from .models.primary import fit_final, fit_walk_forward
from .models.tune import tune_primary
from .validation.metrics import summarize
from .validation.purged_cv import PurgedWalkForwardCV

log = get_logger("pipeline")

_CARRY = ["day", "t_touch", "entry_price", "sigma_effective", "k", "upper", "lower",
          "touch_price", "ret_at_touch", "reason"]


def train_and_validate(n_trials: int = 0) -> pd.DataFrame:
    df = load_dataset()
    X, y, t_entry, t_end = split_xy(df)
    cv = PurgedWalkForwardCV()
    n_folds = cv.get_n_splits(t_entry, t_end)
    log.info("dataset %d rows / %d days / %d walk-forward folds", len(df), df["day"].nunique(), n_folds)
    if n_folds == 0:
        raise RuntimeError("no walk-forward folds - need a longer history (keep the collector running)")

    params = {}
    if n_trials:
        log.info("tuning primary over %d trials ...", n_trials)
        params = tune_primary(X, y, t_entry, t_end, n_trials=n_trials, cv=cv)

    primary = fit_walk_forward(X, y, t_entry, t_end, cv=cv, params=params)
    meta_score = fit_walk_forward_meta(X, y, primary.oof, t_entry, t_end, cv=cv)

    oof = df[_CARRY].copy()
    oof["label"] = y
    oof["primary_pred"] = primary.oof["primary_pred"]
    oof[["proba_dn", "proba_to", "proba_up"]] = primary.oof[["proba_dn", "proba_to", "proba_up"]]
    oof["meta_score"] = meta_score

    summary = summarize(oof, CONFIG.meta.fire_top_fraction)
    log.info("OOF directional precision: %s", summary["directional"])
    log.info("OOF high-confidence (top %.0f%%): %s",
             100 * CONFIG.meta.fire_top_fraction, summary["high_confidence"])

    reports = CONFIG.paths.reports_dir / "cv"
    reports.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(reports / "oof.parquet", engine="pyarrow")
    (reports / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (reports / "fold_metrics.json").write_text(
        json.dumps(primary.fold_metrics, indent=2, default=str), encoding="utf-8")
    primary.feature_importance.to_csv(reports / "feature_importance.csv")

    dir_scores = oof.loc[oof["primary_pred"].fillna(0) != 0, "meta_score"].dropna()
    fire_threshold = (
        float(dir_scores.quantile(1 - CONFIG.meta.fire_top_fraction))
        if len(dir_scores) >= 10 else None
    )

    final_primary = fit_final(X, y, params=params)
    final_meta = fit_final_meta(X, y, primary.oof, params=None)
    save_bundle(final_primary, final_meta, list(X.columns), params, summary, fire_threshold)

    return oof
