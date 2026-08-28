"""Phases 4–5 & 8 — modeling (not yet implemented).

Planned modules:
    primary.py   XGBoost/LightGBM multiclass {-1, 0, +1}, fit inside purged CV
    meta.py      binary "is the primary's call trustworthy?" on out-of-fold preds
    tune.py      Optuna hyperparameter search wrapped in the purged CV splitter
    ensemble.py  stacking layer (Phase 8, once the self-collected dataset is deep)
"""
