"""Save / load the trained model bundle for live prediction."""

from __future__ import annotations

import json
from datetime import datetime

import joblib

from ..calendar import IST, now_ist
from ..config import CONFIG
from ..logging_setup import get_logger

log = get_logger("models.persist")

_BUNDLE = CONFIG.paths.models_dir / "bundle.joblib"
_CARD = CONFIG.paths.models_dir / "model_card.json"


def save_bundle(
    primary, meta, feature_columns: list[str], params: dict, metrics: dict,
    fire_threshold: float | None = None, train_data_end: str | None = None,
) -> None:
    CONFIG.paths.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"primary": primary, "meta": meta, "feature_columns": feature_columns,
         "params": params, "fire_threshold": fire_threshold,
         "train_data_end": train_data_end},
        _BUNDLE,
    )
    _CARD.write_text(json.dumps({
        "trained_at": now_ist().isoformat(),
        "train_data_end": train_data_end,
        "params": params,
        "metrics": metrics,
        "k": CONFIG.labeling.k,
        "barrier_time_scaling": CONFIG.labeling.barrier_time_scaling,
        "fire_top_fraction": CONFIG.meta.fire_top_fraction,
        "fire_threshold": fire_threshold,
    }, indent=2, default=str), encoding="utf-8")
    log.info("saved model bundle -> %s  (fire_threshold=%s)", _BUNDLE, fire_threshold)


def load_bundle() -> dict:
    if not _BUNDLE.exists():
        raise FileNotFoundError(f"{_BUNDLE} missing - run `python scripts/train.py`")
    return joblib.load(_BUNDLE)
