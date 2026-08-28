"""Build / load the modelling dataset: features joined to triple-barrier labels,
one row per rolling entry point.
"""

from __future__ import annotations

import json

import pandas as pd

from .config import CONFIG
from .features.build import build_features, feature_columns
from .labeling.triple_barrier import build_labels
from .logging_setup import get_logger

log = get_logger("dataset")

_PATH = CONFIG.paths.processed / "dataset.parquet"
_META = CONFIG.paths.processed / "dataset_meta.json"


def build(k: float | None = None, consolidate: bool = True) -> pd.DataFrame:
    k = CONFIG.labeling.k if k is None else k

    if consolidate:
        from .data.consolidate import consolidate_all
        consolidate_all()   # merge collector data into the growing unified history

    labels = build_labels(k=k)
    feats = build_features()

    df = feats.join(labels, how="inner")
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    df["day"] = pd.to_datetime(df["day"])          # normalize date -> datetime64
    df["t_touch"] = pd.to_datetime(df["t_touch"])
    df = df.sort_index()

    _PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_PATH, engine="pyarrow")
    _META.write_text(json.dumps({
        "rows": len(df),
        "days": int(df["day"].nunique()),
        "k": k,
        "barrier_time_scaling": CONFIG.labeling.barrier_time_scaling,
        "label_counts": {str(k_): int(v) for k_, v in df["label"].value_counts().items()},
        "feature_columns": feature_columns(df),
        "date_range": [str(df.index.min()), str(df.index.max())],
    }, indent=2), encoding="utf-8")
    log.info("dataset -> %s  (%d rows, %d days, k=%s)", _PATH, len(df), df["day"].nunique(), k)
    return df


def load() -> pd.DataFrame:
    if not _PATH.exists():
        raise FileNotFoundError(f"{_PATH} missing - run `python scripts/build_dataset.py`")
    return pd.read_parquet(_PATH, engine="pyarrow")


def load_meta() -> dict:
    return json.loads(_META.read_text(encoding="utf-8"))


def split_xy(df: pd.DataFrame):
    """-> (X, y, t_entry, t_end) ready for the walk-forward fitters."""
    cols = feature_columns(df)
    X = df[cols].astype("float64")
    y = df["label"].astype(int)
    t_entry = df.index.to_series()
    t_end = df["t_touch"]
    return X, y, t_entry, t_end
