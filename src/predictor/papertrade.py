"""Phase 7 - paper-trading log.

Because every feature is strictly causal (see tests/test_no_lookahead.py), the
prediction the deployed model *would* have made at entry time T can be
reconstructed exactly after the fact from the collector data. So this runs as a
catch-up job on the laptop: whenever you run it, it logs every newly-resolved
entry point with the call the current model makes, plus the real triple-barrier
outcome.

Rows are append-only and stamped with which model made them (`model_trained_at`).
Entries on or before that model's `train_data_end` are marked ``in_sample`` and
excluded from the headline - only genuinely out-of-sample calls count as
paper-trading evidence.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from .calendar import now_ist
from .config import CONFIG
from .dataset import build as build_dataset
from .logging_setup import get_logger
from .models.meta import confidence_features
from .models.persist import _CARD, load_bundle
from .models.primary import PROBA_COLS, _LABELS

log = get_logger("papertrade")

_LOG = CONFIG.paths.reports_dir / "paper_trades.parquet"

_KEEP = ["day", "t_touch", "label", "entry_price", "upper", "lower",
         "sigma_effective", "ret_at_touch", "reason"]


def _load_log() -> pd.DataFrame:
    if _LOG.exists():
        return pd.read_parquet(_LOG)
    return pd.DataFrame()


def run(since: dt.date | None = None) -> pd.DataFrame:
    bundle = load_bundle()
    card = json.loads(_CARD.read_text(encoding="utf-8")) if _CARD.exists() else {}
    train_end = pd.Timestamp(card["train_data_end"]) if card.get("train_data_end") else None

    ds = build_dataset()                       # refreshes unified history + labels + features
    ds = ds[ds["label"].notna()].copy()
    resolved = ds[pd.to_datetime(ds["t_touch"]) < now_ist()]      # both tz-aware IST
    if since:
        resolved = resolved[resolved["day"] >= pd.Timestamp(since)]

    existing = _load_log()
    todo = resolved[~resolved.index.isin(existing.index)] if not existing.empty else resolved
    if todo.empty:
        log.info("paper log up to date (%d rows)", len(existing))
        return existing

    feat_cols = bundle["feature_columns"]
    X = todo.reindex(columns=feat_cols).astype("float64")
    proba = bundle["primary"].predict_proba(X.to_numpy(na_value=np.nan))
    pred = _LABELS[np.argmax(proba, axis=1)]

    proba_df = pd.DataFrame(proba, columns=PROBA_COLS, index=todo.index)
    if bundle.get("meta") is not None and bundle.get("fire_threshold") is not None:
        meta_in = pd.concat([X.reset_index(drop=True),
                             confidence_features(proba_df).reset_index(drop=True)], axis=1)
        score = bundle["meta"].predict_proba(meta_in.to_numpy(na_value=np.nan))[:, 1]
        thr = float(bundle["fire_threshold"])
        fired = (pred != 0) & (score >= thr)
    else:
        score = np.full(len(todo), np.nan)
        thr = np.nan
        fired = np.zeros(len(todo), dtype=bool)

    new = pd.DataFrame(index=todo.index)
    new["logged_at"] = now_ist().isoformat()
    new["model_trained_at"] = card.get("trained_at")
    new["in_sample"] = False if train_end is None else (todo["day"] <= train_end).to_numpy()
    new["primary_pred"] = pred
    new[PROBA_COLS] = proba
    new["meta_score"] = score
    new["fire_threshold"] = thr
    new["fired"] = fired
    new["label"] = todo["label"].astype(int).to_numpy()
    new["correct"] = (pred == todo["label"].to_numpy())
    for c in ("day", "t_touch", "entry_price", "upper", "lower", "sigma_effective",
              "ret_at_touch", "reason"):
        new[c] = todo[c].to_numpy()

    out = pd.concat([existing, new]).sort_index()
    out = out[~out.index.duplicated(keep="first")]      # never rewrite a logged call
    _LOG.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(_LOG)
    log.info("paper log: +%d rows (%d total, %d out-of-sample, %d fired)",
             len(new), len(out), int((~out["in_sample"]).sum()), int(out["fired"].sum()))
    return out


def summarize() -> dict:
    log_df = _load_log()
    if log_df.empty:
        return {"status": "no paper trades logged yet"}

    oos = log_df[~log_df["in_sample"].fillna(False)]
    fired = oos[oos["fired"].fillna(False)]
    n_days = int(oos["day"].nunique()) if len(oos) else 0

    out = {
        "rows_total": len(log_df),
        "rows_out_of_sample": len(oos),
        "oos_date_range": [str(oos["day"].min()), str(oos["day"].max())] if len(oos) else None,
        "oos_directional_precision": _dir_prec(oos),
        "fired": {
            "n": len(fired),
            "precision": float(fired["correct"].mean()) if len(fired) else None,
            "per_day": len(fired) / n_days if n_days else 0.0,
            "hit_up": int(((fired["primary_pred"] == 1) & fired["correct"]).sum()),
            "hit_down": int(((fired["primary_pred"] == -1) & fired["correct"]).sum()),
            "avg_ret_dir_adj": float((fired["primary_pred"] * fired["ret_at_touch"]).mean())
            if len(fired) else None,
        },
    }
    bt = CONFIG.paths.reports_dir / "backtest.json"
    if bt.exists():
        h = json.loads(bt.read_text(encoding="utf-8")).get("headline", {})
        out["backtest_reference"] = {
            "directional_precision": h.get("directional_precision"),
            "trades_per_day": h.get("trades_per_day"),
        }
    return out


def _dir_prec(df: pd.DataFrame) -> dict:
    d = df[df["primary_pred"] != 0]
    return {"n": len(d), "precision": float(d["correct"].mean()) if len(d) else None}
