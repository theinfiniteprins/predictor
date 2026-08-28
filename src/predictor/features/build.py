"""Assemble the feature matrix, one row per rolling entry point.

For entry timestamp T on day D:
  * intraday technical features  -> the last 5-min bar STRICTLY before T
  * daily context / global cues  -> row for day D (already lag-shifted at source)
  * time-of-day features         -> computed directly from T

No column may use information dated at or after T. That invariant is tested in
tests/test_no_lookahead.py.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from ..data.load import load_daily, load_intraday
from ..logging_setup import get_logger
from ..calendar import entry_points
from .context import context_features
from .global_cues import global_cue_features
from .technical import technical_features
from .time_of_day import time_of_day_features

log = get_logger("features.build")


def _entry_index(bars: pd.DataFrame, days: list[dt.date]) -> pd.DatetimeIndex:
    stamps: list[pd.Timestamp] = []
    for d in days:
        day_bars = bars[bars.index.date == d]
        if day_bars.empty:
            continue
        stamps += [ts for ts in entry_points(d) if ts in day_bars.index]
    return pd.DatetimeIndex(sorted(stamps), name="t_entry")


def build_features(
    bars: pd.DataFrame | None = None,
    banknifty: pd.DataFrame | None = None,
    days: list[dt.date] | None = None,
) -> pd.DataFrame:
    bars = load_intraday() if bars is None else bars
    if banknifty is None:
        try:
            banknifty = load_intraday(instrument="BANKNIFTY")
        except FileNotFoundError:
            banknifty = None
    days = days or sorted({ts.date() for ts in bars.index})

    entry_idx = _entry_index(bars, days)
    entries = pd.DataFrame({"t_entry": entry_idx})
    entries["t_entry"] = entries["t_entry"].dt.as_unit("ns")
    entries["day"] = entries["t_entry"].dt.tz_localize(None).dt.normalize()

    # --- intraday technical: strictly-before-T asof join ---
    tech = technical_features(bars, banknifty).reset_index()
    tech = tech.rename(columns={tech.columns[0]: "bar_ts"}).sort_values("bar_ts")
    tech["bar_ts"] = tech["bar_ts"].dt.as_unit("ns")
    feat = pd.merge_asof(
        entries.sort_values("t_entry"),
        tech,
        left_on="t_entry",
        right_on="bar_ts",
        direction="backward",
        allow_exact_matches=False,
    ).drop(columns="bar_ts")

    # --- daily context + global cues: join on day D ---
    ctx = context_features(load_daily())
    feat = feat.merge(ctx, left_on="day", right_index=True, how="left")

    cues = global_cue_features(pd.DatetimeIndex(entries["day"].unique()))
    feat = feat.merge(cues, left_on="day", right_index=True, how="left")

    # --- time-of-day: per entry ---
    tod = pd.DataFrame(
        [time_of_day_features(ts) for ts in feat["t_entry"]], index=feat.index
    )
    feat = pd.concat([feat, tod], axis=1)

    feat = feat.set_index("t_entry").drop(columns="day").sort_index()

    dead = [c for c in feat.columns if feat[c].isna().all()]
    if dead:
        log.warning("dropping all-NaN feature columns: %s", dead)
        feat = feat.drop(columns=dead)

    log.info("feature matrix: %d rows x %d cols", *feat.shape)
    return feat


FEATURE_METADATA_EXCLUDE = {
    "label", "day", "t_touch", "entry_price", "sigma_intraday", "sigma_effective",
    "k", "upper", "lower", "touch_price", "ret_at_touch", "bars_held", "reason",
}


def feature_columns(dataset: pd.DataFrame) -> list[str]:
    """Model input columns = everything that isn't a label/bookkeeping field."""
    return [c for c in dataset.columns if c not in FEATURE_METADATA_EXCLUDE]
