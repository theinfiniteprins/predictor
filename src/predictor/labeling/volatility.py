"""sigma_intraday - the per-day volatility that scales the triple barriers.

Definition (Phase 0): ATR(atr_window) on *daily* bars, divided by close, using only
sessions strictly before the entry day. Held constant across all of that day's
entry points. No look-ahead: the value for day D depends only on daily bars <= D-1.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..config import CONFIG
from ..data.load import load_daily


def true_range(daily: pd.DataFrame) -> pd.Series:
    prev_close = daily["close"].shift(1)
    tr = pd.concat(
        [
            daily["high"] - daily["low"],
            (daily["high"] - prev_close).abs(),
            (daily["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rename("true_range")


def atr(daily: pd.DataFrame, window: int | None = None) -> pd.Series:
    window = window or CONFIG.labeling.atr_window
    return true_range(daily).rolling(window, min_periods=window).mean().rename("atr")


def sigma_series(daily: pd.DataFrame | None = None, window: int | None = None) -> pd.Series:
    """ATR(window)/close per session, as a fractional volatility. Indexed by date."""
    daily = load_daily() if daily is None else daily
    window = window or CONFIG.labeling.atr_window
    s = (atr(daily, window) / daily["close"]).rename("sigma_intraday")
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    s.index.name = "date"
    return s.dropna()


def sigma_for_day(day: dt.date, sigma: pd.Series | None = None) -> float:
    """sigma_intraday to use for entries on ``day`` = the value from the last session < day."""
    sigma = sigma_series() if sigma is None else sigma
    prior = sigma.loc[sigma.index < pd.Timestamp(day)]
    if prior.empty:
        return float("nan")
    return float(prior.iloc[-1])


def effective_sigma(sigma_daily: float, entry_ts: pd.Timestamp, vertical_ts: pd.Timestamp) -> float:
    """Barrier-scaling sigma for a specific entry.

    With ``barrier_time_scaling`` on, shrink the full-session sigma by
    sqrt(minutes_to_vertical / session_minutes) - a mid-session entry has less
    time for the move, so its barriers should be tighter. Off -> return as-is.
    """
    if not CONFIG.labeling.barrier_time_scaling:
        return sigma_daily
    minutes_left = max((vertical_ts - entry_ts).total_seconds() / 60.0, 0.0)
    return sigma_daily * np.sqrt(minutes_left / CONFIG.labeling.session_minutes)
