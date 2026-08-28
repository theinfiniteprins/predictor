"""The rolling intraday entry grid (09:30..14:30 / 15 min per Phase 0).

Thin helpers over ``predictor.calendar.entry_points`` that also attach the traded
price (the open of the 5-min bar at each entry timestamp).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from ..calendar import entry_points, now_ist
from ..config import CONFIG
from ..data.load import load_intraday


def entry_grid(day: dt.date, bars: pd.DataFrame | None = None) -> pd.DataFrame:
    """DataFrame indexed by entry timestamp with an ``entry_price`` column for ``day``."""
    bars = load_intraday() if bars is None else bars
    day_bars = bars[bars.index.date == day]
    recs = [
        {"t_entry": ts, "entry_price": float(day_bars.loc[ts, "open"])}
        for ts in entry_points(day)
        if ts in day_bars.index
    ]
    return pd.DataFrame(recs).set_index("t_entry")


def current_entry_point(ts: pd.Timestamp | None = None) -> pd.Timestamp | None:
    """The most recent entry timestamp at or before ``ts`` (default: now IST).

    Returns None before the day's first entry point. Used by predict_today.
    """
    ts = ts or now_ist()
    grid = entry_points(ts.date())
    past = [e for e in grid if e <= ts]
    return past[-1] if past else None
