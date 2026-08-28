"""Time-of-day features for an entry timestamp."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..calendar import IST
from ..config import CONFIG


def time_of_day_features(entry_ts: pd.Timestamp) -> dict[str, float]:
    ts = entry_ts.tz_convert(IST)
    open_t = CONFIG.session.open
    open_dt = ts.normalize() + pd.Timedelta(hours=open_t.hour, minutes=open_t.minute)
    vb = CONFIG.labeling.vertical_barrier
    vbar_dt = ts.normalize() + pd.Timedelta(hours=vb.hour, minutes=vb.minute)

    mins_since_open = (ts - open_dt).total_seconds() / 60.0
    mins_to_vbar = (vbar_dt - ts).total_seconds() / 60.0
    frac = mins_since_open / CONFIG.labeling.session_minutes

    return {
        "mins_since_open": mins_since_open,
        "mins_to_vbar": mins_to_vbar,
        "session_frac": frac,
        "tod_sin": float(np.sin(2 * np.pi * frac)),
        "tod_cos": float(np.cos(2 * np.pi * frac)),
        "is_opening_30m": float(mins_since_open <= 30),
        "is_after_1430": float(ts.time() >= pd.Timestamp("14:30").time()),
    }
