"""Overnight / global cues: prior-session moves in S&P 500, Nasdaq, USD/INR, crude.

For NSE trading day D these use the last cue observation dated <= D-1, so nothing
that prints during or after D's session leaks in. GIFT Nifty is intentionally
absent (no reliable free ticker).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CONFIG
from ..data.load import load_cue
from ..logging_setup import get_logger

log = get_logger("features.global_cues")

# india_vix is consumed in context.py; gift_nifty has no data
_SKIP = {"india_vix", "gift_nifty"}


def _date_index(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    out = df.copy()
    out.index = idx
    return out[~out.index.duplicated(keep="last")].sort_index()


def global_cue_features(target_dates: pd.DatetimeIndex) -> pd.DataFrame:
    target = pd.DatetimeIndex(pd.Series(target_dates).dt.tz_localize(None).dt.normalize().unique())
    target = target.sort_values()
    out = pd.DataFrame(index=target)

    for name, ticker in CONFIG.global_cues.active().items():
        if name in _SKIP:
            continue
        try:
            cue = _date_index(load_cue(name))
        except FileNotFoundError:
            log.warning("cue %s missing; column will be NaN", name)
            out[f"{name}_ret1d"] = np.nan
            out[f"{name}_ret5d"] = np.nan
            continue
        close = cue["close"]
        ret1 = close.pct_change()
        ret5 = close.pct_change(5)
        # align onto NSE days, carry forward, then shift 1 NSE day for safety
        out[f"{name}_ret1d"] = ret1.reindex(target, method="ffill").shift(1)
        out[f"{name}_ret5d"] = ret5.reindex(target, method="ffill").shift(1)

    out = out.replace([np.inf, -np.inf], np.nan)
    log.info("global-cue features: %d cols over %d days", out.shape[1], len(out))
    return out
