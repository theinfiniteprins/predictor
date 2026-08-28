"""Daily market-context features: one row per trading day, holding only information
known *before* that day's session (prior-day values are shifted by one).

Columns
  prior_day_return, prior_day_range, gap_prev            price action into the day
  rv_5d / rv_10d / rv_20d                                 daily realized vol
  dist_sma20 / dist_sma50                                 trend position
  atr14_daily_rel                                         the barrier-sigma regime
  vol_pctile_1y                                           where rv_20d sits in its trailing year
  india_vix / india_vix_chg                               volatility regime (prior close)
  dow / dom / month                                       seasonality (known in advance)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.load import load_cue, load_daily, load_vix_history
from ..logging_setup import get_logger

log = get_logger("features.context")


def _to_date_index(df: pd.DataFrame | pd.Series):
    idx = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    out = df.copy()
    out.index = idx
    return out[~out.index.duplicated(keep="last")].sort_index()


def context_features(daily: pd.DataFrame | None = None) -> pd.DataFrame:
    daily = _to_date_index(load_daily() if daily is None else daily)
    close = daily["close"]
    ret = close.pct_change()

    feat = pd.DataFrame(index=daily.index)
    feat["prior_day_return"] = ret
    feat["prior_day_range"] = (daily["high"] - daily["low"]) / close
    feat["gap_prev"] = daily["open"] / close.shift(1) - 1.0

    feat["rv_5d"] = ret.rolling(5, min_periods=5).std()
    feat["rv_10d"] = ret.rolling(10, min_periods=10).std()
    feat["rv_20d"] = ret.rolling(20, min_periods=20).std()

    feat["dist_sma20"] = close / close.rolling(20, min_periods=20).mean() - 1.0
    feat["dist_sma50"] = close / close.rolling(50, min_periods=50).mean() - 1.0

    prev_close = close.shift(1)
    tr = pd.concat(
        [daily["high"] - daily["low"],
         (daily["high"] - prev_close).abs(),
         (daily["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    feat["atr14_daily_rel"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / close

    feat["vol_pctile_1y"] = feat["rv_20d"].rolling(252, min_periods=60).rank(pct=True)

    # everything above is same-day-close info -> shift so row D holds D-1 values
    feat = feat.shift(1)

    # India VIX (prior close)
    vix = _to_date_index(load_vix_history().to_frame())["india_vix"]
    if not vix.empty:
        vix = vix.reindex(feat.index).ffill()
        feat["india_vix"] = vix.shift(1)
        feat["india_vix_chg"] = vix.pct_change().shift(1)
    else:
        feat["india_vix"] = np.nan
        feat["india_vix_chg"] = np.nan

    # seasonality - from the day itself, no shift
    feat["dow"] = feat.index.dayofweek
    feat["dom"] = feat.index.day
    feat["month"] = feat.index.month

    feat = feat.replace([np.inf, -np.inf], np.nan)
    log.info("context features: %d cols over %d days", feat.shape[1], len(feat))
    return feat
