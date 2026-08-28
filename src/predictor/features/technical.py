"""Intraday price/volume technical features, indexed by 5-min bar timestamp.

Two families:
  * continuous  - oscillators / momentum computed across the whole bar series
                  (RSI, MACD, Bollinger %b, ATR, realized vol, N-bar returns)
  * session-anchored - reset every day (return since open, VWAP deviation,
                  distance from the running intraday high/low, bars since open)

All causal: row t uses only bars <= t. ``build.py`` then samples, for each entry
timestamp T, the last bar strictly before T.

NOTE: the yfinance index feed carries volume == 0, so volume-derived columns will
be NaN for ^NSEI. They are still emitted (a liquid proxy can be swapped in later).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_setup import get_logger
from . import _indicators as ind

log = get_logger("features.technical")

_RET_BARS = (1, 2, 3, 6, 12)
_RV_BARS = (12, 24, 48)


def _typical(bars: pd.DataFrame) -> pd.Series:
    return (bars["high"] + bars["low"] + bars["close"]) / 3.0


def _session_anchored(bars: pd.DataFrame) -> pd.DataFrame:
    g = bars.groupby(bars.index.date, group_keys=False)
    out = pd.DataFrame(index=bars.index)

    out["ret_since_open"] = bars["close"] / g["open"].transform("first") - 1.0
    out["bars_since_open"] = g.cumcount()

    day_high = g["high"].cummax()
    day_low = g["low"].cummin()
    out["dist_from_day_high"] = bars["close"] / day_high - 1.0
    out["dist_from_day_low"] = bars["close"] / day_low - 1.0

    typ = _typical(bars)
    vol = bars["volume"].fillna(0.0)
    has_vol = g["volume"].transform("sum") > 0
    cum_pv = (typ * vol).groupby(bars.index.date).cumsum()
    cum_v = vol.groupby(bars.index.date).cumsum().replace(0.0, np.nan)
    vwap = cum_pv / cum_v
    sess_avg = typ.groupby(bars.index.date).transform(lambda s: s.expanding().mean())
    ref = vwap.where(has_vol, sess_avg)
    out["vwap_dev"] = bars["close"] / ref - 1.0

    rng = bars["high"] - bars["low"]
    out["range_expansion"] = rng / rng.rolling(20, min_periods=5).mean()
    return out


def _continuous(bars: pd.DataFrame) -> pd.DataFrame:
    close = bars["close"]
    out = pd.DataFrame(index=bars.index)

    for n in _RET_BARS:
        out[f"ret_{n}b"] = ind.rolling_return(close, n)
    for n in _RV_BARS:
        out[f"rv_{n}b"] = ind.realized_vol(close, n)

    out["rsi_14"] = ind.rsi(close, 14)
    out = out.join(ind.macd(close))
    out["bb_pctb"] = ind.bollinger_pctb(close, 20, 2.0)
    out["atr_14_rel"] = ind.atr(bars["high"], bars["low"], close, 14) / close

    vol = bars["volume"].replace(0.0, np.nan)
    out["vol_ratio_20"] = vol / vol.rolling(20, min_periods=10).mean()
    return out


def _cross_asset(bars: pd.DataFrame, bn: pd.DataFrame | None) -> pd.DataFrame:
    out = pd.DataFrame(index=bars.index)
    if bn is None or bn.empty:
        for c in ("bn_divergence_open", "bn_divergence_6b", "bn_corr_24b"):
            out[c] = np.nan
        return out

    bn = bn.reindex(bars.index).ffill(limit=2)
    g_n = bars.groupby(bars.index.date, group_keys=False)
    g_b = bn.groupby(bn.index.date, group_keys=False)

    n_open_ret = bars["close"] / g_n["open"].transform("first") - 1.0
    b_open_ret = bn["close"] / g_b["open"].transform("first") - 1.0
    out["bn_divergence_open"] = b_open_ret - n_open_ret

    out["bn_divergence_6b"] = (
        ind.rolling_return(bn["close"], 6) - ind.rolling_return(bars["close"], 6)
    )
    n_lr = np.log(bars["close"]).diff()
    b_lr = np.log(bn["close"]).diff()
    out["bn_corr_24b"] = n_lr.rolling(24, min_periods=12).corr(b_lr)
    return out


def technical_features(
    bars: pd.DataFrame, banknifty: pd.DataFrame | None = None
) -> pd.DataFrame:
    feats = pd.concat(
        [_continuous(bars), _session_anchored(bars), _cross_asset(bars, banknifty)],
        axis=1,
    )
    feats = feats.replace([np.inf, -np.inf], np.nan)
    log.info("technical features: %d cols over %d bars", feats.shape[1], len(feats))
    return feats
