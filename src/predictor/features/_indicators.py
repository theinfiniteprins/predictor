"""Causal technical indicators - hand-rolled so we control the look-ahead behaviour.

Every function returns a Series/DataFrame aligned to the input index where row *t*
uses only rows <= t. No centered windows, no bfill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).rename("rsi")


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def bollinger_pctb(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    ma = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std()
    upper, lower = ma + n_std * sd, ma - n_std * sd
    return ((close - lower) / (upper - lower)).rename("bb_pctb")


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean().rename("atr")


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    logret = np.log(close).diff()
    return logret.rolling(window, min_periods=window).std().rename(f"rv_{window}")


def rolling_return(close: pd.Series, window: int) -> pd.Series:
    return (close / close.shift(window) - 1.0).rename(f"ret_{window}")
