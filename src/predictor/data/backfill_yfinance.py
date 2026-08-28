"""Track A — yfinance backfill for ^NSEI intraday/daily, Bank Nifty, and global cues.

yfinance history limits (as of 2026): ~60 days at 5m, ~7 days at 1m, decades at 1d.
All of this is re-runnable; each call fully rewrites its parquet file.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from ..calendar import IST
from ..config import CONFIG
from ..logging_setup import get_logger
from ..storage import raw_path, write_parquet

log = get_logger("backfill_yfinance")

_OHLCV = ["open", "high", "low", "close", "volume"]


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=2, max=30))
def _history(ticker: str, *, interval: str, period: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(
        period=period, interval=interval, auto_adjust=False, actions=False
    )
    if df is None or df.empty:
        raise RuntimeError(f"empty history for {ticker} ({interval}, {period})")
    return df


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase OHLCV, force an IST tz-aware index, drop all-NaN rows, de-dup."""
    df = df.rename(columns=str.lower)
    df = df[[c for c in _OHLCV if c in df.columns]].copy()

    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df.index = idx.tz_convert(IST)
    df.index.name = "timestamp"

    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(how="all")
    return df


def fetch_intraday(ticker: str, interval: str, period: str) -> pd.DataFrame:
    df = _normalize(_history(ticker, interval=interval, period=period))
    log.info("%s %s: %d bars  %s -> %s",
             ticker, interval, len(df),
             df.index.min().date() if len(df) else "-",
             df.index.max().date() if len(df) else "-")
    return df


def fetch_daily(ticker: str, period: str | None = None) -> pd.DataFrame:
    period = period or CONFIG.data.daily_period
    df = _normalize(_history(ticker, interval="1d", period=period))
    log.info("%s 1d: %d bars  %s -> %s", ticker, len(df),
             df.index.min().date() if len(df) else "-",
             df.index.max().date() if len(df) else "-")
    return df


def backfill_instrument() -> dict[str, int]:
    """Pull the primary instrument + correlated index at every interval we use."""
    inst = CONFIG.instrument
    d = CONFIG.data
    out: dict[str, int] = {}

    jobs = [
        ("yfinance", f"{inst.name}_{d.bar_interval}", inst.yf_ticker, d.bar_interval, d.backfill_period),
        ("yfinance", f"{inst.name}_{d.fine_interval}", inst.yf_ticker, d.fine_interval, "7d"),
        ("yfinance", f"{inst.name}_1d", inst.yf_ticker, "1d", d.daily_period),
        ("yfinance", f"BANKNIFTY_{d.bar_interval}", inst.correlated_ticker, d.bar_interval, d.backfill_period),
        ("yfinance", "BANKNIFTY_1d", inst.correlated_ticker, "1d", d.daily_period),
    ]
    for source, name, ticker, interval, period in jobs:
        try:
            df = fetch_intraday(ticker, interval, period) if interval != "1d" \
                else fetch_daily(ticker, period)
            write_parquet(df, raw_path(source, name))
            out[name] = len(df)
        except Exception as exc:  # noqa: BLE001 — backfill should be resilient per-job
            log.error("failed %s (%s %s): %s", name, ticker, interval, exc)
            out[name] = 0
    return out


def backfill_global_cues() -> dict[str, int]:
    """Daily bars for S&P 500 / Nasdaq / USDINR / crude (prior-session close = feature)."""
    out: dict[str, int] = {}
    for name, ticker in CONFIG.global_cues.active().items():
        try:
            df = fetch_daily(ticker, CONFIG.data.daily_period)
            write_parquet(df, raw_path("yfinance", f"cue_{name}"))
            out[name] = len(df)
        except Exception as exc:  # noqa: BLE001
            log.error("failed global cue %s (%s): %s", name, ticker, exc)
            out[name] = 0
    return out
