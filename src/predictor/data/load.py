"""Read the raw parquet files produced by the backfill / collector into clean frames.

Everything returned here is IST tz-aware and sorted. These are the only functions
downstream code (labeling, features) should use to touch the data lake.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from ..calendar import IST
from ..config import CONFIG
from ..logging_setup import get_logger
from ..storage import raw_path, read_parquet

log = get_logger("load")

_OHLC = ["open", "high", "low", "close"]


def _load(source: str, name: str) -> pd.DataFrame:
    path = raw_path(source, name)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run `python scripts/run_backfill.py --all` first"
        )
    df = read_parquet(path)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _load_unified_or_raw(instrument: str, interval: str) -> pd.DataFrame:
    """Prefer data/interim/<inst>_<interval>_unified.parquet (yfinance + collector,
    built by predictor.data.consolidate); fall back to the raw yfinance pull."""
    uni = CONFIG.paths.interim / f"{instrument}_{interval}_unified.parquet"
    if uni.exists():
        df = read_parquet(uni)
        return df[~df.index.duplicated(keep="last")].sort_index()
    return _load("yfinance", f"{instrument}_{interval}")


def load_intraday(interval: str | None = None, *, instrument: str = "NIFTY50") -> pd.DataFrame:
    """5-min (or configured) OHLCV bars for the primary or correlated index."""
    return _load_unified_or_raw(instrument, interval or CONFIG.data.bar_interval)


def load_fine(instrument: str = "NIFTY50") -> pd.DataFrame:
    """1-min bars for intrabar barrier tie-breaking (yfinance ~7d + collector growth)."""
    return _load_unified_or_raw(instrument, CONFIG.data.fine_interval)


def load_daily(instrument: str = "NIFTY50", *, prefer: str = "yfinance") -> pd.DataFrame:
    """Daily OHLC, long history. ``prefer='nse'`` uses the jugaad-data bhavcopy pull."""
    if prefer == "nse":
        try:
            return _load("bhavcopy", f"{instrument}_1d_nse")
        except FileNotFoundError:
            log.warning("NSE daily history missing; falling back to yfinance")
    return _load("yfinance", f"{instrument}_1d")


def load_cue(name: str) -> pd.DataFrame:
    """Daily bars for a global cue (sp500 / nasdaq / usdinr / crude / india_vix)."""
    return _load("yfinance", f"cue_{name}")


def load_vix_history() -> pd.Series:
    """India VIX daily close as a Series indexed by IST date-timestamp."""
    try:
        return load_cue("india_vix")["close"].rename("india_vix")
    except FileNotFoundError:
        log.warning("India VIX history missing")
        return pd.Series(dtype="float64", name="india_vix")


def load_option_chain_snapshots() -> pd.DataFrame:
    """Appended PCR/OI snapshots (forward-collected only; empty for the backfill period)."""
    import glob

    files = sorted(glob.glob(str(CONFIG.paths.raw / "option_chain" / "option_chain_*.parquet")))
    if not files:
        return pd.DataFrame()
    df = pd.concat([read_parquet(p) for p in files], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_collected_bars(ticker: str | None = None) -> pd.DataFrame:
    """All collector 1-min bars (data/raw/live/ + collected/), de-duped.

    Returns columns [bar_time, ticker, open, high, low, close, volume], IST-aware.
    """
    import glob

    roots = [CONFIG.paths.raw_live, CONFIG.paths.root / "collected"]
    frames = []
    for root in roots:
        for p in glob.glob(str(root / "date=*" / "quotes.parquet")):
            frames.append(read_parquet(p))
    if not frames:
        return pd.DataFrame(columns=["bar_time", "ticker", *_OHLC, "volume"])

    df = pd.concat(frames, ignore_index=True)
    # tolerate the pre-consolidation schema (poll-time 'timestamp', no 'bar_time')
    if "bar_time" not in df.columns and "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "bar_time"})
    df["bar_time"] = pd.to_datetime(df["bar_time"], utc=True).dt.tz_convert(IST)
    keep = ["bar_time", "ticker", *[c for c in (*_OHLC, "volume") if c in df.columns]]
    df = (
        df[keep]
        .drop_duplicates(subset=["bar_time", "ticker"], keep="last")
        .sort_values(["bar_time", "ticker"])
        .reset_index(drop=True)
    )
    if ticker:
        df = df[df["ticker"] == ticker].reset_index(drop=True)
    return df


def trading_days_in_bars(bars: pd.DataFrame) -> list[dt.date]:
    return sorted({ts.date() for ts in bars.index})
