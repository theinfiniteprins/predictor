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


def load_intraday(interval: str | None = None, *, instrument: str = "NIFTY50") -> pd.DataFrame:
    """5-min (or configured) OHLCV bars for the primary or correlated index."""
    interval = interval or CONFIG.data.bar_interval
    return _load("yfinance", f"{instrument}_{interval}")


def load_fine(instrument: str = "NIFTY50") -> pd.DataFrame:
    """1-min bars (short history) for intrabar barrier tie-breaking."""
    return _load("yfinance", f"{instrument}_{CONFIG.data.fine_interval}")


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


def load_live_ticks(start: dt.date | None = None, end: dt.date | None = None) -> pd.DataFrame:
    """Cloud/laptop collector ticks (data/raw/live + collected/), concatenated."""
    import glob

    roots = [CONFIG.paths.raw_live, CONFIG.paths.root / "collected"]
    frames = []
    for root in roots:
        for p in glob.glob(str(root / "date=*" / "quotes.parquet")):
            frames.append(read_parquet(p))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(IST)
    df = df.drop_duplicates(subset=["timestamp", "ticker"]).sort_values("timestamp")
    if start:
        df = df[df["timestamp"].dt.date >= start]
    if end:
        df = df[df["timestamp"].dt.date <= end]
    return df.reset_index(drop=True)


def trading_days_in_bars(bars: pd.DataFrame) -> list[dt.date]:
    return sorted({ts.date() for ts in bars.index})
