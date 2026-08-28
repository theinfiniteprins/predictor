"""Track A — NSE daily EOD history via jugaad-data.

Used for *long-history context features* (vol regime, long moving averages,
seasonality), not intraday training. yfinance already gives ~10y of index daily
bars; this adds an official-source cross-check and, later, market-breadth data.

jugaad-data scrapes NSE's public site — it can rate-limit or break. Every function
degrades gracefully (logs + returns empty) rather than raising.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from ..calendar import IST
from ..config import CONFIG
from ..logging_setup import get_logger
from ..storage import raw_path, write_parquet

log = get_logger("backfill_bhavcopy")


def fetch_index_history(symbol: str = "NIFTY 50", years: int = 10) -> pd.DataFrame:
    """Daily OHLC for an NSE index, from NSE's own historical endpoint."""
    try:
        from jugaad_data.nse import index_df
    except ImportError:
        log.error("jugaad-data not installed; skipping index history")
        return pd.DataFrame()

    to_date = dt.date.today()
    from_date = to_date.replace(year=to_date.year - years)
    try:
        raw = index_df(symbol=symbol, from_date=from_date, to_date=to_date)
    except Exception as exc:  # noqa: BLE001
        log.error("index_df(%s) failed: %s", symbol, exc)
        return pd.DataFrame()

    if raw is None or raw.empty:
        log.warning("index_df(%s) returned nothing", symbol)
        return pd.DataFrame()

    raw.columns = [c.strip().upper() for c in raw.columns]
    date_col = next((c for c in raw.columns if "DATE" in c), None)
    df = pd.DataFrame({
        "open": pd.to_numeric(raw["OPEN"], errors="coerce"),
        "high": pd.to_numeric(raw["HIGH"], errors="coerce"),
        "low": pd.to_numeric(raw["LOW"], errors="coerce"),
        "close": pd.to_numeric(raw["CLOSE"], errors="coerce"),
    })
    df.index = pd.DatetimeIndex(pd.to_datetime(raw[date_col])).tz_localize(IST)
    df.index.name = "timestamp"
    df = df[~df.index.duplicated(keep="last")].sort_index().dropna(how="all")
    log.info("%s: %d daily bars %s -> %s", symbol, len(df),
             df.index.min().date() if len(df) else "-",
             df.index.max().date() if len(df) else "-")
    return df


def backfill() -> dict[str, int]:
    out: dict[str, int] = {}
    df = fetch_index_history("NIFTY 50", years=int(CONFIG.data.daily_period.rstrip("y") or 10))
    if not df.empty:
        write_parquet(df, raw_path("bhavcopy", "NIFTY50_1d_nse"))
    out["NIFTY50_1d_nse"] = len(df)
    return out
