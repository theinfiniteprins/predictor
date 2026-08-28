"""Merge the growing collector data with the (rolling ~60-day) yfinance pull into a
single history that keeps extending.

Outputs, per instrument, into ``data/interim/``:
  <inst>_1m_unified.parquet   yfinance 1-min  +  collector 1-min
  <inst>_5m_unified.parquet   yfinance 5-min  +  collector 1-min resampled to 5-min

yfinance wins on any overlapping bar (clean official aggregation); the collector
fills everything outside yfinance's window and any gaps. ``load_intraday`` /
``load_fine`` pick these up automatically once they exist.
"""

from __future__ import annotations

import pandas as pd

from ..config import CONFIG
from ..logging_setup import get_logger
from ..storage import raw_path, read_parquet, write_parquet
from .load import load_collected_bars

log = get_logger("consolidate")

_OHLCV = ["open", "high", "low", "close", "volume"]
_INSTRUMENTS = {"NIFTY50": "^NSEI", "BANKNIFTY": "^NSEBANK"}


def _ns(df: pd.DataFrame) -> pd.DataFrame:
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index = pd.DatetimeIndex(df.index).as_unit("ns")
    return df


def _resample_5m(one_min: pd.DataFrame) -> pd.DataFrame:
    if one_min.empty:
        return one_min
    agg = one_min.resample("5min", label="left", closed="left", origin="start_day").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return agg.dropna(subset=["open"])


def _read_ohlcv(p) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame(columns=_OHLCV)
    df = read_parquet(p)
    return _ns(df[[c for c in _OHLCV if c in df.columns]])


def _raw(instrument: str, interval: str) -> pd.DataFrame:
    return _read_ohlcv(raw_path("yfinance", f"{instrument}_{interval}"))


def _prev_unified(instrument: str, interval: str) -> pd.DataFrame:
    return _read_ohlcv(CONFIG.paths.interim / f"{instrument}_{interval}_unified.parquet")


def consolidate(instrument: str = "NIFTY50", ticker: str | None = None) -> dict:
    ticker = ticker or _INSTRUMENTS.get(instrument, CONFIG.instrument.yf_ticker)
    col = load_collected_bars(ticker=ticker)
    col1 = pd.DataFrame(columns=_OHLCV)
    if not col.empty:
        col1 = _ns(col.set_index("bar_time")[[c for c in _OHLCV if c in col.columns]])

    out = {}
    fine = CONFIG.data.fine_interval    # "1m"
    bar = CONFIG.data.bar_interval      # "5m"

    # precedence: fresh yfinance  >  previously-accumulated unified (keeps days that
    # have since aged out of yfinance's rolling window)  >  collector-derived bars
    def _merge(interval: str, collector: pd.DataFrame) -> pd.DataFrame:
        m = _raw(instrument, interval).combine_first(_prev_unified(instrument, interval))
        if not collector.empty:
            m = m.combine_first(collector)
        return _ns(m)

    uni1 = _merge(fine, col1)
    uni5 = _merge(bar, _resample_5m(col1))

    CONFIG.paths.interim.mkdir(parents=True, exist_ok=True)
    for interval, df in ((fine, uni1), (bar, uni5)):
        if df.empty:
            continue
        df.index.name = "timestamp"
        write_parquet(df, CONFIG.paths.interim / f"{instrument}_{interval}_unified.parquet")
        out[interval] = {"rows": len(df),
                         "span": [str(df.index.min().date()), str(df.index.max().date())]}
    log.info("%s consolidated: %s (collector 1m rows: %d)", instrument, out, len(col1))
    return out


def consolidate_all() -> dict:
    res = {}
    for inst, tkr in _INSTRUMENTS.items():
        try:
            res[inst] = consolidate(inst, tkr)
        except Exception as exc:  # noqa: BLE001
            log.warning("consolidate(%s) failed: %s", inst, exc)
    return res
