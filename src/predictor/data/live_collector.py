"""Track B - the live collector. START THIS RUNNING (it already is, on GitHub Actions).

Each poll fetches *all* of today's 1-minute bars for ^NSEI + ^NSEBANK and appends
the completed ones to ``<PREDICTOR_LIVE_ROOT>/date=YYYY-MM-DD/quotes.parquet``,
de-duplicated on (bar_time, ticker). Because every poll re-fetches the whole day,
a missed poll is automatically backfilled by the next one - so a 5-minute cron
still yields a complete 1-minute series.

``scripts/collect_tick.py`` is the one-shot entrypoint (GitHub Actions). ``run()``
is the local looping version for when the laptop is on.
"""

from __future__ import annotations

import time as _time

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from ..calendar import IST, is_trading_day, market_is_open, now_ist, session_bounds
from ..config import CONFIG
from ..logging_setup import get_logger
from ..storage import append_bars
from .option_chain import snapshot as option_snapshot

log = get_logger("live_collector")

_SNAPSHOT_EVERY = pd.Timedelta(minutes=15)
_OHLCV = ["open", "high", "low", "close", "volume"]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _recent_minute_bars(ticker: str, lookback: int = 40) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period="1d", interval="1m", auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"no 1m data for {ticker}")
    df = df.rename(columns=str.lower)
    idx = pd.DatetimeIndex(df.index)
    df.index = (idx.tz_localize("UTC") if idx.tz is None else idx).tz_convert(IST)
    df = df.iloc[:-1]                       # drop the still-forming current minute
    df = df.tail(lookback)[[c for c in _OHLCV if c in df.columns]]
    df.index.name = "bar_time"
    return df


def collect_tick(tickers: list[str] | None = None, lookback: int = 40) -> int:
    """One poll: append recent completed 1-min bars for each ticker. Returns row count."""
    tickers = tickers or [CONFIG.instrument.yf_ticker, CONFIG.instrument.correlated_ticker]
    now = now_ist()
    frames: list[pd.DataFrame] = []
    for t in tickers:
        try:
            b = _recent_minute_bars(t, lookback).reset_index()
        except Exception as exc:  # noqa: BLE001
            log.warning("poll failed for %s: %s", t, exc)
            continue
        b["ticker"] = t
        b["source"] = "yfinance_1m"
        b["ingested_at"] = now
        frames.append(b)

    if not frames:
        log.warning("collect_tick: nothing fetched this poll")
        return 0

    rows = pd.concat(frames, ignore_index=True)
    append_bars(rows, day=now.date())
    latest = rows.sort_values("bar_time").groupby("ticker").last()["close"]
    log.info("poll %s  %d bar-rows  %s", now.strftime("%H:%M:%S"), len(rows),
             "  ".join(f"{k}={v:.2f}" for k, v in latest.items()))
    return len(rows)


def _sleep_until_open() -> None:
    ts = now_ist()
    d = ts.date()
    if is_trading_day(d):
        open_dt, _ = session_bounds(d)
        if ts < open_dt:
            secs = (open_dt - ts).total_seconds()
            log.info("pre-open; sleeping %.0f min until %s", secs / 60, open_dt.strftime("%H:%M"))
            _time.sleep(min(secs, 3600))
            return
    log.info("market closed; sleeping 30 min")
    _time.sleep(1800)


def run(run_option_snapshots: bool = True) -> None:
    log.info("live collector starting - poll=%ss", CONFIG.collector.poll_seconds)
    last_snapshot: pd.Timestamp | None = None

    while True:
        if not market_is_open():
            ts = now_ist()
            if is_trading_day(ts.date()):
                _, close_dt = session_bounds(ts.date())
                if ts > close_dt:
                    log.info("session closed for %s - collector exiting", ts.date())
                    return
            _sleep_until_open()
            continue

        collect_tick()

        if run_option_snapshots:
            now = now_ist()
            if last_snapshot is None or (now - last_snapshot) >= _SNAPSHOT_EVERY:
                try:
                    option_snapshot()
                    last_snapshot = now
                except Exception as exc:  # noqa: BLE001
                    log.warning("option snapshot failed: %s", exc)

        _time.sleep(CONFIG.collector.poll_seconds)
