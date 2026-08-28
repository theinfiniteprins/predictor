"""Track B — the live collector. START THIS RUNNING NOW.

Every ``collector.poll_seconds`` during market hours it appends one quote row for
``^NSEI`` to ``data/raw/live/date=YYYY-MM-DD/quotes.parquet``. Every 15 minutes it
also takes an option-chain + India-VIX snapshot. In 2–3 months this is a real
proprietary intraday dataset — the whole plan hinges on it accumulating.

Run it foreground in a terminal, or point Windows Task Scheduler at
``scripts/run_collector.py`` with a daily 09:10 IST trigger (it exits itself after
close).
"""

from __future__ import annotations

import time as _time

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from ..calendar import IST, is_trading_day, market_is_open, now_ist, session_bounds
from ..config import CONFIG
from ..logging_setup import get_logger
from ..storage import append_rows
from .option_chain import snapshot as option_snapshot

log = get_logger("live_collector")

_SNAPSHOT_EVERY = pd.Timedelta(minutes=15)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def _latest_minute_bar(ticker: str) -> dict:
    df = yf.Ticker(ticker).history(period="1d", interval="1m", auto_adjust=False)
    if df is None or df.empty:
        raise RuntimeError(f"no 1m data for {ticker}")
    df = df.rename(columns=str.lower)
    last = df.iloc[-1]
    bar_ts = pd.DatetimeIndex([df.index[-1]])
    bar_ts = (bar_ts.tz_localize("UTC") if bar_ts.tz is None else bar_ts).tz_convert(IST)[0]
    return {
        "bar_time": bar_ts,
        "open": float(last["open"]),
        "high": float(last["high"]),
        "low": float(last["low"]),
        "close": float(last["close"]),
        "volume": float(last.get("volume", 0) or 0),
    }


def _poll_one(ticker: str, poll_ts: pd.Timestamp) -> dict | None:
    try:
        bar = _latest_minute_bar(ticker)
    except Exception as exc:  # noqa: BLE001
        log.warning("quote poll failed for %s: %s", ticker, exc)
        return None
    return {"timestamp": poll_ts, "ticker": ticker, "source": "yfinance_1m", **bar}


def collect_once(ticker: str | None = None) -> dict | None:
    ticker = ticker or CONFIG.instrument.yf_ticker
    poll_ts = now_ist()
    row = _poll_one(ticker, poll_ts)
    if row is None:
        return None
    append_rows(pd.DataFrame([row]), day=poll_ts.date())
    log.info("tick %s  %s close=%.2f  bar=%s", poll_ts.strftime("%H:%M:%S"),
             ticker, row["close"], row["bar_time"].strftime("%H:%M"))
    return row


def collect_tick(tickers: list[str] | None = None) -> list[dict]:
    """Poll several tickers in one shot and append all rows. Cloud-collector entrypoint."""
    tickers = tickers or [CONFIG.instrument.yf_ticker, CONFIG.instrument.correlated_ticker]
    poll_ts = now_ist()
    rows = [r for r in (_poll_one(t, poll_ts) for t in tickers) if r is not None]
    if not rows:
        log.warning("collect_tick: no rows this poll")
        return []
    append_rows(pd.DataFrame(rows), day=poll_ts.date())
    log.info("tick %s  %s", poll_ts.strftime("%H:%M:%S"),
             "  ".join(f"{r['ticker']}={r['close']:.2f}" for r in rows))
    return rows


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
    log.info("live collector starting - poll=%ss ticker=%s",
             CONFIG.collector.poll_seconds, CONFIG.instrument.yf_ticker)
    last_snapshot: pd.Timestamp | None = None

    while True:
        if not market_is_open():
            ts = now_ist()
            # if the session is over for today, stop; a scheduler restarts us tomorrow
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
