"""Cloud Track-B entrypoint (GitHub Actions) - one poll for EVERY registered
instrument, one commit per run.

Reads instruments.yaml and fetches the deduplicated union of every instrument's own
ticker + benchmark ticker (so a shared benchmark like NIFTY50 is only fetched once
regardless of how many stocks reference it), then appends everything to the same
``<PREDICTOR_LIVE_ROOT>/date=YYYY-MM-DD/quotes.parquet`` the single-instrument
collector always has (rows are disambiguated by their `ticker` column, so no new
per-instrument folder convention is needed - see storage.append_bars).

Registering a new stock only ever edits instruments.yaml, never this script or the
workflow file - that's what keeps concurrent collector runs from ever needing to
merge, since there's still exactly one process making exactly one commit.

    python scripts/collect_all.py            # gated on market hours / trading days
    python scripts/collect_all.py --force    # poll regardless (testing)
"""

from __future__ import annotations

import argparse

import pandas as pd

from predictor.calendar import is_trading_day, now_ist, session_bounds
from predictor.config import load_instruments_registry
from predictor.logging_setup import get_logger

log = get_logger("collect_all")


def should_collect(now: pd.Timestamp) -> tuple[bool, str]:
    """Check if collection should run at the given IST timestamp.

    Allows polling during market hours and post-market on trading days (up to 23:59 IST)
    so runs delayed by cloud schedulers (e.g. GitHub Actions) still capture the full
    session's 1-minute bars.
    """
    d = now.date()
    if not is_trading_day(d):
        return False, f"today {d} is not an NSE trading day"
    open_dt, _ = session_bounds(d)
    if now < open_dt:
        return False, f"pre-market at {now.strftime('%H:%M %Z')} (opens at {open_dt.strftime('%H:%M')})"
    return True, "trading day session or post-market capture"


def main() -> None:
    ap = argparse.ArgumentParser(description="single cloud quote poll, every registered instrument")
    ap.add_argument("--force", action="store_true", help="poll even outside eligible collection hours")
    args = ap.parse_args()

    now = now_ist()
    if not args.force:
        can_poll, reason = should_collect(now)
        if not can_poll:
            log.info("collection skipped at %s: %s", now.strftime("%Y-%m-%d %H:%M %Z"), reason)
            return

    from predictor.data.live_collector import collect_tick

    registry = load_instruments_registry()
    if not registry:
        log.warning("instruments.yaml has no instruments - nothing to poll")
        return

    tickers: list[str] = []
    for entry in registry:
        for t in (entry["yf_ticker"], entry.get("benchmark_ticker")):
            if t and t not in tickers:
                tickers.append(t)

    log.info("polling %d instrument(s), %d unique ticker(s): %s",
             len(registry), len(tickers), ", ".join(tickers))
    n = collect_tick(tickers=tickers)
    log.info("appended %d bar-rows", n)


if __name__ == "__main__":
    main()
