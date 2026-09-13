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

    python scripts/collect_all.py            # gated on market hours
    python scripts/collect_all.py --force    # poll regardless (testing)
"""

from __future__ import annotations

import argparse

from predictor.calendar import market_is_open, now_ist
from predictor.config import load_instruments_registry
from predictor.logging_setup import get_logger

log = get_logger("collect_all")


def main() -> None:
    ap = argparse.ArgumentParser(description="single cloud quote poll, every registered instrument")
    ap.add_argument("--force", action="store_true", help="poll even if the market is closed")
    args = ap.parse_args()

    if not args.force and not market_is_open():
        log.info("market closed at %s - nothing to do", now_ist().strftime("%Y-%m-%d %H:%M %Z"))
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
