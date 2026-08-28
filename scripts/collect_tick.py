"""One quote poll, then exit — the cloud-collector entrypoint (GitHub Actions).

Polls ^NSEI + ^NSEBANK via yfinance and appends to the store pointed at by
$PREDICTOR_LIVE_ROOT (the workflow sets this to ``collected/``). Outside NSE market
hours / on holidays it no-ops and exits 0, so a loose cron is fine.

    python scripts/collect_tick.py            # gated on market hours
    python scripts/collect_tick.py --force    # poll regardless (testing)
"""

from __future__ import annotations

import argparse

from predictor.calendar import market_is_open, now_ist
from predictor.logging_setup import get_logger

log = get_logger("collect_tick")


def main() -> None:
    ap = argparse.ArgumentParser(description="single cloud quote poll")
    ap.add_argument("--force", action="store_true", help="poll even if market is closed")
    args = ap.parse_args()

    if not args.force and not market_is_open():
        log.info("market closed at %s - nothing to do", now_ist().strftime("%Y-%m-%d %H:%M %Z"))
        return

    from predictor.data.live_collector import collect_tick

    n = collect_tick()
    log.info("appended %d bar-rows", n)


if __name__ == "__main__":
    main()
