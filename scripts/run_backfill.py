"""Track A — one-shot historical backfill. Re-runnable; safe to run any time.

    python scripts/run_backfill.py --all
    python scripts/run_backfill.py --instrument --cues
"""

from __future__ import annotations

import argparse

from predictor.config import CONFIG
from predictor.logging_setup import get_logger

log = get_logger("run_backfill")


def main() -> None:
    ap = argparse.ArgumentParser(description="Track A historical backfill")
    ap.add_argument("--all", action="store_true", help="run every backfill job")
    ap.add_argument("--instrument", action="store_true", help="^NSEI + Bank Nifty bars")
    ap.add_argument("--cues", action="store_true", help="S&P/Nasdaq/USDINR/crude daily")
    ap.add_argument("--bhavcopy", action="store_true", help="NSE index daily history")
    ap.add_argument("--option-chain", action="store_true", help="one option-chain + VIX snapshot")
    args = ap.parse_args()

    if not any([args.all, args.instrument, args.cues, args.bhavcopy, args.option_chain]):
        ap.error("pick at least one job (or --all)")

    CONFIG.paths.ensure()
    results: dict[str, object] = {}

    if args.all or args.instrument:
        from predictor.data.backfill_yfinance import backfill_instrument
        results["instrument"] = backfill_instrument()
    if args.all or args.cues:
        from predictor.data.backfill_yfinance import backfill_global_cues
        results["cues"] = backfill_global_cues()
    if args.all or args.bhavcopy:
        from predictor.data.backfill_bhavcopy import backfill
        results["bhavcopy"] = backfill()
    if args.all or args.option_chain:
        from predictor.data.option_chain import snapshot
        results["option_chain"] = snapshot()

    log.info("backfill summary: %s", results)


if __name__ == "__main__":
    main()
