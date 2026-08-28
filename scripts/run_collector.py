r"""Track B entrypoint - run the live collector.

Foreground:
    python scripts/run_collector.py

Windows Task Scheduler: daily trigger at ~09:10 IST, action =
    <venv>\Scripts\python.exe  D:\Predictor\scripts\run_collector.py
The process exits itself shortly after 15:30 IST, so no stop action is needed.

    --no-options   skip the 15-min option-chain snapshots (quotes only)
    --once         take a single quote poll and exit (for testing)
"""

from __future__ import annotations

import argparse

from predictor.config import CONFIG
from predictor.logging_setup import get_logger

log = get_logger("run_collector")


def main() -> None:
    ap = argparse.ArgumentParser(description="Track B live collector")
    ap.add_argument("--no-options", action="store_true", help="quotes only, no option snapshots")
    ap.add_argument("--once", action="store_true", help="single poll then exit")
    args = ap.parse_args()

    CONFIG.paths.ensure()

    from predictor.data.live_collector import collect_tick, run

    if args.once:
        n = collect_tick()
        log.info("single poll appended %d bar-rows", n)
        return

    try:
        run(run_option_snapshots=not args.no_options)
    except KeyboardInterrupt:
        log.info("collector stopped by user")


if __name__ == "__main__":
    main()
