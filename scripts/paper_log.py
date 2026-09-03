"""Phase 7 - run/inspect the paper-trading log.

    python scripts/paper_log.py               # log newly-resolved entries + print summary
    python scripts/paper_log.py --summary     # summary only, no new logging
    python scripts/paper_log.py --rebuild     # wipe + re-log (after changing k / holdout)

Run this BEFORE retraining, so the calls are logged against the model that was
live when the entry resolved (that's what makes them out-of-sample). It's
incremental and append-only. Only out-of-sample calls count toward the headline.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json

from predictor.config import CONFIG
from predictor.logging_setup import get_logger

log = get_logger("paper_log")


def main() -> None:
    ap = argparse.ArgumentParser(description="paper-trading log")
    ap.add_argument("--summary", action="store_true", help="print summary only")
    ap.add_argument("--since", default=None, help="only log entries on/after YYYY-MM-DD")
    ap.add_argument("--rebuild", action="store_true",
                    help="wipe the log and re-log against the current model")
    ap.add_argument("--no-build", action="store_true",
                    help="reuse the existing dataset.parquet (run.ps1 uses this)")
    args = ap.parse_args()

    CONFIG.paths.ensure()
    from predictor import papertrade

    if not args.summary:
        try:
            since = dt.date.fromisoformat(args.since) if args.since else None
            papertrade.run(since=since, rebuild=args.rebuild, build=not args.no_build)
        except FileNotFoundError as exc:
            log.warning("nothing to log yet: %s", exc)
            return

    log.info("summary:\n%s", json.dumps(papertrade.summarize(), indent=2, default=str))


if __name__ == "__main__":
    main()
