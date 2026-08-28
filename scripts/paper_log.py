"""Phase 7 - run/inspect the paper-trading log.

    python scripts/paper_log.py               # log newly-resolved entries + print summary
    python scripts/paper_log.py --summary     # summary only, no new logging
    python scripts/paper_log.py --since 2026-09-01

Run it whenever the laptop is on (after `git pull` for fresh collector data). It's
incremental and append-only. Predictions are stamped with the model that made them;
only out-of-sample calls count toward the headline precision.
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
    args = ap.parse_args()

    CONFIG.paths.ensure()
    from predictor import papertrade

    if not args.summary:
        since = dt.date.fromisoformat(args.since) if args.since else None
        papertrade.run(since=since)

    log.info("summary:\n%s", json.dumps(papertrade.summarize(), indent=2, default=str))


if __name__ == "__main__":
    main()
