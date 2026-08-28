"""Merge collector data with the yfinance pull into the growing unified history.

    python scripts/consolidate.py

Run after `git pull` (to get fresh collector data) and `run_backfill.py --instrument`
(to refresh the yfinance window). `build_dataset.py` also does this automatically.
"""

from __future__ import annotations

from predictor.config import CONFIG
from predictor.logging_setup import get_logger

log = get_logger("consolidate_cli")


def main() -> None:
    CONFIG.paths.ensure()
    from predictor.data.consolidate import consolidate_all

    res = consolidate_all()
    for inst, intervals in res.items():
        for interval, info in intervals.items():
            log.info("%s %s: %d rows  %s", inst, interval, info["rows"], info["span"])


if __name__ == "__main__":
    main()
