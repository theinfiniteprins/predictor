"""Phases 4–5 — walk-forward primary + meta, purged/embargoed CV, save the bundle.

    python scripts/train.py                 # default LightGBM params
    python scripts/train.py --tune 200      # Optuna search first (uses your compute)
"""

from __future__ import annotations

import argparse

from predictor.cli import resolve_instrument
resolve_instrument()

from predictor.config import CONFIG
from predictor.logging_setup import get_logger

log = get_logger("train")


def main() -> None:
    ap = argparse.ArgumentParser(description="walk-forward train + validate")
    ap.add_argument("--tune", type=int, default=0, metavar="N",
                    help="run N Optuna trials on the primary before fitting")
    ap.add_argument("--holdout-days", type=int, default=0, metavar="D",
                    help="train only through (last day - D); the rest becomes "
                         "out-of-sample for paper_log (useful for a quick end-to-end test)")
    ap.add_argument("--instrument", default=None, help="instrument key from instruments.yaml")
    args = ap.parse_args()

    CONFIG.paths.ensure()
    from predictor.pipeline import train_and_validate

    train_and_validate(n_trials=args.tune, holdout_days=args.holdout_days)
    log.info("done - see reports/cv/summary.json and models/model_card.json")


if __name__ == "__main__":
    main()
