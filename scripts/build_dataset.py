"""Phase 2+3 — build the modelling dataset (features joined to triple-barrier labels).

    python scripts/build_dataset.py               # uses config k
    python scripts/build_dataset.py --k 0.6       # override barrier multiplier
"""

from __future__ import annotations

import argparse

from predictor.config import CONFIG
from predictor.logging_setup import get_logger

log = get_logger("build_dataset")


def main() -> None:
    ap = argparse.ArgumentParser(description="build features + labels")
    ap.add_argument("--k", type=float, default=None, help="barrier multiplier (default: config)")
    args = ap.parse_args()

    CONFIG.paths.ensure()
    from predictor import dataset

    df = dataset.build(k=args.k)
    meta = dataset.load_meta()
    log.info("label counts: %s", meta["label_counts"])
    log.info("%d feature columns", len(meta["feature_columns"]))


if __name__ == "__main__":
    main()
