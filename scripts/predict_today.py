"""Phase 7 — live inference. Prints the call for the current entry point, or 'no signal'.

    python scripts/predict_today.py               # most recent entry point, now
    python scripts/predict_today.py --at 11:15    # a specific entry point today
    python scripts/predict_today.py --date 2026-08-27 --at 13:30

Needs today's intraday bars present: run `python scripts/run_backfill.py --instrument`
first (or wire in the collector store).
"""

from __future__ import annotations

import argparse
import datetime as dt

import numpy as np
import pandas as pd

from predictor.calendar import IST, now_ist
from predictor.config import CONFIG
from predictor.logging_setup import get_logger

log = get_logger("predict_today")


def main() -> None:
    ap = argparse.ArgumentParser(description="live prediction for one entry point")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default: today IST)")
    ap.add_argument("--at", default=None, help="HH:MM entry point (default: most recent)")
    args = ap.parse_args()

    from predictor.features.build import build_features
    from predictor.labeling.entry_points import current_entry_point
    from predictor.models.meta import confidence_features
    from predictor.models.persist import load_bundle
    from predictor.models.primary import PROBA_COLS, _LABELS

    day = dt.date.fromisoformat(args.date) if args.date else now_ist().date()
    if args.at:
        hh, mm = map(int, args.at.split(":"))
        entry_ts = pd.Timestamp(dt.datetime.combine(day, dt.time(hh, mm)), tz=IST)
    else:
        entry_ts = current_entry_point()
    if entry_ts is None:
        log.info("no entry point reached yet today"); return

    bundle = load_bundle()
    feats = build_features(days=[entry_ts.date()])
    if entry_ts not in feats.index:
        log.info("no feature row for %s (missing bar / before first entry)", entry_ts); return

    row = feats.loc[[entry_ts]].reindex(columns=bundle["feature_columns"])
    proba = bundle["primary"].predict_proba(row.to_numpy(dtype="float64", na_value=np.nan))[0]
    pred = int(_LABELS[int(np.argmax(proba))])
    pdict = {c: round(float(p), 3) for c, p in zip(PROBA_COLS, proba)}

    if pred == 0:
        log.info("%s  ->  NO SIGNAL (timeout most likely)  %s", entry_ts.strftime("%Y-%m-%d %H:%M"), pdict)
        return

    side = "UP" if pred == 1 else "DOWN"
    if bundle["meta"] is None or bundle.get("fire_threshold") is None:
        log.info("%s  ->  primary says %s %s, but meta-model not available yet (need more history)",
                 entry_ts.strftime("%Y-%m-%d %H:%M"), side, pdict)
        return

    conf = confidence_features(pd.DataFrame([proba], columns=PROBA_COLS))
    meta_in = pd.concat([row.reset_index(drop=True), conf], axis=1)
    score = float(bundle["meta"].predict_proba(meta_in.to_numpy(dtype="float64", na_value=np.nan))[0, 1])
    thr = bundle["fire_threshold"]

    verdict = "FIRE" if score >= thr else "no signal (below confidence bar)"
    log.info("%s  ->  primary %s %s | meta score %.3f (fire >= %.3f)  =>  %s",
             entry_ts.strftime("%Y-%m-%d %H:%M"), side, pdict, score, thr, verdict)


if __name__ == "__main__":
    main()
