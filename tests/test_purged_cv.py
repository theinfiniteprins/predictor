"""The purged walk-forward splitter must never leak label windows into a test fold."""

import numpy as np
import pandas as pd

from predictor.calendar import IST
from predictor.validation.purged_cv import PurgedWalkForwardCV


def _samples(n_days=40, per_day=20, max_hold_min=300):
    rng = np.random.default_rng(0)
    rows = []
    for d in pd.bdate_range("2026-01-05", periods=n_days, tz=IST):
        for m in range(per_day):
            entry = d + pd.Timedelta(hours=9, minutes=30 + 15 * m)
            hold = pd.Timedelta(minutes=int(rng.integers(5, max_hold_min)))
            rows.append({"t_entry": entry, "t_end": entry + hold})
    return pd.DataFrame(rows)


def test_no_label_window_leaks_into_test_period():
    s = _samples()
    cv = PurgedWalkForwardCV(test_window_days=5, step_days=5, min_train_days=15,
                             embargo_minutes=60, min_train_samples=20)
    embargo = pd.Timedelta(minutes=60)
    n_folds = 0
    for tr, te in cv.split(s["t_entry"], s["t_end"]):
        n_folds += 1
        test_start = s["t_entry"].iloc[te].min()
        # every training label is fully resolved at least `embargo` before the test starts
        assert s["t_end"].iloc[tr].max() <= test_start - embargo
        # and every test entry is strictly after every training entry
        assert s["t_entry"].iloc[tr].max() < s["t_entry"].iloc[te].min()
    assert n_folds >= 3


def test_folds_roll_forward_in_time():
    s = _samples()
    cv = PurgedWalkForwardCV(test_window_days=5, step_days=5, min_train_days=15,
                             min_train_samples=20)
    starts = [s["t_entry"].iloc[te].min() for _, te in cv.split(s["t_entry"], s["t_end"])]
    assert starts == sorted(starts)
