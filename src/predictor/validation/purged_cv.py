"""Purged, embargoed walk-forward cross-validation (Lopez de Prado, adapted).

Triple-barrier labels span [t_entry, t_touch]. A naive split lets a training
sample whose outcome resolves *inside* the test period leak the future. So:

  * test folds roll forward in time (expanding train window),
  * a training sample is kept only if its label is fully resolved at least
    ``embargo`` before the test fold starts  (t_touch <= test_start - embargo).

That single rule delivers both the purge (drop overlap) and the embargo (a gap
after the last usable label) for the walk-forward case, where training data is
always strictly earlier than test data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import CONFIG
from ..logging_setup import get_logger

log = get_logger("purged_cv")


@dataclass
class PurgedWalkForwardCV:
    test_window_days: int = CONFIG.cv.test_window_days
    step_days: int = CONFIG.cv.step_days
    min_train_days: int = CONFIG.cv.min_train_days
    embargo_minutes: int = CONFIG.cv.embargo_minutes
    max_train_days: int | None = None          # None -> expanding window
    min_train_samples: int = 50                 # skip folds with a thinner train set

    def _folds(self, start: pd.Timestamp, end: pd.Timestamp):
        first_test = start + pd.Timedelta(days=self.min_train_days)
        cur = first_test
        step = pd.Timedelta(days=self.step_days)
        width = pd.Timedelta(days=self.test_window_days)
        while cur < end:
            yield cur, min(cur + width, end)
            cur += step

    def split(self, t_entry, t_end):
        t_entry = pd.DatetimeIndex(t_entry)
        t_end = pd.DatetimeIndex(t_end)
        n = len(t_entry)
        pos = np.arange(n)
        embargo = pd.Timedelta(minutes=self.embargo_minutes)

        start, end = t_entry.min(), t_entry.max()
        for test_start, test_end in self._folds(start, end):
            test_mask = (t_entry >= test_start) & (t_entry < test_end)
            if not test_mask.any():
                continue
            train_mask = t_end <= (test_start - embargo)
            if self.max_train_days is not None:
                train_mask &= t_entry >= (test_start - pd.Timedelta(days=self.max_train_days))
            if train_mask.sum() < self.min_train_samples:
                continue
            yield pos[train_mask], pos[test_mask]

    def get_n_splits(self, t_entry, t_end) -> int:
        return sum(1 for _ in self.split(t_entry, t_end))
