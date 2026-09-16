"""Average label uniqueness - the correction for overlapping triple-barrier labels."""

import numpy as np
import pandas as pd

from predictor.calendar import IST
from predictor.labeling.uniqueness import average_uniqueness, concurrency, uniqueness_weights


def _spans(pairs):
    entry = pd.DatetimeIndex([pd.Timestamp(a, tz=IST) for a, _ in pairs])
    end = pd.DatetimeIndex([pd.Timestamp(b, tz=IST) for _, b in pairs])
    return entry, end


def test_non_overlapping_labels_are_fully_unique():
    entry, end = _spans([
        ("2026-01-06 09:30", "2026-01-06 10:00"),
        ("2026-01-06 10:30", "2026-01-06 11:00"),
        ("2026-01-06 11:30", "2026-01-06 12:00"),
    ])
    assert np.allclose(average_uniqueness(entry, end), 1.0)


def test_fully_overlapping_labels_split_their_weight():
    # three labels covering exactly the same span each carry a third of the evidence
    entry, end = _spans([("2026-01-06 09:30", "2026-01-06 15:20")] * 3)
    assert np.allclose(average_uniqueness(entry, end), 1 / 3)


def test_partial_overlap_sits_between():
    entry, end = _spans([
        ("2026-01-06 09:30", "2026-01-06 15:20"),
        ("2026-01-06 14:00", "2026-01-06 15:20"),
    ])
    u = average_uniqueness(entry, end)
    assert 0.5 < u[0] < 1.0        # mostly alone, shares only its tail
    assert np.isclose(u[1], 0.5)   # every moment of its life is shared


def test_concurrency_counts_live_labels():
    entry, end = _spans([
        ("2026-01-06 09:30", "2026-01-06 15:20"),
        ("2026-01-06 09:45", "2026-01-06 15:20"),
    ])
    c = concurrency(entry, end)
    assert c.loc[pd.Timestamp("2026-01-06 09:30", tz=IST)] == 1
    assert c.loc[pd.Timestamp("2026-01-06 09:45", tz=IST)] == 2


def test_weights_are_normalised_to_mean_one():
    entry, end = _spans([
        ("2026-01-06 09:30", "2026-01-06 15:20"),
        ("2026-01-06 14:00", "2026-01-06 15:20"),
        ("2026-01-07 09:30", "2026-01-07 10:00"),
    ])
    w = uniqueness_weights(entry, end)
    assert np.isclose(w.mean(), 1.0)
    assert (w > 0).all()
