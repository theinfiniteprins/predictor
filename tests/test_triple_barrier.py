"""Triple-barrier first-touch logic on synthetic paths with known outcomes."""

import numpy as np
import pandas as pd
import pytest

from predictor.calendar import IST
from predictor.labeling.triple_barrier import label_entry


def _bars(closes, day="2026-01-06", start="09:30", freq="5min", spread=0.0):
    idx = pd.date_range(f"{day} {start}", periods=len(closes), freq=freq, tz=IST)
    c = np.asarray(closes, dtype="float64")
    return pd.DataFrame(
        {"open": c, "high": c + spread, "low": c - spread, "close": c}, index=idx
    )


VERT = pd.Timestamp("2026-01-06 15:20", tz=IST)


def test_upper_barrier_first():
    bars = _bars([100, 101, 103, 104])          # rising
    r = label_entry(bars, bars.index[0], 100.0, upper=102.0, lower=98.0, vertical_ts=VERT)
    assert r.label == 1.0 and r.reason == "upper"
    assert r.t_touch == bars.index[2]


def test_lower_barrier_first():
    bars = _bars([100, 99, 97, 96])
    r = label_entry(bars, bars.index[0], 100.0, upper=102.0, lower=98.0, vertical_ts=VERT)
    assert r.label == -1.0 and r.reason == "lower"


def test_timeout_when_no_barrier_touched():
    bars = _bars([100, 100.5, 99.7, 100.2])
    r = label_entry(bars, bars.index[0], 100.0, upper=102.0, lower=98.0, vertical_ts=VERT)
    assert r.label == 0.0 and r.reason == "vertical"
    assert r.touch_price == pytest.approx(100.2)


def test_intrabar_ambiguous_dropped_without_fine_data():
    bars = _bars([100, 100], spread=0.0)
    bars.iloc[1, bars.columns.get_loc("high")] = 103.0
    bars.iloc[1, bars.columns.get_loc("low")] = 97.0
    r = label_entry(bars, bars.index[0], 100.0, upper=102.0, lower=98.0, vertical_ts=VERT)
    assert np.isnan(r.label) and r.reason == "intrabar_ambiguous"


def test_intrabar_tie_broken_by_one_minute_data():
    bars = _bars([100, 100])
    bars.iloc[1, bars.columns.get_loc("high")] = 103.0
    bars.iloc[1, bars.columns.get_loc("low")] = 97.0
    # within that 5-min window, price hits the LOWER barrier first
    fine = _bars([99, 96.5, 103.5], day="2026-01-06", start="09:35", freq="1min")
    r = label_entry(bars, bars.index[0], 100.0, 102.0, 98.0, VERT, fine_day=fine)
    assert r.label == -1.0 and r.reason == "lower"


def test_entry_bar_itself_can_touch():
    bars = _bars([100, 100, 100])
    bars.iloc[0, bars.columns.get_loc("high")] = 102.5
    r = label_entry(bars, bars.index[0], 100.0, upper=102.0, lower=98.0, vertical_ts=VERT)
    assert r.label == 1.0 and r.t_touch == bars.index[0]
