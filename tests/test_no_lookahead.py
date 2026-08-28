"""Feature rows must not change when *future* bars are added or altered."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from predictor.calendar import IST
from predictor.features.build import build_features


@pytest.fixture
def synthetic_bars():
    rng = np.random.default_rng(1)
    frames = []
    for d in pd.bdate_range("2026-02-02", periods=8, tz=IST):
        idx = pd.date_range(d + pd.Timedelta(hours=9, minutes=15), periods=75, freq="5min", tz=IST)
        price = 100 + np.cumsum(rng.normal(0, 0.2, len(idx)))
        frames.append(pd.DataFrame(
            {"open": price, "high": price + 0.1, "low": price - 0.1, "close": price,
             "volume": 0.0}, index=idx))
    bars = pd.concat(frames)
    bars.index.name = "timestamp"
    return bars


def test_features_ignore_future_bars(synthetic_bars):
    bars = synthetic_bars
    days = sorted({ts.date() for ts in bars.index})
    cut_day = days[4]
    cut = pd.Timestamp(dt.datetime.combine(cut_day, dt.time(11, 0)), tz=IST)

    full = build_features(bars=bars, banknifty=None, days=days)

    truncated_bars = bars[bars.index < cut]
    trunc = build_features(bars=truncated_bars, banknifty=None,
                           days=sorted({ts.date() for ts in truncated_bars.index}))

    common = full.index.intersection(trunc.index)
    common = common[common < cut]
    assert len(common) > 10

    a = full.loc[common].drop(columns=[c for c in ("india_vix", "india_vix_chg") if c in full], errors="ignore")
    b = trunc.loc[common].reindex(columns=a.columns)
    pd.testing.assert_frame_equal(a, b, check_dtype=False, rtol=1e-9, atol=1e-9)
