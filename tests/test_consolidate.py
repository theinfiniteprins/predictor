"""Collector 1-min bars -> 5-min bars, and the append_bars de-dup."""

import datetime as dt

import numpy as np
import pandas as pd

from predictor import storage
from predictor.calendar import IST
from predictor.data.consolidate import _resample_5m


def test_resample_5m_aggregates_ohlcv_on_the_yfinance_grid():
    idx = pd.date_range("2026-01-06 09:15", periods=10, freq="1min", tz=IST)
    one_min = pd.DataFrame({
        "open":  [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        "high":  [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
        "low":   [99.5, 100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5],
        "close": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "volume": [1] * 10,
    }, index=idx)

    out = _resample_5m(one_min)
    assert list(out.index.strftime("%H:%M")) == ["09:15", "09:20"]
    first = out.iloc[0]
    assert first["open"] == 100 and first["close"] == 105
    assert first["high"] == 104.5 and first["low"] == 99.5
    assert first["volume"] == 5


def test_append_bars_dedups_on_bar_time_and_ticker(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "_live_root", lambda: tmp_path)
    day = dt.date(2026, 1, 6)
    t = pd.date_range("2026-01-06 09:15", periods=3, freq="1min", tz=IST)

    storage.append_bars(pd.DataFrame({
        "bar_time": t, "ticker": "^NSEI", "open": [1, 2, 3], "high": [1, 2, 3],
        "low": [1, 2, 3], "close": [1, 2, 3], "volume": [0, 0, 0]}), day=day)
    # overlapping re-fetch, last bar revised + one new bar
    storage.append_bars(pd.DataFrame({
        "bar_time": [t[2], t[2] + pd.Timedelta(minutes=1)], "ticker": "^NSEI",
        "open": [99, 4], "high": [99, 4], "low": [99, 4], "close": [99, 4],
        "volume": [0, 0]}), day=day)

    got = storage.read_live(day)
    assert len(got) == 4
    assert got.set_index("bar_time").loc[t[2], "close"] == 99
