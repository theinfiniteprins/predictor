import datetime as dt

import pandas as pd
import pytest

from predictor import storage
from predictor.calendar import IST


@pytest.fixture(autouse=True)
def _tmp_data_lake(tmp_path, monkeypatch):
    """Redirect the live store into a tmp dir for the duration of each test."""
    monkeypatch.setattr(storage, "_live_root", lambda: tmp_path)
    yield


def test_append_rows_dedupes_and_sorts():
    day = dt.date(2026, 1, 6)
    t0 = pd.Timestamp("2026-01-06 09:30:00", tz=IST)
    t1 = pd.Timestamp("2026-01-06 09:31:00", tz=IST)

    storage.append_rows(pd.DataFrame({"timestamp": [t1], "close": [100.0]}), day=day)
    storage.append_rows(pd.DataFrame({"timestamp": [t0], "close": [99.0]}), day=day)
    # re-poll of t1 with a corrected value
    storage.append_rows(pd.DataFrame({"timestamp": [t1], "close": [101.0]}), day=day)

    got = storage.read_live(day)
    assert list(got["timestamp"]) == [t0, t1]
    assert got.loc[got["timestamp"] == t1, "close"].iloc[0] == 101.0


def test_append_rows_requires_timestamp_column():
    with pytest.raises(ValueError):
        storage.append_rows(pd.DataFrame({"close": [1.0]}), day=dt.date(2026, 1, 6))


def test_append_rows_keeps_multiple_tickers_at_one_timestamp():
    day = dt.date(2026, 1, 6)
    ts = pd.Timestamp("2026-01-06 09:30:00", tz=IST)
    storage.append_rows(
        pd.DataFrame({"timestamp": [ts, ts], "ticker": ["^NSEI", "^NSEBANK"],
                      "close": [24000.0, 51000.0]}),
        day=day,
    )
    got = storage.read_live(day)
    assert len(got) == 2
    assert set(got["ticker"]) == {"^NSEI", "^NSEBANK"}

    # a re-poll of the same (timestamp, ticker) overwrites, doesn't duplicate
    storage.append_rows(
        pd.DataFrame({"timestamp": [ts], "ticker": ["^NSEI"], "close": [24010.0]}),
        day=day,
    )
    got = storage.read_live(day)
    assert len(got) == 2
    assert got.loc[got["ticker"] == "^NSEI", "close"].iloc[0] == 24010.0
