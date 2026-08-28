import datetime as dt

import pandas as pd

from predictor import calendar as cal


def test_weekend_is_not_a_trading_day():
    # 2026-01-03 is a Saturday
    assert not cal.is_trading_day(dt.date(2026, 1, 3))
    assert not cal.is_trading_day(dt.date(2026, 1, 4))  # Sunday


def test_republic_day_2026_is_holiday():
    assert not cal.is_trading_day(dt.date(2026, 1, 26))


def test_entry_points_count_and_bounds():
    # a plain weekday that is a trading day
    d = dt.date(2026, 1, 6)
    pts = cal.entry_points(d)
    assert pts[0].strftime("%H:%M") == "09:30"
    assert pts[-1].strftime("%H:%M") == "14:30"
    # 09:30..14:30 inclusive, every 15 min -> 21 points
    assert len(pts) == 21
    assert all(p.tzinfo is not None for p in pts)


def test_last_n_sessions_are_ordered_and_exclusive():
    asof = dt.date(2026, 1, 12)  # Monday
    sess = cal.last_n_sessions(asof, 5)
    assert len(sess) == 5
    assert sess == sorted(sess)
    assert asof not in sess
    assert all(cal.is_trading_day(s) for s in sess)


def test_session_bounds_reject_non_trading_day():
    try:
        cal.session_bounds(dt.date(2026, 1, 3))
    except ValueError:
        return
    raise AssertionError("expected ValueError for a weekend")
