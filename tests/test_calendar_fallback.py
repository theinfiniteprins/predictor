"""The hardcoded fallback calendar is what the cloud collector actually runs on.

data/ is gitignored, so the refreshed nse_holidays.json never reaches the GitHub
Actions runner and calendar.py falls back to its built-in set. A wrong entry there
silently costs a whole session of 1-minute collection, so it gets its own guard.
"""

import datetime as dt

import predictor.calendar as cal


def _fallback_2026():
    return {dt.date.fromisoformat(s) for s in cal._FALLBACK_HOLIDAYS if s.startswith("2026")}


def test_fallback_excludes_days_the_market_demonstrably_traded():
    """These five were in the list because the old refresh merged every NSE market
    segment; the currency/debt segments close on bank holidays when equity trades."""
    traded = {dt.date(2026, 1, 1), dt.date(2026, 2, 19), dt.date(2026, 3, 19),
              dt.date(2026, 4, 1), dt.date(2026, 8, 26)}
    assert not (_fallback_2026() & traded), (
        "fallback marks a real trading day as a holiday - the collector would skip it")


def test_fallback_covers_the_current_year_with_a_plausible_count():
    """NSE publishes roughly 14-20 equity holidays a year, and does include ones that
    land on weekends (2026-08-15 is a Saturday). Weekend entries are harmless - they
    are already excluded by the weekday check - but a wildly wrong count means the
    list was pasted from the wrong segment or never refreshed."""
    weekdays = {d for d in _fallback_2026() if d.weekday() < 5}
    assert 8 <= len(weekdays) <= 20, (
        f"{len(weekdays)} weekday holidays in 2026 looks wrong: {sorted(weekdays)}")


def test_known_holidays_are_present():
    for d in (dt.date(2026, 1, 26), dt.date(2026, 8, 15), dt.date(2026, 10, 2),
              dt.date(2026, 12, 25)):
        assert d in _fallback_2026()


def test_trading_day_logic_uses_the_calendar():
    assert cal.is_trading_day(dt.date(2026, 12, 26)) is False   # Saturday
    assert cal.is_trading_day(dt.date(2026, 12, 25)) is False   # holiday
