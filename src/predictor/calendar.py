"""NSE trading calendar: trading-day checks, session bounds, entry-point grid.

Holidays are loaded from ``data/reference/nse_holidays.json`` (refresh it with
``python scripts/refresh_calendar.py``). If that file is missing we fall back to a
hardcoded list and log a warning — good enough for the Track B collector, but
Phase 3 labeling should always run against a freshly refreshed list.
"""

from __future__ import annotations

import datetime as dt
import json
from zoneinfo import ZoneInfo

import pandas as pd

from .config import CONFIG
from .logging_setup import get_logger

log = get_logger("calendar")

IST = ZoneInfo(CONFIG.session.timezone)

# Best-effort fallback (verify against https://www.nseindia.com/resources/exchange-communication-holidays).
# Only used if the refreshed JSON is absent.
_FALLBACK_HOLIDAYS: set[str] = {
    # 2025
    "2025-02-26", "2025-03-14", "2025-03-31", "2025-04-10", "2025-04-14",
    "2025-04-18", "2025-05-01", "2025-08-15", "2025-08-27", "2025-10-02",
    "2025-10-21", "2025-10-22", "2025-11-05", "2025-12-25",
    # 2026 (provisional — MUST be refreshed from NSE once published)
    "2026-01-26", "2026-03-06", "2026-03-25", "2026-04-01", "2026-04-03",
    "2026-04-14", "2026-05-01", "2026-08-15", "2026-10-02", "2026-11-09",
    "2026-12-25",
}

_HOLIDAY_FILE = CONFIG.paths.data_dir / "reference" / "nse_holidays.json"


def load_holidays() -> set[dt.date]:
    if _HOLIDAY_FILE.exists():
        raw = json.loads(_HOLIDAY_FILE.read_text(encoding="utf-8"))
        return {dt.date.fromisoformat(s) for s in raw["holidays"]}
    log.warning(
        "no refreshed holiday file at %s - using hardcoded fallback. "
        "Run `python scripts/refresh_calendar.py` before Phase 3 labeling.",
        _HOLIDAY_FILE,
    )
    return {dt.date.fromisoformat(s) for s in _FALLBACK_HOLIDAYS}


def save_holidays(dates: set[dt.date]) -> None:
    _HOLIDAY_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "refreshed_at": dt.datetime.now(tz=IST).isoformat(),
        "holidays": sorted(d.isoformat() for d in dates),
    }
    _HOLIDAY_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


_HOLIDAYS = load_holidays()


def is_trading_day(d: dt.date) -> bool:
    return d.weekday() < 5 and d not in _HOLIDAYS


def _combine(d: dt.date, t: dt.time) -> pd.Timestamp:
    return pd.Timestamp(dt.datetime.combine(d, t), tz=IST)


def session_bounds(d: dt.date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(open, close) as IST-aware Timestamps. Raises if ``d`` is not a trading day."""
    if not is_trading_day(d):
        raise ValueError(f"{d} is not an NSE trading day")
    return _combine(d, CONFIG.session.open), _combine(d, CONFIG.session.close)


def vertical_barrier(d: dt.date) -> pd.Timestamp:
    return _combine(d, CONFIG.labeling.vertical_barrier)


def entry_points(d: dt.date) -> list[pd.Timestamp]:
    """Rolling intraday entry timestamps for day ``d`` per config (09:30..14:30 / 15m)."""
    start = _combine(d, CONFIG.labeling.entry_start)
    end = _combine(d, CONFIG.labeling.entry_end)
    freq = f"{CONFIG.labeling.entry_freq_minutes}min"
    return list(pd.date_range(start, end, freq=freq))


def trading_days(start: dt.date, end: dt.date) -> list[dt.date]:
    days = pd.date_range(start, end, freq="D")
    return [d.date() for d in days if is_trading_day(d.date())]


def last_n_sessions(asof: dt.date, n: int, inclusive: bool = False) -> list[dt.date]:
    """The ``n`` most recent trading days strictly before ``asof`` (or including it)."""
    out: list[dt.date] = []
    cur = asof if inclusive else asof - dt.timedelta(days=1)
    while len(out) < n:
        if is_trading_day(cur):
            out.append(cur)
        cur -= dt.timedelta(days=1)
    return list(reversed(out))


def now_ist() -> pd.Timestamp:
    return pd.Timestamp.now(tz=IST)


def market_is_open(ts: pd.Timestamp | None = None) -> bool:
    ts = ts or now_ist()
    ts = ts.tz_convert(IST)
    if not is_trading_day(ts.date()):
        return False
    open_dt, close_dt = session_bounds(ts.date())
    return open_dt <= ts <= close_dt
