"""Refresh NSE trading holidays from the exchange and cache them locally.

    python scripts/refresh_calendar.py

Writes data/reference/nse_holidays.json. Run this before Phase 3 labeling and
whenever NSE publishes a new year's holiday list.
"""

from __future__ import annotations

import datetime as dt

from predictor.calendar import save_holidays
from predictor.config import CONFIG
from predictor.logging_setup import get_logger

log = get_logger("refresh_calendar")


def fetch_from_nse() -> set[dt.date]:
    from nsepython import nsefetch

    payload = nsefetch("https://www.nseindia.com/api/holiday-master?type=trading")
    out: set[dt.date] = set()
    for _segment, rows in payload.items():
        for row in rows:
            raw = row.get("tradingDate") or row.get("date")
            if not raw:
                continue
            for fmt in ("%d-%b-%Y", "%d-%b-%Y ", "%Y-%m-%d"):
                try:
                    out.add(dt.datetime.strptime(raw.strip(), fmt).date())
                    break
                except ValueError:
                    continue
    return out


def main() -> None:
    CONFIG.paths.ensure()
    try:
        holidays = fetch_from_nse()
    except Exception as exc:  # noqa: BLE001
        log.error("could not fetch NSE holidays: %s", exc)
        raise SystemExit(1)

    if not holidays:
        log.error("NSE returned no holidays — not overwriting cache")
        raise SystemExit(1)

    save_holidays(holidays)
    log.info("saved %d holidays (%s .. %s)", len(holidays),
             min(holidays), max(holidays))


if __name__ == "__main__":
    main()
