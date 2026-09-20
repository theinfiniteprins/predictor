"""Refresh NSE trading holidays from the exchange and cache them locally.

    python scripts/refresh_calendar.py             # fetch, reconcile, save
    python scripts/refresh_calendar.py --prune     # no fetch: just drop provably wrong entries
    python scripts/refresh_calendar.py --dry-run   # show the diff, write nothing

Writes data/_shared/reference/nse_holidays.json. Run this before Phase 3 labeling and
whenever NSE publishes a new year's holiday list.

Only the **equity** segment counts. NSE's holiday-master API returns one list per
market segment (CM = capital markets/equity, CD = currency, CBM = corporate bonds,
...), and the non-equity segments close on bank holidays when equity trades normally.
Merging every segment - which this script used to do - marked real trading days as
holidays, which makes market_is_open() false all day and the cloud collector skip the
session entirely. That is silent, permanent data loss on a project whose whole
bottleneck is data depth, so the date set is also reconciled against observed market
data before it is saved.
"""

from __future__ import annotations

import argparse
import datetime as dt

from predictor.cli import resolve_instrument
resolve_instrument()

from predictor.calendar import load_holidays, save_holidays
from predictor.config import CONFIG
from predictor.logging_setup import get_logger

log = get_logger("refresh_calendar")

# NSE segment keys that mean "equity / capital market". The API has used a few
# spellings over the years; anything else (currency, debt, mutual funds) is ignored.
_EQUITY_SEGMENTS = {"cm", "capital market", "equities", "equity"}


def _parse_date(raw: str) -> dt.date | None:
    for fmt in ("%d-%b-%Y", "%Y-%m-%d"):
        try:
            return dt.datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


def fetch_from_nse() -> set[dt.date]:
    """Equity-segment trading holidays only."""
    from nsepython import nsefetch

    payload = nsefetch("https://www.nseindia.com/api/holiday-master?type=trading")
    segments = {k: v for k, v in payload.items() if k.strip().lower() in _EQUITY_SEGMENTS}
    if not segments:
        raise RuntimeError(
            f"no equity segment in NSE holiday payload (got {sorted(payload)}) - "
            "refusing to guess, because merging every segment is exactly the bug "
            "that made the collector skip real trading days")

    out: set[dt.date] = set()
    for name, rows in segments.items():
        for row in rows:
            raw = row.get("tradingDate") or row.get("date")
            if raw and (d := _parse_date(raw)):
                out.add(d)
        log.info("segment %s -> %d holidays", name, len(out))
    return out


def traded_dates() -> set[dt.date]:
    """Dates we have actual daily bars for. A date with a bar cannot be a holiday."""
    import pandas as pd
    from predictor.data.load import load_daily

    try:
        daily = load_daily()
    except FileNotFoundError:
        log.warning("no daily bars yet - skipping reconciliation against market data")
        return set()
    return set(pd.DatetimeIndex(daily.index).tz_localize(None).normalize().date)


def reconcile(holidays: set[dt.date]) -> tuple[set[dt.date], set[dt.date]]:
    """Drop any 'holiday' the market demonstrably traded through."""
    traded = traded_dates()
    wrong = {h for h in holidays if h in traded}
    return holidays - wrong, wrong


def main() -> None:
    ap = argparse.ArgumentParser(description="refresh NSE equity trading holidays")
    ap.add_argument("--prune", action="store_true",
                    help="skip the NSE fetch; only reconcile the cached list against market data")
    ap.add_argument("--dry-run", action="store_true", help="print the diff, write nothing")
    ap.add_argument("--instrument", default=None, help="instrument key from instruments.yaml")
    args = ap.parse_args()

    CONFIG.paths.ensure()

    if args.prune:
        holidays = set(load_holidays())
    else:
        try:
            holidays = fetch_from_nse()
        except Exception as exc:  # noqa: BLE001
            log.error("could not fetch NSE holidays: %s", exc)
            log.error("tip: `--prune` still fixes provably-wrong cached entries offline")
            raise SystemExit(1)
        if not holidays:
            log.error("NSE returned no holidays - not overwriting cache")
            raise SystemExit(1)

    kept, wrong = reconcile(holidays)
    if wrong:
        log.warning("dropping %d date(s) marked as holidays that actually traded: %s",
                    len(wrong), ", ".join(str(d) for d in sorted(wrong)))

    before = set(load_holidays())
    added, removed = kept - before, before - kept
    log.info("holidays: %d cached -> %d new (+%d / -%d)",
             len(before), len(kept), len(added), len(removed))
    for d in sorted(added):
        log.info("  + %s", d)
    for d in sorted(removed):
        log.info("  - %s", d)

    if args.dry_run:
        log.info("dry run - nothing written")
        return
    if not kept:
        log.error("refusing to write an empty holiday list")
        raise SystemExit(1)

    save_holidays(kept)
    log.info("saved %d holidays (%s .. %s)", len(kept), min(kept), max(kept))


if __name__ == "__main__":
    main()
