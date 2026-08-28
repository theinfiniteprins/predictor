"""Parquet read/write helpers for the local data lake.

Conventions
-----------
* Everything is a tz-aware ``DatetimeIndex`` in IST (or a ``timestamp`` column for
  the append-only live store).
* Bar datasets live at ``data/raw/<source>/<name>.parquet`` (single file, fully
  rewritten on refresh — these come from re-runnable pulls).
* The live collector uses ``append_rows`` into date-partitioned files under
  ``data/raw/live/date=YYYY-MM-DD/quotes.parquet`` so a crash never corrupts more
  than the current day and concurrent reads stay cheap.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pandas as pd

from .calendar import IST
from .config import CONFIG
from .logging_setup import get_logger

log = get_logger("storage")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Atomically write ``df`` to ``path`` (write temp, then replace)."""
    _ensure_parent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, engine="pyarrow")
    tmp.replace(path)
    try:
        shown = path.relative_to(CONFIG.paths.root)
    except ValueError:
        shown = path
    log.info("wrote %d rows -> %s", len(df), shown)
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine="pyarrow")
    return df


def raw_path(source: str, name: str) -> Path:
    return CONFIG.paths.raw / source / f"{name}.parquet"


# --------------------------------------------------------------------------- #
# append-only live store (Track B)
# --------------------------------------------------------------------------- #

def _live_root() -> Path:
    """Where the append-only quote store lives.

    ``PREDICTOR_LIVE_ROOT`` overrides it (relative paths resolve under the project
    root) — the GitHub Actions collector points this at ``collected/`` so cloud
    ticks land in a git-tracked directory instead of the gitignored data lake.
    Also the indirection point tests use to redirect into a tmp dir.
    """
    override = os.environ.get("PREDICTOR_LIVE_ROOT")
    if override:
        p = Path(override)
        return p if p.is_absolute() else CONFIG.paths.root / p
    return CONFIG.paths.raw_live


def _live_path(day: dt.date) -> Path:
    return _live_root() / f"date={day.isoformat()}" / "quotes.parquet"


def append_rows(rows: pd.DataFrame, day: dt.date | None = None) -> Path:
    """Append ``rows`` to the live store for ``day`` (defaults to today IST).

    ``rows`` must have a ``timestamp`` column (tz-aware). Rows are de-duplicated on
    (timestamp, ticker) — or just timestamp if there's no ``ticker`` column —
    keeping the last, so a re-poll after a transient failure is harmless and one
    poll can carry several tickers.
    """
    if "timestamp" not in rows.columns:
        raise ValueError("rows must contain a 'timestamp' column")
    day = day or pd.Timestamp.now(tz=IST).date()
    path = _live_path(day)

    if path.exists():
        existing = read_parquet(path)
        combined = pd.concat([existing, rows], ignore_index=True)
    else:
        combined = rows.copy()

    key = ["timestamp", "ticker"] if "ticker" in combined.columns else ["timestamp"]
    combined = (
        combined.drop_duplicates(subset=key, keep="last")
        .sort_values(key)
        .reset_index(drop=True)
    )
    return write_parquet(combined, path)


def append_bars(rows: pd.DataFrame, day: dt.date | None = None) -> Path:
    """Append 1-min bar rows to the live store, de-duped on (bar_time, ticker).

    ``rows`` needs ``bar_time`` and ``ticker`` columns. Re-appending an overlapping
    fetch is harmless (keeps the last value for each bar).
    """
    for col in ("bar_time", "ticker"):
        if col not in rows.columns:
            raise ValueError(f"rows must contain a '{col}' column")
    day = day or pd.Timestamp.now(tz=IST).date()
    path = _live_path(day)

    combined = pd.concat([read_parquet(path), rows], ignore_index=True) if path.exists() else rows.copy()
    combined = (
        combined.drop_duplicates(subset=["bar_time", "ticker"], keep="last")
        .sort_values(["bar_time", "ticker"])
        .reset_index(drop=True)
    )
    return write_parquet(combined, path)


def read_live(day: dt.date) -> pd.DataFrame:
    path = _live_path(day)
    if not path.exists():
        return pd.DataFrame(columns=["timestamp"])
    return read_parquet(path)


def read_live_range(start: dt.date, end: dt.date) -> pd.DataFrame:
    frames = []
    d = start
    while d <= end:
        p = _live_path(d)
        if p.exists():
            frames.append(read_parquet(p))
        d += dt.timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=["timestamp"])
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
