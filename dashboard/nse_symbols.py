"""Cache + search NSE's official equity list, for the dashboard's "register a stock
by name" search box. Stdlib only.

The list is the same source ``nsepython.nse_eq_symbols()`` uses, fetched directly here
because we also want the company name column (nsepython only returns symbols). Cached
under the shared reference dir (not namespaced per instrument) with a TTL so a search
doesn't re-fetch the ~180KB file every keystroke.
"""

from __future__ import annotations

import csv
import time
import urllib.request
from pathlib import Path

EQUITY_LIST_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
_TTL_SECONDS = 24 * 3600
_TIMEOUT = 10


def _cache_path(reference_dir: Path) -> Path:
    return reference_dir / "nse_equity_list.csv"


def refresh_if_stale(reference_dir: Path) -> Path | None:
    """Return a usable local copy of the equity list, refreshing it if stale.

    Falls back to a stale-but-present cache (or None) if the network fetch fails -
    the dashboard's search should degrade to "type the exact symbol", not crash.
    """
    path = _cache_path(reference_dir)
    if path.exists() and (time.time() - path.stat().st_mtime) < _TTL_SECONDS:
        return path
    try:
        req = urllib.request.Request(EQUITY_LIST_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = resp.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return path
    except Exception:
        return path if path.exists() else None


def search(reference_dir: Path, query: str, limit: int = 15) -> list[dict]:
    """Case-insensitive substring match on symbol or company name."""
    path = refresh_if_stale(reference_dir)
    q = query.strip().lower()
    if not path or not q:
        return []

    results = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        # NSE's header has stray leading spaces on some columns; normalize.
        fieldmap = {(f or "").strip(): f for f in (reader.fieldnames or [])}
        symbol_col = fieldmap.get("SYMBOL")
        name_col = fieldmap.get("NAME OF COMPANY")
        if not symbol_col:
            return []
        for row in reader:
            symbol = (row.get(symbol_col) or "").strip()
            name = (row.get(name_col) or "").strip() if name_col else ""
            if q in symbol.lower() or q in name.lower():
                results.append({"symbol": symbol, "name": name})
                if len(results) >= limit:
                    break
    return results
