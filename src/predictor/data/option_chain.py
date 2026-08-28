"""Track A — NSE index option chain + India VIX -> daily context features.

Unofficial NSE JSON endpoints (via nsepython). No SLA, needs browser-like headers,
can break without notice. Every call degrades to an empty/partial result + a log
line rather than raising, and snapshots are appended so a gap is just a gap.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from ..calendar import IST, now_ist
from ..config import CONFIG
from ..logging_setup import get_logger
from ..storage import read_parquet, write_parquet

log = get_logger("option_chain")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _option_chain_raw(symbol: str) -> dict:
    from nsepython import nse_optionchain_scrapper

    data = nse_optionchain_scrapper(symbol)
    if not data or "records" not in data:
        raise RuntimeError("option chain payload missing 'records'")
    return data


def fetch_option_chain_snapshot(symbol: str | None = None) -> dict:
    symbol = symbol or CONFIG.instrument.option_symbol
    try:
        data = _option_chain_raw(symbol)
    except Exception as exc:  # noqa: BLE001
        log.error("option chain fetch failed for %s: %s", symbol, exc)
        return {}

    records = data["records"]
    filt = data.get("filtered", {})
    underlying = records.get("underlyingValue")

    ce_oi = filt.get("CE", {}).get("totOI")
    pe_oi = filt.get("PE", {}).get("totOI")
    ce_vol = filt.get("CE", {}).get("totVol")
    pe_vol = filt.get("PE", {}).get("totVol")

    pcr_oi = (pe_oi / ce_oi) if ce_oi else None
    pcr_vol = (pe_vol / ce_vol) if ce_vol else None

    # ATM implied vol: nearest strike to underlying on the front expiry
    atm_iv = None
    try:
        rows = records["data"]
        front_expiry = records["expiryDates"][0]
        nearest = min(
            (r for r in rows if r.get("expiryDate") == front_expiry),
            key=lambda r: abs(r["strikePrice"] - underlying),
        )
        ivs = [nearest.get("CE", {}).get("impliedVolatility"),
               nearest.get("PE", {}).get("impliedVolatility")]
        ivs = [v for v in ivs if v]
        atm_iv = sum(ivs) / len(ivs) if ivs else None
    except Exception:  # noqa: BLE001
        pass

    return {
        "timestamp": now_ist().isoformat(),
        "symbol": symbol,
        "underlying": underlying,
        "total_ce_oi": ce_oi,
        "total_pe_oi": pe_oi,
        "pcr_oi": pcr_oi,
        "pcr_vol": pcr_vol,
        "atm_iv": atm_iv,
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def fetch_india_vix() -> dict:
    try:
        from nsepython import nsefetch
    except ImportError:
        log.error("nsepython not installed; skipping India VIX")
        return {}
    try:
        payload = nsefetch("https://www.nseindia.com/api/allIndices")
        row = next(r for r in payload["data"] if r["index"].upper() == "INDIA VIX")
    except Exception as exc:  # noqa: BLE001
        log.error("India VIX fetch failed: %s", exc)
        return {}
    return {
        "timestamp": now_ist().isoformat(),
        "india_vix": row.get("last"),
        "india_vix_pct_change": row.get("percentChange"),
    }


def _append_snapshot(row: dict, name: str) -> None:
    if not row:
        return
    year = now_ist().year
    path = CONFIG.paths.raw / "option_chain" / f"{name}_{year}.parquet"
    new = pd.DataFrame([row])
    if path.exists():
        combined = pd.concat([read_parquet(path), new], ignore_index=True)
    else:
        combined = new
    combined = combined.drop_duplicates(subset="timestamp", keep="last")
    write_parquet(combined, path)


def snapshot() -> dict:
    """Take one option-chain + VIX snapshot and append it to the context store."""
    oc = fetch_option_chain_snapshot()
    vix = fetch_india_vix()
    _append_snapshot(oc, "option_chain")
    _append_snapshot(vix, "india_vix")
    merged = {**oc, **{k: v for k, v in vix.items() if k != "timestamp"}}
    log.info("snapshot: underlying=%s pcr_oi=%s vix=%s",
             merged.get("underlying"), merged.get("pcr_oi"), merged.get("india_vix"))
    return merged
