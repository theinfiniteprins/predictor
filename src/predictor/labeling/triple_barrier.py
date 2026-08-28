"""Triple-barrier labeling (Lopez de Prado) with rolling intraday entry points.

For each entry point:
  upper = entry_price * (1 + k * sigma_intraday)   -> label +1
  lower = entry_price * (1 - k * sigma_intraday)   -> label -1
  vertical = 15:20 IST                             -> label  0

Whichever barrier price touches FIRST in time order sets the label. When a single
5-min bar straddles both barriers the order is unknown: we zoom into that bar's
1-min data to break the tie, and if that is also ambiguous (or 1-min data is
missing) the instance is DROPPED (label = NaN) rather than guessed.

Every feature/label here uses only information at or after the entry timestamp for
resolution, and `sigma_intraday` uses only prior sessions - so labels carry no
look-ahead beyond their own (t_entry -> t_touch) window, which the purged CV then
accounts for.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..calendar import entry_points, vertical_barrier
from ..config import CONFIG
from ..logging_setup import get_logger
from ..data.load import load_fine, load_intraday
from .volatility import effective_sigma, sigma_for_day, sigma_series

log = get_logger("triple_barrier")

_BAR = pd.Timedelta(CONFIG.data.bar_interval)


@dataclass
class BarrierResult:
    label: float          # -1.0, 0.0, +1.0, or np.nan (dropped)
    t_touch: pd.Timestamp
    touch_price: float
    bars_held: int
    reason: str           # "upper" | "lower" | "vertical" | "intrabar_ambiguous"


def _first_touch_1m(fine_win: pd.DataFrame, upper: float, lower: float) -> str | None:
    """Walk 1-min bars within one 5-min window; return 'upper'/'lower'/None/'ambiguous'."""
    for _ts, b in fine_win.iterrows():
        hi, lo = b["high"] >= upper, b["low"] <= lower
        if hi and lo:
            return "ambiguous"
        if hi:
            return "upper"
        if lo:
            return "lower"
    return None


def label_entry(
    day_bars: pd.DataFrame,
    entry_ts: pd.Timestamp,
    entry_price: float,
    upper: float,
    lower: float,
    vertical_ts: pd.Timestamp,
    fine_day: pd.DataFrame | None = None,
) -> BarrierResult:
    path = day_bars.loc[(day_bars.index >= entry_ts) & (day_bars.index < vertical_ts)]

    for i, (bar_ts, bar) in enumerate(path.iterrows(), start=1):
        hit_upper = bar["high"] >= upper
        hit_lower = bar["low"] <= lower
        if not (hit_upper or hit_lower):
            continue

        if hit_upper and hit_lower:
            resolved: str | None = "ambiguous"
            if CONFIG.labeling.intrabar_tiebreak == "one_minute" and fine_day is not None:
                win = fine_day.loc[
                    (fine_day.index >= bar_ts) & (fine_day.index < bar_ts + _BAR)
                ]
                if not win.empty:
                    resolved = _first_touch_1m(win, upper, lower)
            if resolved == "upper":
                return BarrierResult(1.0, bar_ts, upper, i, "upper")
            if resolved == "lower":
                return BarrierResult(-1.0, bar_ts, lower, i, "lower")
            return BarrierResult(np.nan, bar_ts, float("nan"), i, "intrabar_ambiguous")

        if hit_upper:
            return BarrierResult(1.0, bar_ts, upper, i, "upper")
        return BarrierResult(-1.0, bar_ts, lower, i, "lower")

    # no barrier touched -> timeout
    exit_price = float(path["close"].iloc[-1]) if len(path) else entry_price
    return BarrierResult(0.0, vertical_ts, exit_price, len(path), "vertical")


def build_labels(
    bars: pd.DataFrame | None = None,
    fine: pd.DataFrame | None = None,
    k: float | None = None,
    days: list[dt.date] | None = None,
) -> pd.DataFrame:
    """Label every rolling entry point across the available intraday history."""
    bars = load_intraday() if bars is None else bars
    k = CONFIG.labeling.k if k is None else k
    if fine is None:
        try:
            fine = load_fine()
        except FileNotFoundError:
            fine = None
            log.warning("no 1-min data - intrabar ties will be dropped")

    sigma = sigma_series()
    all_days = days or sorted({ts.date() for ts in bars.index})

    rows: list[dict] = []
    n_dropped = n_skipped = 0
    for day in all_days:
        sig = sigma_for_day(day, sigma)
        if not np.isfinite(sig):
            n_skipped += 1
            continue
        day_bars = bars[bars.index.date == day]
        if day_bars.empty:
            continue
        fine_day = fine[fine.index.date == day] if fine is not None else None
        v_ts = vertical_barrier(day)

        for entry_ts in entry_points(day):
            if entry_ts not in day_bars.index:
                n_skipped += 1
                continue
            entry_price = float(day_bars.loc[entry_ts, "open"])
            sig_eff = effective_sigma(sig, entry_ts, v_ts)
            upper = entry_price * (1 + k * sig_eff)
            lower = entry_price * (1 - k * sig_eff)
            res = label_entry(day_bars, entry_ts, entry_price, upper, lower, v_ts, fine_day)
            if not np.isfinite(res.label):
                n_dropped += 1
            rows.append({
                "day": day,
                "t_entry": entry_ts,
                "t_touch": res.t_touch,
                "entry_price": entry_price,
                "sigma_intraday": sig,
                "sigma_effective": sig_eff,
                "k": k,
                "upper": upper,
                "lower": lower,
                "label": res.label,
                "touch_price": res.touch_price,
                "ret_at_touch": res.touch_price / entry_price - 1.0,
                "bars_held": res.bars_held,
                "reason": res.reason,
            })

    df = pd.DataFrame(rows).set_index("t_entry").sort_index()
    kept = df["label"].notna().sum()
    log.info(
        "labeled %d entries over %d days | kept %d, dropped %d (intrabar), skipped %d "
        "(no bar / insufficient daily history)",
        len(df), df["day"].nunique() if len(df) else 0, kept, n_dropped, n_skipped,
    )
    if kept:
        dist = df.loc[df["label"].notna(), "label"].value_counts().sort_index()
        log.info("label distribution: %s",
                 {int(k_): int(v_) for k_, v_ in dist.items()})
    return df
