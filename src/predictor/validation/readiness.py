"""Is this thing ready to trade with real money?

The evidence gate decides whether the model may make *paper* calls. This is a much
higher bar, and it is deliberately harder to clear, because the cost of being wrong
stops being a number on a dashboard.

The difference that matters: the gate is judged on out-of-fold predictions, which are
still a backtest - the labels existed before the prediction did. Readiness additionally
demands a *forward* record: calls the deployed model actually made, on days it had
never seen, that resolved afterwards. That is what paper_trades.parquet accumulates.

Every check must pass. A single failure means not ready, and the report says which one
and what would fix it - "not yet, and here is why" is the useful answer, not a score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_setup import get_logger
from .significance import ROUND_TRIP_COST, day_block_bootstrap

log = get_logger("readiness")

# Deliberately conservative. These are not tuned to be reachable - they are what it
# would take to believe a result with money behind it.
MIN_LIVE_CALLS = 50          # forward, out-of-sample, actually-fired calls
MIN_LIVE_DAYS = 60           # spread over this many distinct sessions
MIN_RECENT_DAYS = 20         # the edge must still be there lately, not only in 2024


def _check(name: str, passed: bool, detail: str, blocking: str = "") -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail,
            "how_to_fix": blocking if not passed else ""}


def readiness_report(
    card: dict | None,
    paper: pd.DataFrame | None,
    cost: float = ROUND_TRIP_COST,
    n_boot: int = 2000,
) -> dict:
    """Checklist for going live. ``paper`` is the paper_trades log, ``card`` the model card."""
    checks: list[dict] = []
    card = card or {}
    edge = card.get("edge") or {}
    gate = card.get("fire_gate") or {}

    # 1. the model is even allowed to make paper calls
    firing = card.get("fire_threshold") is not None
    checks.append(_check(
        "Model has proven an edge on historical data",
        firing,
        "The evidence gate is open." if firing else
        "The evidence gate is shut - the model has not beaten a simple baseline.",
        "Keep collecting data. Nothing else here can pass until this does."))

    # 2. that edge survives costs
    clears = bool(edge.get("clears_breakeven"))
    be = edge.get("breakeven_precision")
    checks.append(_check(
        "Edge is big enough to cover trading costs",
        clears,
        (f"Precision clears the {100*be:.1f}% break-even bar." if clears and be
         else f"Needs {100*be:.1f}% precision to break even; not there yet."
         if be else "Break-even not computable yet."),
        "An edge that loses money after costs is not worth trading."))

    # 3. it held up across repeated retrains
    passes = int(gate.get("consecutive_passes", 0) or 0)
    need = int(gate.get("passes_required", 0) or 0)
    sustained = bool(need and passes >= need)
    checks.append(_check(
        "Edge held up across repeated retrains",
        sustained,
        f"Passed {passes} of {need} required consecutive retrains.",
        "A one-off pass is usually luck; it has to repeat as new data arrives."))

    # 4-7: the forward record. This is the part a backtest cannot fake.
    live = pd.DataFrame()
    if paper is not None and not paper.empty:
        live = paper[(~paper.get("in_sample", pd.Series(False, index=paper.index)).fillna(False))
                     & paper.get("fired", pd.Series(False, index=paper.index)).fillna(False)]
        live = live.dropna(subset=["primary_pred", "label"])

    n_live = len(live)
    n_days = int(live["day"].nunique()) if n_live else 0
    enough = n_live >= MIN_LIVE_CALLS and n_days >= MIN_LIVE_DAYS
    checks.append(_check(
        "Enough real forward calls to judge",
        enough,
        f"{n_live} live calls across {n_days} sessions "
        f"(need {MIN_LIVE_CALLS} across {MIN_LIVE_DAYS}).",
        "These must be calls the deployed model made before the outcome existed - "
        "backtested calls do not count, however good they look."))

    if n_live:
        live = live.copy()
        live["net"] = live["primary_pred"] * live["ret_at_touch"] - cost
        mean_net = float(live["net"].mean())
        lo, _, hi = day_block_bootstrap(live, lambda z: float(z["net"].mean()) * 1e4,
                                        n_boot=n_boot)
        profitable = bool(np.isfinite(lo) and lo > 0)
        checks.append(_check(
            "Forward record is actually profitable",
            profitable,
            f"Mean {1e4*mean_net:+.2f} bps/call, 95% CI "
            f"[{lo:+.1f}, {hi:+.1f}] bps." if np.isfinite(lo)
            else f"Mean {1e4*mean_net:+.2f} bps/call (interval not computable).",
            "The whole interval must sit above zero, not just the average."))

        recent_cut = np.sort(pd.unique(live["day"]))[-MIN_RECENT_DAYS:]
        recent = live[live["day"].isin(recent_cut)]
        recent_ok = bool(len(recent) and float(recent["net"].mean()) > 0)
        checks.append(_check(
            "Still working recently, not just historically",
            recent_ok,
            f"Last {len(recent_cut)} sessions: {1e4*float(recent['net'].mean()):+.2f} bps/call."
            if len(recent) else "No recent live calls.",
            "An edge that worked last year and not this quarter has decayed."))
    else:
        checks.append(_check("Forward record is actually profitable", False,
                             "No live calls yet.", "Nothing to measure until calls exist."))
        checks.append(_check("Still working recently, not just historically", False,
                             "No live calls yet.", "Nothing to measure until calls exist."))

    passed = sum(c["passed"] for c in checks)
    ready = all(c["passed"] for c in checks)
    blockers = [c["name"] for c in checks if not c["passed"]]

    return {
        "ready": ready,
        "checks": checks,
        "passed": passed,
        "total": len(checks),
        "blockers": blockers,
        "headline": ("Ready to consider real money" if ready else
                     f"Not ready - {len(blockers)} of {len(checks)} checks still failing"),
        "detail": ("Every check passed. This is the tool saying the evidence supports "
                   "live trading; it is not financial advice, and position sizing and "
                   "risk limits are still your call."
                   if ready else
                   "The first blocking item is: " + blockers[0] + "."),
        "thresholds": {"min_live_calls": MIN_LIVE_CALLS, "min_live_days": MIN_LIVE_DAYS,
                       "min_recent_days": MIN_RECENT_DAYS},
    }
