"""Is a measured result real, or is it noise?

Every headline number in this project is computed on rolling intraday entries whose
triple-barrier labels overlap heavily: all ~21 entries in a session resolve by 15:20
that same day, and 23 of the features are constant within a day. Measured directly,
~1500 rows carry only ~115 independent observations. So the usual binomial/Wilson
interval - which assumes independent rows - is roughly 2-3x too narrow here and will
call noise significant.

Two corrections live in this module:

1. **Day-block bootstrap.** Resample whole trading days with replacement rather than
   individual rows, so the interval inherits the real correlation structure.
2. **The right baseline.** "Beat a coin flip" is the wrong bar. Barrier touches are
   asymmetric (a falling market hits the lower barrier more often), so "always name
   the majority direction" is a strong naive strategy that captures nothing but drift.
   A model only has an edge if it beats *that*.

Used by the pipeline to decide whether the system is allowed to fire at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_setup import get_logger

log = get_logger("significance")

DEFAULT_N_BOOT = 2000
DEFAULT_ALPHA = 0.05
ROUND_TRIP_COST = 3e-4      # 2 x 1.5bps slippage, matching backtest.engine defaults


def day_block_bootstrap(
    df: pd.DataFrame,
    stat,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 11,
    day_col: str = "day",
) -> tuple[float, float, float]:
    """(lo, median, hi) for ``stat(df)`` resampling whole days with replacement.

    Rows inside one session are not independent, so resampling rows would produce a
    spuriously tight interval. Returns NaNs if there is nothing to resample.
    """
    if df.empty or day_col not in df.columns:
        return (np.nan, np.nan, np.nan)
    days = df[day_col].to_numpy()
    uniq = pd.unique(days)
    if len(uniq) < 2:
        return (np.nan, np.nan, np.nan)
    by_day = {d: np.where(days == d)[0] for d in uniq}

    rng = np.random.default_rng(seed)
    out: list[float] = []
    for _ in range(n_boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([by_day[uniq[i]] for i in pick])
        try:
            v = float(stat(df.iloc[idx]))
        except Exception:  # noqa: BLE001 - a degenerate resample is not an error
            continue
        if np.isfinite(v):
            out.append(v)
    if len(out) < max(50, n_boot // 20):
        return (np.nan, np.nan, np.nan)
    lo, med, hi = np.percentile(out, [100 * alpha / 2, 50, 100 * (1 - alpha / 2)])
    return (float(lo), float(med), float(hi))


def naive_majority_side(labels: np.ndarray) -> int:
    """The direction a no-skill strategy would always name: whichever barrier gets
    touched more often in this sample. Beating this is the bar for having an edge,
    because matching it only reproduces the period's drift."""
    labels = np.asarray(labels)
    n_up = int(np.sum(labels == 1))
    n_dn = int(np.sum(labels == -1))
    return 1 if n_up > n_dn else -1


def _precision(df: pd.DataFrame) -> float:
    return float((df["primary_pred"] == df["label"]).mean())


def edge_report(
    fired: pd.DataFrame,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    min_trades: int = 30,
    min_days: int = 15,
    cost: float = ROUND_TRIP_COST,
) -> dict:
    """Does this set of fired calls beat the naive baseline, accounting for overlap?

    ``fired`` needs columns: primary_pred, label, day (and ret_at_touch for economics).
    The verdict that matters is ``has_edge``: the day-block bootstrap lower bound on
    (model precision - naive precision) is above zero.
    """
    fired = fired.dropna(subset=["primary_pred", "label"])
    fired = fired[fired["primary_pred"] != 0]
    n = len(fired)
    n_days = int(fired["day"].nunique()) if n else 0
    out: dict = {
        "n_fired": n, "n_days": n_days,
        "min_trades_required": min_trades, "min_days_required": min_days,
    }
    if n == 0:
        out.update(precision=np.nan, naive_precision=np.nan, naive_side=None,
                   margin=np.nan, margin_ci=[np.nan, np.nan], has_edge=False,
                   verdict="no directional calls to evaluate")
        return out

    labels = fired["label"].to_numpy()
    side = naive_majority_side(labels)
    precision = _precision(fired)
    naive = float((labels == side).mean())

    def margin(z: pd.DataFrame) -> float:
        return _precision(z) - float((z["label"].to_numpy() == side).mean())

    m_lo, m_med, m_hi = day_block_bootstrap(fired, margin, n_boot, alpha)
    p_lo, _, p_hi = day_block_bootstrap(fired, _precision, n_boot, alpha)

    enough = n >= min_trades and n_days >= min_days
    has_edge = bool(enough and np.isfinite(m_lo) and m_lo > 0)

    out.update(
        precision=precision,
        precision_ci=[p_lo, p_hi],
        naive_precision=naive,
        naive_side="up" if side == 1 else "down",
        margin=precision - naive,
        margin_ci=[m_lo, m_hi],
        has_edge=has_edge,
    )

    if "ret_at_touch" in fired.columns:
        def net_total(z: pd.DataFrame) -> float:
            return float((z["primary_pred"] * z["ret_at_touch"] - cost).sum())
        n_lo, n_med, n_hi = day_block_bootstrap(fired, net_total, n_boot, alpha)
        out["net_return_total"] = float(
            (fired["primary_pred"] * fired["ret_at_touch"] - cost).sum())
        out["net_return_ci"] = [n_lo, n_hi]
        out["profitable"] = bool(np.isfinite(n_lo) and n_lo > 0)

    if not enough:
        out["verdict"] = (
            f"not enough evidence yet - {n} calls over {n_days} days "
            f"(need {min_trades} calls over {min_days} days to even test)")
    elif has_edge:
        out["verdict"] = (
            f"beats the naive 'always {out['naive_side']}' baseline by "
            f"{100*out['margin']:.1f}pp (95% CI {100*m_lo:+.1f} to {100*m_hi:+.1f}pp)")
    else:
        out["verdict"] = (
            f"no measurable edge - {100*precision:.1f}% vs naive {100*naive:.1f}%, "
            f"margin 95% CI {100*m_lo:+.1f} to {100*m_hi:+.1f}pp spans zero")
    return out


def choose_fire_threshold(
    candidates: pd.DataFrame,
    min_trades: int = 30,
    min_days: int = 15,
    n_boot: int = DEFAULT_N_BOOT,
    alpha: float = DEFAULT_ALPHA,
    n_grid: int = 9,
) -> tuple[float | None, dict]:
    """Pick the meta-score cutoff to fire at - or None, meaning fire nothing.

    ``candidates`` = OOF rows where the primary named a direction, with a meta_score.
    We sweep a small grid of cutoffs and keep the best, so alpha is Bonferroni-adjusted
    by the grid size: picking the best of N tries needs a correspondingly higher bar.

    Returning None is a normal, expected outcome - it means the evidence does not yet
    support making live calls, and the system should stay silent.
    """
    cand = candidates.dropna(subset=["primary_pred", "label", "meta_score"])
    cand = cand[cand["primary_pred"] != 0]
    if cand.empty:
        return None, {"reason": "no scored directional candidates", "has_edge": False}

    grid = np.unique(np.quantile(cand["meta_score"], np.linspace(0.0, 0.9, n_grid)))
    alpha_adj = alpha / max(len(grid), 1)

    best_thr, best_rep, best_lo = None, None, -np.inf
    for thr in grid:
        sel = cand[cand["meta_score"] >= thr]
        rep = edge_report(sel, n_boot=n_boot, alpha=alpha_adj,
                          min_trades=min_trades, min_days=min_days)
        lo = rep.get("margin_ci", [np.nan])[0]
        if np.isfinite(lo) and lo > best_lo:
            best_thr, best_rep, best_lo = float(thr), rep, lo

    if best_rep is None:
        return None, {"reason": "no threshold produced an evaluable set", "has_edge": False}

    best_rep = {**best_rep, "threshold": best_thr, "grid_size": int(len(grid)),
                "alpha_adjusted": alpha_adj}
    if best_rep.get("has_edge"):
        log.info("fire threshold %.4f cleared the evidence bar: %s", best_thr, best_rep["verdict"])
        return best_thr, best_rep

    log.info("no fire threshold cleared the evidence bar - the model will stay silent (%s)",
             best_rep.get("verdict"))
    return None, best_rep
