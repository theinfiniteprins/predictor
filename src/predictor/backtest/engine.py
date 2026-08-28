"""Backtest the fired signals through the real triple-barrier outcome.

Inputs come from the walk-forward OOF table (so every trade is on data the models
never trained on). Each labeled entry already carries how the trade resolved
(`ret_at_touch`, `reason`), so there is no re-simulation of price paths and no
look-ahead: we just apply direction, costs, and the fire filter.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CONFIG
from ..logging_setup import get_logger

log = get_logger("backtest")

DEFAULT_SLIPPAGE_BPS = 1.5   # per side
DEFAULT_COST_BPS = 0.0       # brokerage/STT - personal use, prediction only


def _trade_returns(df: pd.DataFrame, slippage_bps: float, cost_bps: float) -> pd.Series:
    gross = df["primary_pred"] * df["ret_at_touch"]
    costs = (2 * slippage_bps + cost_bps) / 1e4
    return (gross - costs).rename("net_ret")


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity - peak)) if len(equity) else 0.0


def evaluate(
    fired: pd.DataFrame,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    cost_bps: float = DEFAULT_COST_BPS,
    n_days: int | None = None,
) -> dict:
    if fired.empty:
        return {"n_trades": 0}

    r = _trade_returns(fired, slippage_bps, cost_bps).to_numpy()
    correct = (fired["primary_pred"] == fired["label"]).to_numpy()
    equity = np.cumsum(r)
    n_days = n_days or fired["day"].nunique()

    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    return {
        "n_trades": int(len(r)),
        "trades_per_day": len(r) / n_days if n_days else np.nan,
        "directional_precision": float(correct.mean()),
        "win_rate": float((r > 0).mean()),
        "avg_net_ret": float(r.mean()),
        "median_net_ret": float(np.median(r)),
        "total_net_ret": float(r.sum()),
        "ret_std": float(r.std(ddof=1)) if len(r) > 1 else np.nan,
        "per_trade_ir": float(r.mean() / r.std(ddof=1)) if len(r) > 1 and r.std() else np.nan,
        "profit_factor": float(pos / neg) if neg else np.inf,
        "max_drawdown": _max_drawdown(equity),
        "hit_up": int(np.sum(correct & (fired["primary_pred"] == 1).to_numpy())),
        "hit_down": int(np.sum(correct & (fired["primary_pred"] == -1).to_numpy())),
    }


def apply_fire_filter(
    oof: pd.DataFrame, threshold: float | None = None, top_fraction: float | None = None
) -> pd.DataFrame:
    """Rows the system would actually trade: a directional primary call that clears
    the meta-score bar (absolute ``threshold`` or the top ``top_fraction``)."""
    cand = oof[(oof["primary_pred"] != 0) & oof["primary_pred"].notna() & oof["meta_score"].notna()]
    if cand.empty:
        return cand
    if threshold is not None:
        return cand[cand["meta_score"] >= threshold]
    frac = top_fraction if top_fraction is not None else CONFIG.meta.fire_top_fraction
    cutoff = cand["meta_score"].quantile(1 - frac)
    return cand[cand["meta_score"] >= cutoff]


def threshold_sweep(
    oof: pd.DataFrame,
    fractions=(0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0),
    **kw,
) -> pd.DataFrame:
    n_days = oof["day"].nunique()
    out = []
    for f in fractions:
        fired = apply_fire_filter(oof, top_fraction=f)
        m = evaluate(fired, n_days=n_days, **kw)
        m["top_fraction"] = f
        out.append(m)
    return pd.DataFrame(out).set_index("top_fraction")
