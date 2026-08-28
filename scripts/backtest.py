"""Phase 6 — backtest the fired signals from the walk-forward OOF predictions.

    python scripts/backtest.py                    # threshold sweep + config default
    python scripts/backtest.py --slippage-bps 2   # stress costs
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from predictor.config import CONFIG
from predictor.logging_setup import get_logger

log = get_logger("backtest_cli")


def main() -> None:
    ap = argparse.ArgumentParser(description="backtest fired signals")
    ap.add_argument("--slippage-bps", type=float, default=1.5, help="per side")
    ap.add_argument("--cost-bps", type=float, default=0.0, help="brokerage/STT per round trip")
    ap.add_argument("--top-fraction", type=float, default=None, help="override fire fraction")
    args = ap.parse_args()

    oof_path = CONFIG.paths.reports_dir / "cv" / "oof.parquet"
    if not oof_path.exists():
        ap.error(f"{oof_path} missing - run scripts/train.py first")
    oof = pd.read_parquet(oof_path)

    from predictor.backtest.engine import apply_fire_filter, evaluate, threshold_sweep

    sweep = threshold_sweep(oof, slippage_bps=args.slippage_bps, cost_bps=args.cost_bps)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    log.info("threshold sweep:\n%s", sweep[
        ["n_trades", "trades_per_day", "directional_precision", "win_rate",
         "avg_net_ret", "total_net_ret", "per_trade_ir", "max_drawdown"]
    ].round(4).to_string())

    frac = args.top_fraction or CONFIG.meta.fire_top_fraction
    fired = apply_fire_filter(oof, top_fraction=frac)
    headline = evaluate(fired, slippage_bps=args.slippage_bps, cost_bps=args.cost_bps,
                        n_days=oof["day"].nunique())
    out = CONFIG.paths.reports_dir / "backtest.json"
    out.write_text(json.dumps({"top_fraction": frac, "headline": headline,
                               "sweep": sweep.reset_index().to_dict(orient="records")},
                              indent=2, default=str), encoding="utf-8")
    log.info("headline (top %.0f%%): %s", 100 * frac, {k: round(v, 4) if isinstance(v, float) else v
                                                       for k, v in headline.items()})
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
