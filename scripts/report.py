"""One place to see everything: dataset, model, walk-forward CV, backtest, paper log.

    python scripts/report.py            # full report
    python scripts/report.py --signals  # also list recent fired / directional calls
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from predictor.config import CONFIG

pd.set_option("display.width", 200, "display.max_columns", 40, "display.max_rows", 60)


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _hr(title: str) -> None:
    print(f"\n{'=' * 3} {title} {'=' * (72 - len(title))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", action="store_true", help="list recent directional / fired calls")
    args = ap.parse_args()
    p = CONFIG.paths

    _hr("DATASET")
    meta = _load_json(p.processed / "dataset_meta.json")
    if meta:
        print(f"rows={meta['rows']}  days={meta['days']}  k={meta['k']}  "
              f"time_scaling={meta['barrier_time_scaling']}")
        print(f"label counts (dn/to/up): {meta['label_counts']}")
        print(f"date range: {meta['date_range'][0]}  ->  {meta['date_range'][1]}")
        print(f"features: {len(meta['feature_columns'])}")
    else:
        print("no dataset - run scripts/build_dataset.py")

    _hr("MODEL CARD")
    card = _load_json(p.models_dir / "model_card.json")
    if card:
        print(f"trained_at    : {card['trained_at']}")
        print(f"train_data_end: {card.get('train_data_end')}")
        print(f"fire_threshold: {card.get('fire_threshold')}   fire_top_fraction: {card['fire_top_fraction']}")
        hc = card.get("metrics", {}).get("high_confidence", {})
        dirn = card.get("metrics", {}).get("directional", {})
        print(f"OOF directional : n_fired={dirn.get('n_fired')}  precision={dirn.get('precision')}")
        print(f"OOF high-conf   : n_fired={hc.get('n_fired')}  precision={hc.get('precision')}  "
              f"fires/day={hc.get('fires_per_day')}")
    else:
        print("no model - run scripts/train.py")

    _hr("WALK-FORWARD CV (per-class, out-of-fold)")
    summary = _load_json(p.reports_dir / "cv" / "summary.json")
    if summary:
        pc = summary.get("per_class", {})
        if pc:
            print(pd.DataFrame(pc).T.rename(index={"-1": "down", "0": "timeout", "1": "up"})
                  .round(3).to_string())
        print(f"\nn_samples={summary['n_samples']}  n_days={summary['n_days']}")

    _hr("BACKTEST (threshold sweep on OOF trades)")
    bt = _load_json(p.reports_dir / "backtest.json")
    if bt:
        sweep = pd.DataFrame(bt["sweep"]).set_index("top_fraction")
        cols = [c for c in ["n_trades", "trades_per_day", "directional_precision", "win_rate",
                            "avg_net_ret", "total_net_ret", "per_trade_ir", "max_drawdown"]
                if c in sweep.columns]
        print(sweep[cols].round(4).to_string())
        print(f"\nheadline (top {bt['top_fraction']:.0%}): "
              f"{ {k: (round(v, 4) if isinstance(v, float) else v) for k, v in bt['headline'].items()} }")
    else:
        print("no backtest - run scripts/backtest.py")

    _hr("PAPER TRADING (out-of-sample only)")
    log_path = p.reports_dir / "paper_trades.parquet"
    if log_path.exists():
        from predictor.papertrade import summarize
        print(json.dumps(summarize(), indent=2, default=str))
    else:
        print("no paper log - run scripts/paper_log.py")

    if args.signals:
        _hr("RECENT CALLS")
        oof_path = p.reports_dir / "cv" / "oof.parquet"
        if oof_path.exists():
            oof = pd.read_parquet(oof_path)
            d = oof[oof["primary_pred"] != 0].tail(25)
            if len(d):
                show = ["day", "primary_pred", "proba_up", "proba_dn", "meta_score", "label"]
                print(d[[c for c in show if c in d.columns]].round(3).to_string())
            else:
                print("the model made no directional calls in the OOF set")


if __name__ == "__main__":
    main()
