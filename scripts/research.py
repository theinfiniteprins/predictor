"""Honest model research: does anything here beat a no-skill baseline?

Every row of the output is scored the same way - purged walk-forward out-of-fold
predictions, compared against the naive "always name the majority direction"
strategy, with a day-block bootstrap interval (rolling entries overlap, so row-wise
intervals are far too narrow). A variant only counts as real if `edge` is True.

    python scripts/research.py                     # baselines + model variants
    python scripts/research.py --k-sweep           # also re-label at several k (slow)
    python scripts/research.py --instrument LAURUSLABS

Re-run this as the collector accumulates history: the point is to catch the moment a
configuration genuinely starts beating the baseline, instead of guessing.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from predictor.cli import resolve_instrument
resolve_instrument()

from predictor.config import CONFIG
from predictor.logging_setup import get_logger

log = get_logger("research")

pd.set_option("display.width", 220, "display.max_columns", 40)


def _score(name, pred, d, n_boot):
    from predictor.validation.significance import edge_report
    f = pd.DataFrame({
        "primary_pred": pred, "label": d["label"].to_numpy(),
        "day": d["day"].to_numpy(), "ret_at_touch": d["ret_at_touch"].to_numpy(),
    })
    rep = edge_report(f, n_boot=n_boot, min_trades=30, min_days=10)
    return {
        "variant": name,
        "fired": rep["n_fired"],
        "days": rep["n_days"],
        "precision": round(rep["precision"], 4) if np.isfinite(rep.get("precision", np.nan)) else np.nan,
        "naive": round(rep["naive_precision"], 4) if np.isfinite(rep.get("naive_precision", np.nan)) else np.nan,
        "margin": round(rep["margin"], 4) if np.isfinite(rep.get("margin", np.nan)) else np.nan,
        "margin_lo": round(rep["margin_ci"][0], 4) if np.isfinite(rep["margin_ci"][0]) else np.nan,
        "margin_hi": round(rep["margin_ci"][1], 4) if np.isfinite(rep["margin_ci"][1]) else np.nan,
        "net_ret_%": round(100 * rep.get("net_return_total", np.nan), 3),
        "edge": rep["has_edge"],
    }


def _walk_forward_pred(X, y, t_entry, t_end, params=None, use_uniqueness=True):
    from predictor.models.primary import fit_walk_forward
    res = fit_walk_forward(X, y, t_entry, t_end, params=params, use_uniqueness=use_uniqueness)
    return res.oof["primary_pred"].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description="honest edge research")
    ap.add_argument("--k-sweep", action="store_true", help="re-label at several barrier k (slow)")
    ap.add_argument("--k-values", default="0.5,0.6,0.8,1.0,1.5")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--instrument", default=None, help="instrument key from instruments.yaml")
    args = ap.parse_args()

    from predictor.dataset import load as load_dataset, split_xy
    from predictor.labeling.uniqueness import average_uniqueness

    d = load_dataset()
    X, y, t_entry, t_end = split_xy(d)
    rows = []

    uniq = average_uniqueness(t_entry, t_end)
    print(f"\ninstrument     : {CONFIG.instrument.name}")
    print(f"rows / days    : {len(d)} / {d['day'].nunique()}")
    print(f"features       : {X.shape[1]}")
    print(f"label mix      : {y.value_counts().sort_index().to_dict()}")
    print(f"mean uniqueness: {uniq.mean():.4f}  ->  ~{uniq.sum():.0f} independent "
          f"observations, not {len(d)}")
    print("   (that ratio is why every interval below is a day-block bootstrap)\n")

    # ---- reference points a model has to beat -------------------------------
    rng = np.random.default_rng(0)
    coin = np.zeros(len(y))
    pick = rng.choice(len(y), size=max(int(0.05 * len(y)), 1), replace=False)
    coin[pick] = rng.choice([-1, 1], size=len(pick))
    rows.append(_score("BASELINE coin flip (5% of rows)", coin, d, args.n_boot))

    if "ret_since_open" in X.columns:
        v = X["ret_since_open"].to_numpy()
        c = v - np.nanmedian(v)
        thr = np.nanquantile(np.abs(c), 0.90)
        rows.append(_score("BASELINE momentum (ret_since_open)",
                           np.where(np.abs(c) >= thr, np.sign(c), 0.0), d, args.n_boot))
        rows.append(_score("BASELINE mean-reversion",
                           np.where(np.abs(c) >= thr, -np.sign(c), 0.0), d, args.n_boot))

    # ---- model variants ------------------------------------------------------
    rows.append(_score("MODEL current defaults",
                       _walk_forward_pred(X, y, t_entry, t_end), d, args.n_boot))
    rows.append(_score("MODEL without uniqueness weights",
                       _walk_forward_pred(X, y, t_entry, t_end, use_uniqueness=False),
                       d, args.n_boot))

    per_day = X.groupby(d["day"].values).nunique().mean()
    intraday = per_day[per_day > 1.0001].index.tolist()
    if intraday and len(intraday) < X.shape[1]:
        rows.append(_score(f"MODEL intraday features only ({len(intraday)})",
                           _walk_forward_pred(X[intraday], y, t_entry, t_end),
                           d, args.n_boot))

    heavy = dict(num_leaves=4, min_child_samples=60, n_estimators=200, reg_alpha=5.0,
                 reg_lambda=10.0, learning_rate=0.02, colsample_bytree=0.5)
    rows.append(_score("MODEL heavy regularisation",
                       _walk_forward_pred(X, y, t_entry, t_end, params=heavy),
                       d, args.n_boot))

    # ---- barrier geometry ----------------------------------------------------
    if args.k_sweep:
        from predictor.features.build import build_features, feature_columns
        from predictor.labeling.triple_barrier import build_labels
        feats = build_features()
        for k in [float(s) for s in args.k_values.split(",")]:
            lab = build_labels(k=k)
            dk = feats.join(lab, how="inner")
            dk = dk[dk["label"].notna()].copy()
            dk["label"] = dk["label"].astype(int)
            dk["day"] = pd.to_datetime(dk["day"])
            dk["t_touch"] = pd.to_datetime(dk["t_touch"])
            dk = dk.sort_index()
            cols = feature_columns(dk)
            pred = _walk_forward_pred(dk[cols].astype("float64"), dk["label"].astype(int),
                                      dk.index.to_series(), dk["t_touch"])
            r = _score(f"BARRIER k={k}", pred, dk, args.n_boot)
            r["timeout_%"] = round(100 * float((dk["label"] == 0).mean()), 1)
            rows.append(r)

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print("\nedge=True means: beat 'always name the majority direction' with the\n"
          "day-block bootstrap lower bound above zero. Anything else is not yet real.")
    if not out["edge"].any():
        print("\nNo variant shows a real edge yet. The correct response is to keep\n"
              "collecting and keep firing nothing - not to loosen the threshold.")

    path = CONFIG.paths.reports_dir / "research.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "instrument": CONFIG.instrument.name,
        "generated_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "rows": len(d), "days": int(d["day"].nunique()),
        "mean_uniqueness": float(uniq.mean()),
        "effective_observations": float(uniq.sum()),
        "results": out.to_dict(orient="records"),
    }, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", path)


if __name__ == "__main__":
    main()
