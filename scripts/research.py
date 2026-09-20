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
        "sigma_effective": d["sigma_effective"].to_numpy(),
        "k": d["k"].to_numpy(),
    })
    rep = edge_report(f, n_boot=n_boot, min_trades=30, min_days=10)

    def r(key, nd=4):
        v = rep.get(key, np.nan)
        return round(v, nd) if isinstance(v, (int, float)) and np.isfinite(v) else np.nan

    return {
        "variant": name,
        "fired": rep["n_fired"],
        "days": rep["n_days"],
        "precision": r("precision"),
        "naive": r("naive_precision"),
        "breakeven": r("breakeven_precision"),
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
    print("\nedge=True needs BOTH: the day-block bootstrap lower bound on margin over\n"
          "'always name the majority direction' above zero, AND precision clearing the\n"
          "break-even bar (costs). Beating the baseline while losing money is not an edge.")
    if not out["edge"].any():
        print("\nNo variant shows a real edge yet. The correct response is to keep\n"
              "collecting and keep firing nothing - not to loosen the threshold.")

    # How far off is 'knowing'? Compare the width of what we can measure against the
    # size of edge that would actually pay.
    ref = out[out["variant"] == "MODEL current defaults"]
    if len(ref):
        best = ref.iloc[0]
        if np.isfinite(best.get("breakeven", np.nan)) and np.isfinite(best.get("naive", np.nan)):
            need = best["breakeven"] - best["naive"]
            width = (best["margin_hi"] - best["margin_lo"]) if np.isfinite(best["margin_hi"]) else np.nan
            print(f"\nScale check on '{best['variant']}':")
            print(f"  edge needed to be worth trading : {100*need:+.1f}pp over naive")
            if np.isfinite(width):
                print(f"  width of what we can measure    : {100*width:.1f}pp")
                if width > abs(need) * 2:
                    print("  -> the measurement is far coarser than the edge that would pay.\n"
                          "     More calendar time alone is slow; more registered instruments\n"
                          "     adds independent calls per day and narrows this faster.")

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
