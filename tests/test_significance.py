"""The evidence gate: the system may only fire when it has actually shown an edge."""

import numpy as np
import pandas as pd

from predictor.validation.significance import (
    choose_fire_threshold,
    day_block_bootstrap,
    edge_report,
    naive_majority_side,
)


def _calls(n_days, per_day, pred, label, day0="2026-01-05"):
    """Build a fired-calls frame; pred/label may be scalars or per-row callables."""
    rows = []
    for d in range(n_days):
        day = pd.Timestamp(day0) + pd.Timedelta(days=d)
        for i in range(per_day):
            rows.append({
                "day": day,
                "primary_pred": pred(d, i) if callable(pred) else pred,
                "label": label(d, i) if callable(label) else label,
                "ret_at_touch": 0.004,
            })
    return pd.DataFrame(rows)


def test_naive_baseline_is_the_majority_direction():
    assert naive_majority_side(np.array([-1, -1, -1, 1])) == -1
    assert naive_majority_side(np.array([1, 1, -1])) == 1


def test_day_block_bootstrap_widens_with_correlated_rows():
    """Rows repeated within a day carry no extra information, so the interval must
    not shrink the way a row-wise bootstrap would."""
    few_days = _calls(4, 25, 1, lambda d, i: 1 if d % 2 else -1)
    many_days = _calls(100, 1, 1, lambda d, i: 1 if d % 2 else -1)
    stat = lambda z: float((z["primary_pred"] == z["label"]).mean())  # noqa: E731
    lo_few, _, hi_few = day_block_bootstrap(few_days, stat, n_boot=300)
    lo_many, _, hi_many = day_block_bootstrap(many_days, stat, n_boot=300)
    assert (hi_few - lo_few) > (hi_many - lo_many)


def test_no_edge_when_the_model_only_matches_the_drift():
    """Always naming 'down' in a market that mostly falls is not skill."""
    fired = _calls(30, 4, -1, lambda d, i: -1 if i < 3 else 1)
    rep = edge_report(fired, n_boot=300)
    assert rep["naive_side"] == "down"
    assert np.isclose(rep["precision"], rep["naive_precision"])
    assert rep["has_edge"] is False


def test_real_edge_is_detected():
    # model is right far more often than always-say-down would be
    fired = _calls(40, 4, lambda d, i: 1 if i % 2 else -1,
                   lambda d, i: 1 if i % 2 else -1)
    rep = edge_report(fired, n_boot=300)
    assert rep["precision"] == 1.0
    assert rep["has_edge"] is True
    assert rep["margin_ci"][0] > 0


def test_insufficient_data_is_not_an_edge():
    fired = _calls(3, 3, 1, 1)          # perfect, but only 3 days
    rep = edge_report(fired, n_boot=200, min_trades=30, min_days=15)
    assert rep["has_edge"] is False
    assert "not enough evidence" in rep["verdict"]


def test_choose_fire_threshold_returns_none_without_evidence():
    fired = _calls(30, 4, -1, lambda d, i: -1 if i < 3 else 1)
    fired["meta_score"] = np.linspace(0, 1, len(fired))
    thr, rep = choose_fire_threshold(fired, n_boot=200)
    assert thr is None
    assert rep["has_edge"] is False


def test_choose_fire_threshold_finds_a_cutoff_when_one_works():
    """High meta_score rows are right, low ones are coin flips - the gate should
    find the cutoff rather than giving up."""
    rng = np.random.default_rng(3)
    rows = []
    for d in range(40):
        day = pd.Timestamp("2026-01-05") + pd.Timedelta(days=d)
        for i in range(4):
            good = i >= 2
            pred = 1 if rng.random() < 0.5 else -1
            label = pred if good else (pred if rng.random() < 0.5 else -pred)
            rows.append({"day": day, "primary_pred": pred, "label": label,
                         "meta_score": 0.9 if good else 0.1, "ret_at_touch": 0.004})
    thr, rep = choose_fire_threshold(pd.DataFrame(rows), n_boot=300)
    assert thr is not None
    assert rep["has_edge"] is True


def test_empty_input_is_handled():
    empty = pd.DataFrame(columns=["day", "primary_pred", "label"])
    rep = edge_report(empty)
    assert rep["n_fired"] == 0 and rep["has_edge"] is False
    thr, _ = choose_fire_threshold(empty.assign(meta_score=[]))
    assert thr is None


def test_breakeven_bar_blocks_a_real_but_unprofitable_edge():
    """Beating the naive baseline is not enough if costs still eat the result."""
    from predictor.validation.significance import breakeven_precision

    rows = []
    rng = np.random.default_rng(5)
    for d in range(40):
        day = pd.Timestamp("2026-01-05") + pd.Timedelta(days=d)
        for i in range(6):
            # barriers so tight that break-even precision is pushed very high
            pred = 1 if rng.random() < 0.5 else -1
            label = pred if i == 0 else 0
            rows.append({"day": day, "primary_pred": pred, "label": label,
                         "ret_at_touch": 0.0001, "sigma_effective": 0.00005, "k": 1.0})
    f = pd.DataFrame(rows)
    be = breakeven_precision(f)
    assert be > 0.5, f"tight barriers should demand high precision, got {be}"
    rep = edge_report(f, n_boot=300)
    assert rep["clears_breakeven"] is False
    assert rep["has_edge"] is False


def test_breakeven_is_nan_without_barrier_columns():
    from predictor.validation.significance import breakeven_precision
    f = pd.DataFrame({"day": [1], "primary_pred": [1], "label": [1]})
    assert np.isnan(breakeven_precision(f))
