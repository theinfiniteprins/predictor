"""The real-money checklist. Every check must pass; the default answer is "no"."""

import numpy as np
import pandas as pd

from predictor.validation.readiness import (
    MIN_LIVE_CALLS,
    MIN_LIVE_DAYS,
    readiness_report,
)


def _card(firing=True, clears=True, passes=3, need=3):
    return {
        "fire_threshold": 0.7 if firing else None,
        "edge": {"clears_breakeven": clears, "breakeven_precision": 0.08},
        "fire_gate": {"consecutive_passes": passes, "passes_required": need},
    }


def _paper(n_days, per_day, net_per_call, in_sample=False, fired=True):
    rows = []
    for d in range(n_days):
        day = pd.Timestamp("2026-01-05") + pd.Timedelta(days=d)
        for _ in range(per_day):
            rows.append({
                "day": day, "in_sample": in_sample, "fired": fired,
                "primary_pred": 1, "label": 1,
                # net = pred*ret - cost, so ret must carry the cost back
                "ret_at_touch": net_per_call + 3e-4,
            })
    return pd.DataFrame(rows)


def test_nothing_is_ready_without_a_model():
    r = readiness_report(None, None, n_boot=200)
    assert r["ready"] is False
    assert r["passed"] == 0
    assert len(r["blockers"]) == r["total"]


def test_closed_gate_blocks_everything_downstream():
    r = readiness_report(_card(firing=False), None, n_boot=200)
    assert r["ready"] is False
    assert "Model has proven an edge on historical data" in r["blockers"]


def test_backtest_alone_is_never_enough():
    """Gate open, costs cleared, streak met - but no forward record. Still no."""
    r = readiness_report(_card(), None, n_boot=200)
    assert r["ready"] is False
    assert any("forward calls" in b for b in r["blockers"])


def test_in_sample_paper_rows_do_not_count_as_forward_evidence():
    paper = _paper(MIN_LIVE_DAYS + 10, 3, 0.002, in_sample=True)
    r = readiness_report(_card(), paper, n_boot=200)
    assert r["ready"] is False
    assert any("forward calls" in b for b in r["blockers"])


def test_unfired_rows_do_not_count_either():
    paper = _paper(MIN_LIVE_DAYS + 10, 3, 0.002, fired=False)
    r = readiness_report(_card(), paper, n_boot=200)
    assert r["ready"] is False


def test_losing_forward_record_is_not_ready():
    paper = _paper(MIN_LIVE_DAYS + 10, 3, -0.001)
    r = readiness_report(_card(), paper, n_boot=400)
    assert r["ready"] is False
    assert any("profitable" in b for b in r["blockers"])


def test_a_genuinely_good_record_does_pass():
    """The bar must be reachable - a gate that can never open is just as useless."""
    paper = _paper(MIN_LIVE_DAYS + 20, 3, 0.003)
    r = readiness_report(_card(), paper, n_boot=400)
    assert r["ready"] is True, r["blockers"]
    assert r["passed"] == r["total"]
    assert "not financial advice" in r["detail"]


def test_enough_calls_but_too_few_sessions_fails():
    """Many calls crammed into a handful of days is not a track record."""
    paper = _paper(5, MIN_LIVE_CALLS, 0.003)
    r = readiness_report(_card(), paper, n_boot=200)
    assert r["ready"] is False
    assert any("forward calls" in b for b in r["blockers"])


def test_report_always_explains_the_first_blocker():
    r = readiness_report(_card(firing=False), None, n_boot=200)
    assert r["blockers"][0] in r["detail"]
    for c in r["checks"]:
        if not c["passed"]:
            assert c["how_to_fix"], f"{c['name']} gives no guidance"
