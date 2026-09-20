"""Guards that stop a lucky result from switching on live calls.

The gate is re-run on every nightly retrain, so a single pass at alpha=0.05 means
very little on its own. Two things have to happen before anything fires: the edge has
to replicate on sessions that had no say in choosing the threshold, and it has to
survive several retrains in a row.
"""

import json
import types

import numpy as np
import pandas as pd
import pytest

from predictor.pipeline import _require_consecutive_passes
from predictor.validation.significance import choose_fire_threshold


def _rows(day_from, day_to, skilled, seed=0, score_hi=0.9):
    """Directional calls; `skilled` decides whether the high-score half is right."""
    rng = np.random.default_rng(seed)
    out = []
    for d in range(day_from, day_to):
        day = pd.Timestamp("2026-01-05") + pd.Timedelta(days=d)
        for i in range(4):
            hi = i >= 2
            pred = 1 if rng.random() < 0.5 else -1
            if hi and skilled:
                label = pred
            else:
                label = pred if rng.random() < 0.5 else -pred
            out.append({"day": day, "primary_pred": pred, "label": label,
                        "meta_score": score_hi if hi else 0.1, "ret_at_touch": 0.004})
    return out


def test_edge_that_does_not_replicate_is_refused():
    """Skill in the selection window, noise in the confirmation window -> silent."""
    df = pd.DataFrame(_rows(0, 60, skilled=True, seed=1) +
                      _rows(60, 80, skilled=False, seed=2))
    thr, rep = choose_fire_threshold(df, n_boot=300, confirm_days=20)
    assert thr is None
    assert rep["has_edge"] is False
    assert "confirm" in rep["verdict"].lower() or "replicate" in rep["verdict"].lower()


def test_edge_that_replicates_passes_selection_and_confirmation():
    df = pd.DataFrame(_rows(0, 80, skilled=True, seed=3))
    thr, rep = choose_fire_threshold(df, n_boot=300, confirm_days=20)
    assert thr is not None
    assert rep["has_edge"] is True
    assert rep["confirmation"]["margin"] > 0


def test_too_little_history_to_split_stays_silent():
    df = pd.DataFrame(_rows(0, 10, skilled=True, seed=4))
    thr, _ = choose_fire_threshold(df, n_boot=200, confirm_days=20)
    assert thr is None


@pytest.fixture
def card(tmp_path, monkeypatch):
    """Point the pass counter at a throwaway model card."""
    import predictor.pipeline as pipeline
    stub = types.SimpleNamespace(paths=types.SimpleNamespace(models_dir=tmp_path))
    monkeypatch.setattr(pipeline, "CONFIG", stub)

    def write(passes):
        (tmp_path / "model_card.json").write_text(
            json.dumps({"fire_gate": {"consecutive_passes": passes}}), encoding="utf-8")
    return write


def test_first_pass_does_not_fire(card):
    card(0)
    thr, gate = _require_consecutive_passes(0.7, {"verdict": "edge"}, min_passes=3)
    assert thr is None
    assert gate["consecutive_passes"] == 1
    assert "1 of 3" in gate["verdict"]


def test_fires_once_the_streak_is_long_enough(card):
    card(2)
    thr, gate = _require_consecutive_passes(0.7, {"verdict": "edge"}, min_passes=3)
    assert thr == 0.7
    assert gate["consecutive_passes"] == 3


def test_a_failed_run_resets_the_streak(card):
    card(5)
    thr, gate = _require_consecutive_passes(None, {"verdict": "no edge"}, min_passes=3)
    assert thr is None
    assert gate["consecutive_passes"] == 0


def test_missing_or_corrupt_card_starts_from_zero(card, tmp_path):
    thr, gate = _require_consecutive_passes(0.7, {}, min_passes=3)   # no card at all
    assert thr is None and gate["consecutive_passes"] == 1

    (tmp_path / "model_card.json").write_text("{not json", encoding="utf-8")
    thr, gate = _require_consecutive_passes(0.7, {}, min_passes=3)
    assert thr is None and gate["consecutive_passes"] == 1
