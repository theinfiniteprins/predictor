"""Markers for tests that land with their phase. Keeps the intent visible."""

import pytest


@pytest.mark.skip(reason="Phase 3 — triple_barrier.py not implemented yet")
def test_triple_barrier_first_touch_wins():
    ...


@pytest.mark.skip(reason="Phase 3 — intrabar tie broken with 1-min bars, else dropped")
def test_triple_barrier_intrabar_ambiguity_dropped():
    ...


@pytest.mark.skip(reason="Phase 4 — purged_cv.py not implemented yet")
def test_purged_cv_has_no_train_test_overlap_across_embargo():
    ...


@pytest.mark.skip(reason="Phase 2 — feature at time T must use only data <= T")
def test_features_have_no_lookahead():
    ...
