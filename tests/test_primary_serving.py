"""Serving-time primary probabilities, including old bundles without fold models."""

import numpy as np

from predictor.models.primary import predict_proba


class _Stub:
    """Stands in for a fitted LGBMClassifier."""

    def __init__(self, row):
        self.row = np.asarray(row, dtype="float64")

    def predict_proba(self, X):
        return np.tile(self.row, (len(X), 1))


X = np.zeros((4, 3))


def test_falls_back_to_the_single_model_for_bundles_without_folds():
    bundle = {"primary": _Stub([0.2, 0.5, 0.3])}          # pre-ensemble bundle
    out = predict_proba(bundle, X)
    assert out.shape == (4, 3)
    assert np.allclose(out[0], [0.2, 0.5, 0.3])


def test_empty_fold_list_also_falls_back():
    bundle = {"primary": _Stub([0.1, 0.8, 0.1]), "primary_folds": []}
    assert np.allclose(predict_proba(bundle, X)[0], [0.1, 0.8, 0.1])


def test_fold_models_are_averaged_when_present():
    bundle = {
        "primary": _Stub([1.0, 0.0, 0.0]),                 # must NOT be used
        "primary_folds": [_Stub([0.0, 1.0, 0.0]), _Stub([0.0, 0.0, 1.0])],
    }
    out = predict_proba(bundle, X)
    assert np.allclose(out[0], [0.0, 0.5, 0.5])


def test_rows_are_preserved():
    bundle = {"primary_folds": [_Stub([0.3, 0.4, 0.3])], "primary": _Stub([0, 0, 1])}
    assert predict_proba(bundle, np.zeros((7, 3))).shape == (7, 3)
