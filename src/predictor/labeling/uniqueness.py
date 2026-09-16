"""Average label uniqueness (Lopez de Prado) - how much independent information
each training row actually carries.

A triple-barrier label spans [t_entry, t_touch]. Rolling entries every 15 minutes
that nearly all resolve at the same 15:20 vertical barrier overlap almost completely:
measured on this dataset the mean uniqueness is ~0.08, i.e. ~1500 rows carry roughly
115 independent observations.

Feeding those rows to a learner unweighted tells it that one day's move is ~21
separate pieces of evidence, which inflates its confidence and lets it overfit a
single session. Weighting each row by its uniqueness corrects that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _aligned(t_entry, t_end) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    ei = pd.DatetimeIndex(t_entry)
    xi = pd.DatetimeIndex(t_end)
    if ei.tz is not None and xi.tz is not None:
        xi = xi.tz_convert(ei.tz)
    return ei, xi


def _grid_and_concurrency(ei: pd.DatetimeIndex, xi: pd.DatetimeIndex):
    """Event grid plus live-label count at each point.

    Works on the underlying int64/UTC values so tz handling stays in one place -
    `.values` on a tz-aware index drops the zone, which is fine for arithmetic but
    must not leak into anything returned to a caller.
    """
    ev, xv = ei.values, xi.values
    grid = np.unique(np.concatenate([ev, xv]))
    delta = np.zeros(len(grid) + 1, dtype="float64")
    np.add.at(delta, np.searchsorted(grid, ev, side="left"), 1.0)
    np.add.at(delta, np.searchsorted(grid, xv, side="left") + 1, -1.0)
    return grid, np.cumsum(delta[:-1])


def concurrency(t_entry, t_end) -> pd.Series:
    """How many labels are live at each event timestamp."""
    ei, xi = _aligned(t_entry, t_end)
    grid, conc = _grid_and_concurrency(ei, xi)
    idx = pd.DatetimeIndex(grid)
    if ei.tz is not None:
        idx = idx.tz_localize("UTC").tz_convert(ei.tz)
    return pd.Series(conc, index=idx, name="concurrency")


def average_uniqueness(t_entry, t_end) -> np.ndarray:
    """Per-row mean of 1/concurrency over the row's own [t_entry, t_touch] span.

    Returns raw uniqueness in (0, 1]; 1.0 means the label overlaps nothing.
    """
    ei, xi = _aligned(t_entry, t_end)
    grid, conc = _grid_and_concurrency(ei, xi)

    inv = np.where(conc > 0, 1.0 / np.maximum(conc, 1e-12), 0.0)
    prefix = np.concatenate([[0.0], np.cumsum(inv)])

    lo = np.searchsorted(grid, ei.values, side="left")
    hi = np.searchsorted(grid, xi.values, side="right")     # exclusive end
    span = np.maximum(hi - lo, 1)
    return (prefix[hi] - prefix[lo]) / span


def uniqueness_weights(t_entry, t_end) -> np.ndarray:
    """Average uniqueness normalised to mean 1.0, ready to use as sample_weight.

    Normalising keeps the effective weight scale comparable to unweighted fitting, so
    regularisation strengths tuned without it stay in the same ballpark.
    """
    u = average_uniqueness(t_entry, t_end)
    m = float(np.mean(u))
    return u / m if m > 0 else np.ones_like(u)
