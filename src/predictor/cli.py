"""Tiny, dependency-free helper for the scripts/*.py entrypoints.

``config.py`` resolves every path at import time from the ``PREDICTOR_INSTRUMENT``
env var, so a script that wants to accept ``--instrument`` on the command line has to
set that env var *before* anything else in the ``predictor`` package gets imported.
This module is a leaf (imports nothing else from ``predictor``), so importing it first
is always safe.

Usage, at the very top of a script, before any other ``predictor`` import::

    from predictor.cli import resolve_instrument
    resolve_instrument()

    from predictor.config import CONFIG
    ...
"""

from __future__ import annotations

import argparse
import os


def resolve_instrument(default: str = "NIFTY50") -> str:
    """Peek at ``--instrument`` in argv (without consuming it) and set
    PREDICTOR_INSTRUMENT accordingly. Safe to call before the script's own argparse
    runs - argparse ignores arguments it doesn't ask for from a value that's already
    just an env var by the time it looks.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--instrument", default=os.environ.get("PREDICTOR_INSTRUMENT", default))
    ns, _ = pre.parse_known_args()
    os.environ["PREDICTOR_INSTRUMENT"] = ns.instrument
    return ns.instrument
