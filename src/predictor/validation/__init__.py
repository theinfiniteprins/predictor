"""Phase 4 — validation (not yet implemented).

Planned modules:
    purged_cv.py  purged + embargoed walk-forward splitter (Lopez de Prado);
                  purge train samples whose label window overlaps the test fold,
                  embargo `cv.embargo_minutes` after each test fold.
    metrics.py    per-class precision/recall; precision on the post-meta-filter
                  high-confidence subset (the number that actually matters).
"""
