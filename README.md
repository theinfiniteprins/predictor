# Intraday Triple-Barrier Direction Predictor (NSE Nifty 50)

Personal-use tool that predicts, at rolling intraday entry points, whether Nifty 50 will
hit an **upper** target, a **lower** target, or **neither** (timeout) before 15:20 IST —
and only surfaces a call when a meta-labeling confidence filter says the primary model's
prediction is trustworthy.

See `intraday-prediction-tool-plan.md` (kept in Downloads) for the full design rationale.

## Status

| Phase | What | State |
|---|---|---|
| 0 | Finalize spec | ✅ done — see `config.yaml` |
| 1 | Data pipeline (Track A backfill + Track B live collector) | 🚧 in progress |
| 2 | Feature engineering | not started |
| 3 | Triple-barrier labeling | not started |
| 4 | Baseline primary model + purged walk-forward CV | not started |
| 5 | Meta-labeling confidence filter | not started |
| 6 | Backtest, tune k + confidence threshold | not started |
| 7 | Paper-trade 4–8 weeks | not started |
| 8 | (optional) sequence models / ensembling on larger dataset | not started |

## Phase 0 parameters (finalized)

- **Instrument:** NIFTY 50 index, `^NSEI` (yfinance)
- **Barrier multiplier k:** 1.0, symmetric (swept 0.5–2.0 later in Phase 6)
- **σ_intraday:** ATR(14) on daily bars ÷ close, held constant across the day, prior sessions only
- **Entry cadence:** every 15 min, first 09:30, last 14:30 IST (~21 entries/day)
- **Vertical barrier:** 15:20 IST
- **Bar interval:** 5-min (Track A / yfinance ≈60d history); Track B poller every ~90s
- **Timezone:** Asia/Kolkata; NSE trading-holiday calendar applied
- **Intrabar ambiguity:** break ties with 1-min bars, else drop the instance
- **Barriers:** symmetric

## Setup (Windows, Python 3.12)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# Track B — the live collector. Run this every trading day (or via Task Scheduler).
python scripts/run_collector.py

# Track A — one-shot historical backfill (run whenever; re-runnable)
python scripts/run_backfill.py --all
```

## Layout

```
src/predictor/      all logic (importable, tested)
  config.py         loads config.yaml -> frozen dataclasses
  calendar.py       NSE trading days / session bounds
  storage.py        parquet read/write, append-safe
  data/             Phase 1  — backfill + live collector
  labeling/         Phase 3  — σ_intraday, entry grid, triple-barrier
  features/         Phase 2  — technical / context / global / time-of-day
  validation/       Phase 4  — purged+embargoed walk-forward CV, metrics
  models/           Phase 4-5 — primary, meta, tuning
  backtest/         Phase 6  — true triple-barrier exits, costs
scripts/            thin CLI entrypoints
data/               local data lake (gitignored)
tests/              triple-barrier, purged-CV, no-look-ahead
```

## Not investment advice

Personal decision-support only. No automated order placement. See §12 of the plan.
