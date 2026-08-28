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
| 2 | Feature engineering (`features/`) | ✅ built |
| 3 | Triple-barrier labeling (`labeling/`) | ✅ built |
| 4 | Primary model + purged/embargoed walk-forward CV | ✅ built |
| 5 | Meta-labeling confidence filter | ✅ built |
| 6 | Backtest, threshold sweep | ✅ built |
| 7 | Paper-trade 4–8 weeks | pending data depth |
| 8 | (optional) sequence models / ensembling on larger dataset | not started |

> **Data depth, not code, is the blocker now.** The full pipeline runs end-to-end,
> but with only ~59 days of yfinance history (~105 directional labels at k=1.0) the
> models cannot yet learn a real edge — the primary just predicts "timeout". This
> is expected (see §11 of the plan). Keep the cloud collector running; revisit
> training in 2–3 months. `k=0.6` produces enough directional labels to exercise
> the meta path in the meantime.

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
# Full modelling pipeline
python scripts/build_dataset.py            # features + triple-barrier labels -> data/processed/
python scripts/train.py --tune 200         # walk-forward primary + meta, Optuna search, save bundle
python scripts/backtest.py                 # threshold sweep on the OOF predictions
python scripts/predict_today.py            # live call for the current entry point (or --at HH:MM)

# Track A — one-shot historical backfill (run whenever; re-runnable)
python scripts/run_backfill.py --all

# Track B (laptop) — the looping live collector, full feed incl. option chain
python scripts/run_collector.py

# Track B (laptop) — one-off catch-up of NSE-direct data (option chain / PCR / VIX)
python scripts/run_backfill.py --option-chain
```

### Track B in the cloud (primary — runs even when the laptop is off)

`.github/workflows/collect.yml` polls `^NSEI` + `^NSEBANK` every ~5 min during NSE
market hours on GitHub Actions and commits one parquet/day under `collected/`.
NSE-direct endpoints (option chain, PCR/OI, FII-DII) block datacenter IPs, so the
cloud job is **quotes only** — those features are a laptop catch-up job (above).

Setup (one time):
1. Push this repo to a **public** GitHub repo.
2. Actions → enable workflows. Settings → Actions → General → Workflow permissions
   → "Read and write permissions".
3. It starts on the next 5-min slot; trigger once manually from the Actions tab.

Pull the accumulated data to the laptop: `git pull`. Merge `collected/` into the
main store when building datasets (Phase 2+).

> Scheduled Actions are best-effort — a slot can lag 5–15 min or occasionally be
> skipped. Fine for 5-min bars; if you later want the true 90-second feed, run
> `scripts/run_collector.py` on an always-on box (home Pi / Oracle free VM)
> alongside the Actions job — both append to the same store.

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
