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
| 7 | Paper-trade 4–8 weeks (`papertrade.py`, `scripts/paper_log.py`) | ✅ harness built, accumulating |
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

### Whenever the laptop is on (keeps the record growing)

```bash
git pull                                   # fresh collector data from GitHub Actions
python scripts/run_backfill.py --all        # refresh the yfinance window + cues + option chain
python scripts/paper_log.py                 # log newly-resolved entries + print live-vs-backtest summary
```

### Retrain (monthly, or whenever enough new data has accumulated)

```bash
python scripts/build_dataset.py            # consolidates collector+yfinance, then features + labels
python scripts/train.py --tune 200         # walk-forward primary + meta, Optuna search, save bundle
python scripts/backtest.py                 # threshold sweep on the OOF predictions
```

### Ad hoc

```bash
python scripts/consolidate.py              # merge collector + yfinance into data/interim/*_unified.parquet
python scripts/predict_today.py --at 11:15 # what the model says for one entry point
python scripts/run_collector.py            # local looping collector (for when the laptop IS on in-hours)
```

### Track B in the cloud (primary — runs even when the laptop is off)

`.github/workflows/collect.yml` runs every ~5 min during NSE market hours on GitHub
Actions. Each run fetches **all** of the day's completed 1-min bars for `^NSEI` +
`^NSEBANK` and commits them under `collected/date=YYYY-MM-DD/quotes.parquet`
(de-duped on `(bar_time, ticker)`, so a missed slot is backfilled by the next run).
NSE-direct endpoints (option chain, PCR/OI, FII-DII) block datacenter IPs, so the
cloud job is **bars only** — those are a laptop catch-up job (`run_backfill.py`).

`scripts/consolidate.py` (also called by `build_dataset.py`) merges `collected/`
with the rolling yfinance pull into `data/interim/<inst>_{1m,5m}_unified.parquet`,
which is what the model reads. This is how the training history grows past
yfinance's ~60-day window — days that age out of yfinance are retained from the
previously-built unified store.

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
