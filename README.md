# Intraday Triple-Barrier Direction Predictor (NSE stocks & indices)

Personal-use tool that predicts, at rolling intraday entry points, whether an NSE stock
or index will hit an **upper** target, a **lower** target, or **neither** (timeout)
before 15:20 IST — and only surfaces a call when a meta-labeling confidence filter says
the primary model's prediction is trustworthy. **No call is always fine; a wrong call
is not** — nothing in this tool relaxes that to force a signal out of thin data.

Started as a Nifty 50-only tool; now supports any number of NSE-listed instruments
side by side, each with its own fully independent data/model/reports (see
[Multi-instrument](#multi-instrument) below). "Nifty 50" in the rest of this doc is the
running example, not a limitation.

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
| 5b | Evidence gate — refuse to fire without a proven edge | ✅ built |
| 6 | Backtest, threshold sweep | ✅ built |
| 7 | Paper-trade 4–8 weeks (`papertrade.py`, `scripts/paper_log.py`) | ✅ harness built, accumulating |
| 8 | (optional) sequence models / ensembling on larger dataset | not started |

> **Data depth, not code, is the blocker now.** The full pipeline runs end-to-end,
> but with only ~70 days of history the models cannot yet learn a real edge — the
> primary mostly predicts "timeout". This is expected (see §11 of the plan). Keep the
> cloud collector running; revisit training in 2–3 months.

## How we know whether it works yet

The honest answer today is **it doesn't, and the system now says so itself.**

Measured on the current history, no configuration beats a no-skill baseline. Run
`python scripts/research.py` to reproduce that verdict at any time — it re-runs the
comparison as data accumulates, so you find out the moment something genuinely changes.

Three measurement traps this project explicitly guards against:

- **Overlapping labels.** Rolling entries every 15 min nearly all resolve at the same
  15:20 vertical barrier, and 23 of the 53 features are constant within a day. ~1,500
  rows carry only **~115 independent observations**. Row-wise (Wilson/binomial)
  intervals are ~2–3× too narrow here, so every headline interval is a **day-block
  bootstrap** instead (`validation/significance.py`). Training also weights rows by
  [average uniqueness](src/predictor/labeling/uniqueness.py) so one session can't count
  as 21 pieces of evidence.
- **The wrong baseline.** Barrier touches are asymmetric — in a drifting market the
  lower barrier is hit far more often, so "always say down" scores well while
  predicting nothing. The bar to clear is that baseline, not a coin flip.
- **A cutoff that always fires.** `fire_top_fraction` is a *relative* quantile: the top
  15% fires no matter how bad the calls are. With `require_proven_edge: true` (default)
  a threshold is saved **only** if the out-of-fold record beats the naive baseline with
  the bootstrap lower bound above zero. Otherwise `fire_threshold` is `null` and the
  model stays silent. Silence is a correct output, not a failure — the dashboard's
  "Is it actually working?" card says which state you're in and why.
- **Testing the same idea every night.** The gate re-runs on every retrain, so at
  alpha=0.05 a false pass becomes near-certain over a year of daily runs. Three
  compounding guards: the threshold is chosen on older sessions and must replicate on a
  held-out `confirm_days` window; it must pass `min_consecutive_passes` retrains in a
  row; and it must clear the **break-even** precision implied by the barrier width and
  costs, because beating a baseline while still losing money is not an edge.

### What the research has ruled out so far

Reproduce any of this with `scripts/research.py`.

| Idea | Result |
|---|---|
| Barrier width k from 0.3 to 1.5 | No edge at any k. The break-even gap barely moves — P(touch) and break-even scale together |
| Denser target (sign of the 15:20 return, ~10x more directional rows) | No edge. 48.5% accuracy vs a 59.6% majority baseline |
| Pooling across 12 liquid NSE names (14,868 rows) | **No edge, and pooling did not help accuracy**: 47.9% pooled vs 48.7% solo on identical rows |
| Momentum / mean-reversion rules | No edge |
| Uniqueness weights, feature reduction, heavy regularisation | No edge |

Two numbers worth knowing:

- **The bar is low.** Break-even needs only **~0.6–1.1pp** of precision above the naive
  baseline (at k=0.6 and k=1.0 respectively). Timeouts resolve near flat, so they cost
  little — it is the opposite-barrier touches that pay for the wins.
- **But we cannot see that finely.** At the observed rate of ~0.83 directional calls per
  calendar day, detecting even a *5pp* edge takes on the order of 1,000 sessions. The
  measurement is far coarser than the edge that would pay for itself.

That gap is the real constraint, and it points at the one lever that actually moves:
evaluating across 12 instruments instead of one **halved** the confidence-interval width
on the same 44 days (0.219 → 0.112). Registering more instruments does not make the
model smarter — the pooling test says it does not — but it buys independent calls per
calendar day, which is how you find out sooner.

## Phase 0 parameters (finalized)

- **Instrument:** NIFTY 50 index, `^NSEI` (yfinance) — the default; any NSE stock can
  be registered alongside it, see [Multi-instrument](#multi-instrument)
- **Barrier multiplier k:** 1.0, symmetric (swept 0.5–2.0 later in Phase 6) — shared
  across every instrument via `config.yaml`
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

### The daily command — everything in one go

```powershell
D:\Predictor\run.ps1
```

Runs: `git pull` → refresh raw data → build dataset → paper-log yesterday's calls
→ retrain → backtest → print the report. ~1 min. Options: `-Instrument LAURUSLABS`
(which stock — default `NIFTY50`, see [Multi-instrument](#multi-instrument)),
`-Tune 200` (Optuna search), `-K 0.6` (barrier override), `-Holdout 14` (hold out the
tail as out-of-sample), `-SkipPull`, `-SkipBackfill`.

Unattended (runs next time the laptop is on after the trigger):

```powershell
schtasks /create /tn "PredictorDaily" /tr "powershell -ExecutionPolicy Bypass -File D:\Predictor\run.ps1" /sc daily /st 18:30 /f
```

For another registered stock, name the task `PredictorDaily-<KEY>` (e.g.
`PredictorDaily-LAURUSLABS`) and pass `-Instrument <KEY>` in `/tr` — the dashboard's
automation card looks for that naming convention. Registering a stock through the
dashboard does **not** create this task automatically; that stays a manual step.

### GUI dashboard (recommended if you just want to watch it work)

Double-click **`Start Dashboard.bat`** in the project root. It opens a control-panel
page in your browser (`http://127.0.0.1:8787`) with a stock picker, a big
**Run Predictor Now** button and a live console, plus plain-language cards for: how
much market data has been collected (and how fresh it is), the training dataset, the
model's current verdict, backtest results, paper-trading results, and whether the
daily automatic run is scheduled. No command line needed — this is the same `run.ps1`
flow, just with a UI. Pick **"+ Register a new stock…"** in the dropdown to add another
instrument by company name (see [Multi-instrument](#multi-instrument)). See
[dashboard/server.py](dashboard/server.py) (stdlib-only + the project's existing
pandas/yfinance deps, no extra installs) if you want to see how it talks to the
pipeline.

### Individual steps / ad hoc

```bash
python scripts/report.py --signals              # the dashboard: dataset / model / CV / backtest / paper log
python scripts/research.py                      # does anything beat a no-skill baseline? (re-run as data grows)
python scripts/research.py --k-sweep            # ...also re-label at several barrier widths (slow)
python scripts/build_dataset.py --k 0.6         # rebuild with a different barrier multiplier
python scripts/train.py --tune 200              # walk-forward primary + meta, Optuna search
python scripts/predict_today.py --at 11:15      # what the model says for one entry point
python scripts/consolidate.py                   # merge collector + yfinance -> data/<inst>/interim/*_unified.parquet
python scripts/run_collector.py                 # local looping collector (single instrument only)
python scripts/collect_all.py --force           # one cloud-style poll of every registered instrument

# any script also takes --instrument KEY (default NIFTY50):
python scripts/report.py --instrument LAURUSLABS --signals
```

### Track B in the cloud (primary — runs even when the laptop is off)

`.github/workflows/collect.yml` runs every ~15 min during NSE market hours on GitHub
Actions. Each run reads `instruments.yaml`, fetches **all** of the day's completed
1-min bars for every registered instrument's own ticker + benchmark ticker
(deduplicated — a shared benchmark like Nifty 50 is only fetched once no matter how
many stocks use it), and makes **one commit** under
`collected/date=YYYY-MM-DD/quotes.parquet` (rows disambiguated by a `ticker` column,
de-duped on `(bar_time, ticker)`, so a missed slot is backfilled by the next run).
NSE-direct endpoints (option chain, PCR/OI, FII-DII) block datacenter IPs, so the
cloud job is **bars only** — those are a laptop catch-up job (`run_backfill.py`).
Because it's one job making one commit regardless of how many stocks are registered,
registering a new stock (which only ever edits `instruments.yaml`) can never race
with the collector's own commits.

`scripts/consolidate.py` (also called by `build_dataset.py`) merges `collected/`
with the rolling yfinance pull into `data/<instrument>/interim/<inst>_{1m,5m}_unified.parquet`,
which is what the model reads. This is how the training history grows past
yfinance's ~60-day window — days that age out of yfinance are retained from the
previously-built unified store.

> Scheduled Actions are best-effort — a slot can lag 5–15 min or occasionally be
> skipped. Fine for 5-min bars; if you later want the true 90-second feed, run
> `scripts/run_collector.py` on an always-on box (home Pi / Oracle free VM)
> alongside the Actions job — both append to the same store (single-instrument only
> today; the cloud job is the multi-stock path).

## Multi-instrument

`instruments.yaml` (repo root, git-tracked) is the single source of truth for what's
tracked — both the cloud collector and the dashboard read it. Each entry:

```yaml
- key: LAURUSLABS            # machine id -> data/models/reports/logs/<key>/
  display_name: "Laurus Labs Limited"
  yf_ticker: "LAURUSLABS.NS"
  benchmark_key: NIFTY50     # another registered instrument, used for divergence features
  benchmark_ticker: "^NSEI"
  option_symbol: null        # NSE F&O symbol, or null if none listed
  added_at: "2026-09-13"
```

**Adding a stock** is easiest through the dashboard's "+ Register a new stock…" picker
entry: search by company name, confirm the match, and it validates the ticker has real
yfinance history, appends it to `instruments.yaml`, commits, pushes straight to `main`
(no PRs — see `dashboard/server.py:register_instrument`), and kicks off an initial
backfill + training run automatically. You can also hand-edit `instruments.yaml` and
commit/push it yourself.

Every script accepts `--instrument KEY` (default `NIFTY50`), or set the
`PREDICTOR_INSTRUMENT` env var directly — `config.py` resolves every path from it, so
each instrument's `data/`, `models/`, `reports/` and `logs/` are completely
independent; two stocks can even run at once (`run.ps1 -Instrument X` in one terminal,
`-Instrument Y` in another, or two runs triggered from the dashboard) without touching
each other's files. Only `data/_shared/` (the NSE holiday calendar, the cached NSE
symbol list) and `collected/` (disambiguated by ticker) are shared across instruments.

A freshly registered stock starts with only what `yfinance`'s ~60-day window gives it
— expect "still learning, not confident enough yet" from the dashboard for a while.
That's the intended behavior, not a bug: the meta-labeling filter simply won't fire
until it has enough history to be trustworthy for *that* stock.

## Layout

```
src/predictor/      all logic (importable, tested)
  config.py         loads config.yaml + instruments.yaml -> frozen dataclasses
  cli.py            --instrument / PREDICTOR_INSTRUMENT resolution for scripts
  calendar.py       NSE trading days / session bounds (shared across instruments)
  storage.py        parquet read/write, append-safe
  data/             Phase 1  — backfill + live collector
  labeling/         Phase 3  — σ_intraday, entry grid, triple-barrier
  features/         Phase 2  — technical / context / global / time-of-day
  validation/       Phase 4  — purged+embargoed walk-forward CV, metrics
  models/           Phase 4-5 — primary, meta, tuning
  backtest/         Phase 6  — true triple-barrier exits, costs
scripts/            thin CLI entrypoints (collect_all.py is the multi-stock cloud one)
dashboard/          stdlib-only local GUI (Start Dashboard.bat) - stock picker + registration
instruments.yaml    registry of tracked instruments (git-tracked)
data/<key>/         local data lake, per instrument (gitignored)
data/_shared/       NSE holiday calendar + symbol-search cache (gitignored)
tests/              triple-barrier, purged-CV, no-look-ahead
```

## Not investment advice

Personal decision-support only. No automated order placement. See §12 of the plan.
