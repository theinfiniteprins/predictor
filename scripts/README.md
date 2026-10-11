# Predictor Command-Line Scripts

This directory contains standalone automation scripts, entrypoints, and CLI utilities
for data collection, historical backfilling, model training, paper trade logging,
and performance reporting. These scripts can be run individually or invoked sequentially
by the central daily runner script `run.ps1` and GitHub Actions workflows.

## Direct Children

- `backtest.py`: Evaluates primary and meta-labeling prediction signals over walk-forward folds and saves backtest statistics.
- `build_dataset.py`: Coordinates consolidation of raw/collected bars, feature computation, and triple-barrier labeling into processed datasets.
- `collect_all.py`: Multi-instrument cloud quote poller that pulls 1-minute bars during and post session for all registered instruments in `instruments.yaml`.
- `collect_tick.py`: Single-instrument one-shot quote poller that appends 1-minute bars for the active instrument.
- `consolidate.py`: Merges raw historical pulls and append-only collector quotes into continuous unified interim parquet datasets.
- `paper_log.py`: Logs real-world forward paper-trading evaluations against model predictions prior to retraining.
- `predict_today.py`: Generates live intraday directional predictions at specific entry timestamps for the active instrument.
- `refresh_calendar.py`: Downloads and caches the official NSE equity trading holiday calendar to avoid hardcoded fallback drift.
- `report.py`: CLI summary reporting utility that prints dataset volume, model performance cards, and recent prediction records.
- `research.py`: Rigorous statistical analysis tool testing feature efficacy, barrier sweeps, and baseline superiority.
- `run_backfill.py`: One-shot historical backfiller fetching daily/intraday bars, market cues, and bhavcopy records from Yahoo Finance.
- `run_collector.py`: Continuous local looping poller that captures 1-minute quote ticks and option chain snapshots during market hours.
- `train.py`: Trains primary classifiers and meta-labeling confidence models using purged and embargoed walk-forward cross-validation.
