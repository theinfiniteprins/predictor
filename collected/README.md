# Collected Market Quotes Data Store

This directory contains date-partitioned raw 1-minute intraday quote data captured
by the Track B cloud collector (GitHub Actions) and local backfills.
Each partition contains completed 1-minute bars for registered NSE instruments and benchmarks.
Data stored here is committed to git to create an append-only historical store that never expires.

## Direct Children

- `README.md`: Overview documentation for the collected quotes storage directory.
- `date=YYYY-MM-DD/`: Daily partition folders each containing `quotes.parquet` with 1-minute bars across all registered instruments for that trading date.
