# Data Acquisition and Processing Modules

This package provides data extraction, ingestion, and consolidation pipelines
for the predictor system. It supports Track A historical backfilling via Yahoo Finance
and NSE archives, alongside Track B live intraday quote capture and consolidation into
unified parquet datasets.

## Direct Children

- `__init__.py`: Package exports for data acquisition, loading, and consolidation functions.
- `backfill_bhavcopy.py`: Downloads historical NSE Bhavcopy archives for daily cash market and index records.
- `backfill_yfinance.py`: Pulls multi-year daily bars, rolling intraday bars, and global macro cues from Yahoo Finance.
- `consolidate.py`: Merges raw historical pulls with append-only collector quotes into continuous unified interim parquet tables.
- `live_collector.py`: Implements Track B live quote collection, completed 1-minute bar extraction, and option chain snapshotting.
- `load.py`: Data access helpers that load raw bars, collected quote slices, and unified intraday series into pandas DataFrames.
- `option_chain.py`: Scrapes and parses live NSE option chain metrics including Put-Call Ratio and open interest distribution.
