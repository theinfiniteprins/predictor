"""Phase 1 — data pipeline.

Track A (re-runnable historical backfill):
    backfill_yfinance   ^NSEI intraday + daily, Bank Nifty, global cues
    backfill_bhavcopy    NSE daily EOD, long history (context features)
    option_chain         NSE index option chain -> PCR / OI skew, India VIX

Track B (start immediately, append-only):
    live_collector       poll delayed quotes every ~90s during market hours
"""
