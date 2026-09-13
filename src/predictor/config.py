"""Load config.yaml into frozen dataclasses. Import `CONFIG` everywhere else.

Paths in config.yaml are resolved relative to the project root (the dir holding
config.yaml), so scripts work regardless of the current working directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import time
from functools import lru_cache
from pathlib import Path

import yaml


def _project_root() -> Path:
    # src/predictor/config.py  ->  project root is two parents up from this file's dir
    return Path(__file__).resolve().parents[2]


def _parse_hhmm(value: str) -> time:
    hh, mm = value.split(":")
    return time(int(hh), int(mm))


@dataclass(frozen=True)
class InstrumentConfig:
    name: str                    # machine key, e.g. "NIFTY50" / "LAURUSLABS" (used for file naming)
    display_name: str
    yf_ticker: str
    option_symbol: str | None
    benchmark_key: str           # e.g. "BANKNIFTY" for NIFTY50, "NIFTY50" for any equity
    benchmark_ticker: str


@dataclass(frozen=True)
class SessionConfig:
    timezone: str
    open: time
    close: time


@dataclass(frozen=True)
class LabelingConfig:
    k: float
    atr_window: int
    vertical_barrier: time
    entry_start: time
    entry_end: time
    entry_freq_minutes: int
    intrabar_tiebreak: str
    barriers_symmetric: bool
    barrier_time_scaling: bool
    session_minutes: int


@dataclass(frozen=True)
class DataConfig:
    bar_interval: str
    fine_interval: str
    backfill_period: str
    daily_period: str


@dataclass(frozen=True)
class GlobalCuesConfig:
    sp500: str
    nasdaq: str
    usdinr: str
    crude: str
    india_vix: str
    gift_nifty: str

    def active(self) -> dict[str, str]:
        """Return only the non-empty {name: ticker} cues."""
        return {k: v for k, v in vars(self).items() if v}


@dataclass(frozen=True)
class CollectorConfig:
    poll_seconds: int
    source: str


@dataclass(frozen=True)
class CVConfig:
    embargo_minutes: int
    min_train_days: int
    test_window_days: int
    step_days: int


@dataclass(frozen=True)
class MetaConfig:
    fire_top_fraction: float


@dataclass(frozen=True)
class Paths:
    root: Path
    data_dir: Path          # root/data/<instrument key>  - everything below is per-instrument
    models_dir: Path        # root/models/<instrument key>
    reports_dir: Path       # root/reports/<instrument key>
    logs_dir: Path          # root/logs/<instrument key>
    shared_dir: Path        # root/data/_shared  - NOT namespaced by instrument

    # convenient sub-locations of the data lake
    @property
    def raw(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed(self) -> Path:
        return self.data_dir / "processed"

    @property
    def raw_live(self) -> Path:
        return self.raw / "live"

    @property
    def reference(self) -> Path:
        """Shared across every instrument (NSE holiday calendar, NSE symbol cache)."""
        return self.shared_dir / "reference"

    def ensure(self) -> None:
        for p in (
            self.data_dir, self.raw, self.interim, self.processed, self.raw_live,
            self.raw / "yfinance", self.raw / "bhavcopy", self.raw / "option_chain",
            self.models_dir, self.reports_dir, self.logs_dir, self.reference,
        ):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    instrument: InstrumentConfig
    session: SessionConfig
    labeling: LabelingConfig
    data: DataConfig
    global_cues: GlobalCuesConfig
    collector: CollectorConfig
    cv: CVConfig
    meta: MetaConfig
    paths: Paths
    raw: dict = field(repr=False, default_factory=dict)


def load_instruments_registry(root: Path | None = None) -> list[dict]:
    """All registered instruments from instruments.yaml (git-tracked at repo root)."""
    root = root or _project_root()
    reg_path = root / "instruments.yaml"
    if not reg_path.exists():
        raise FileNotFoundError(
            f"{reg_path} missing - every instrument (including NIFTY50) is defined there"
        )
    with open(reg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return raw.get("instruments", [])


def _resolve_instrument(root: Path, key: str) -> dict:
    for entry in load_instruments_registry(root):
        if entry["key"] == key:
            return entry
    known = ", ".join(e["key"] for e in load_instruments_registry(root))
    raise KeyError(f"instrument '{key}' not found in instruments.yaml (known: {known})")


def load_config(path: str | Path | None = None, instrument: str | None = None) -> Config:
    """Resolve the active instrument (arg > PREDICTOR_INSTRUMENT env var > "NIFTY50")
    fresh on every call, then delegate to the cached builder. Re-reading the env var
    here (rather than caching on the sentinel ``None``) matters for long-running
    processes like the dashboard server that resolve many different instruments'
    configs within one process.
    """
    inst_key = instrument or os.environ.get("PREDICTOR_INSTRUMENT", "NIFTY50")
    return _build_config(path, inst_key)


@lru_cache(maxsize=None)
def _build_config(path: str | Path | None, inst_key: str) -> Config:
    root = _project_root()
    cfg_path = Path(path) if path else root / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    inst_entry = _resolve_instrument(root, inst_key)

    p = raw["paths"]
    paths = Paths(
        root=root,
        data_dir=root / p["data_dir"] / inst_key,
        models_dir=root / p["models_dir"] / inst_key,
        reports_dir=root / p["reports_dir"] / inst_key,
        logs_dir=root / p["logs_dir"] / inst_key,
        shared_dir=root / p["data_dir"] / "_shared",
    )

    s = raw["session"]
    lab = raw["labeling"]
    d = raw["data"]
    gc = raw["global_cues"]
    col = raw["collector"]
    cv = raw["cv"]

    return Config(
        instrument=InstrumentConfig(
            name=inst_entry["key"],
            display_name=inst_entry.get("display_name", inst_entry["key"]),
            yf_ticker=inst_entry["yf_ticker"],
            option_symbol=inst_entry.get("option_symbol") or None,
            benchmark_key=inst_entry["benchmark_key"],
            benchmark_ticker=inst_entry["benchmark_ticker"],
        ),
        session=SessionConfig(
            timezone=s["timezone"],
            open=_parse_hhmm(s["open"]),
            close=_parse_hhmm(s["close"]),
        ),
        labeling=LabelingConfig(
            k=float(lab["k"]),
            atr_window=int(lab["atr_window"]),
            vertical_barrier=_parse_hhmm(lab["vertical_barrier"]),
            entry_start=_parse_hhmm(lab["entry_start"]),
            entry_end=_parse_hhmm(lab["entry_end"]),
            entry_freq_minutes=int(lab["entry_freq_minutes"]),
            intrabar_tiebreak=lab["intrabar_tiebreak"],
            barriers_symmetric=bool(lab["barriers_symmetric"]),
            barrier_time_scaling=bool(lab.get("barrier_time_scaling", False)),
            session_minutes=int(lab.get("session_minutes", 375)),
        ),
        data=DataConfig(
            bar_interval=d["bar_interval"],
            fine_interval=d["fine_interval"],
            backfill_period=d["backfill_period"],
            daily_period=d["daily_period"],
        ),
        global_cues=GlobalCuesConfig(
            sp500=gc.get("sp500", ""),
            nasdaq=gc.get("nasdaq", ""),
            usdinr=gc.get("usdinr", ""),
            crude=gc.get("crude", ""),
            india_vix=gc.get("india_vix", ""),
            gift_nifty=gc.get("gift_nifty", ""),
        ),
        collector=CollectorConfig(
            poll_seconds=int(col["poll_seconds"]),
            source=col["source"],
        ),
        cv=CVConfig(
            embargo_minutes=int(cv["embargo_minutes"]),
            min_train_days=int(cv["min_train_days"]),
            test_window_days=int(cv["test_window_days"]),
            step_days=int(cv["step_days"]),
        ),
        meta=MetaConfig(fire_top_fraction=float(raw["meta"]["fire_top_fraction"])),
        paths=paths,
        raw=raw,
    )


CONFIG = load_config()
