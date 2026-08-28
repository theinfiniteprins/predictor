"""Load config.yaml into frozen dataclasses. Import `CONFIG` everywhere else.

Paths in config.yaml are resolved relative to the project root (the dir holding
config.yaml), so scripts work regardless of the current working directory.
"""

from __future__ import annotations

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
    name: str
    yf_ticker: str
    option_symbol: str
    correlated_ticker: str


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
    data_dir: Path
    models_dir: Path
    reports_dir: Path
    logs_dir: Path

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

    def ensure(self) -> None:
        for p in (
            self.data_dir, self.raw, self.interim, self.processed, self.raw_live,
            self.raw / "yfinance", self.raw / "bhavcopy", self.raw / "option_chain",
            self.models_dir, self.reports_dir, self.logs_dir,
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


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> Config:
    root = _project_root()
    cfg_path = Path(path) if path else root / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    p = raw["paths"]
    paths = Paths(
        root=root,
        data_dir=root / p["data_dir"],
        models_dir=root / p["models_dir"],
        reports_dir=root / p["reports_dir"],
        logs_dir=root / p["logs_dir"],
    )

    s = raw["session"]
    lab = raw["labeling"]
    d = raw["data"]
    gc = raw["global_cues"]
    col = raw["collector"]
    cv = raw["cv"]

    return Config(
        instrument=InstrumentConfig(**raw["instrument"]),
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
