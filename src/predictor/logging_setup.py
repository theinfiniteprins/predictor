"""Shared logging config: console + a rotating file under paths.logs_dir."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from .config import CONFIG

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        CONFIG.paths.logs_dir.mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        root = logging.getLogger("predictor")
        root.setLevel(logging.INFO)

        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)

        fileh = RotatingFileHandler(
            CONFIG.paths.logs_dir / "predictor.log",
            maxBytes=5_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        fileh.setFormatter(fmt)
        root.addHandler(fileh)

        _CONFIGURED = True

    return logging.getLogger(f"predictor.{name}")
