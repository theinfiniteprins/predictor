"""Local control-panel GUI for the predictor - now multi-instrument. Stdlib only,
plus pandas/yfinance which are already project dependencies.

Run it with the project's own virtualenv (it needs the installed `predictor`
package, exactly like the other scripts/*.py entrypoints):

    D:\\Predictor\\.venv\\Scripts\\python.exe dashboard\\server.py

or just double-click `Start Dashboard.bat` in the project root.

What it does:
  - Serves a one-page dashboard with a stock picker. Every registered instrument
    (see instruments.yaml) gets its own fully independent view: data collected, model,
    backtest, paper trading, run button/console - all scoped by an `?instrument=KEY`
    query param / request body field.
  - GET  /api/instruments         -> the registry, for the picker.
  - GET  /api/instruments/search  -> fuzzy NSE symbol search, for "register a stock".
  - POST /api/instruments         -> register: validate ticker -> append to
                                      instruments.yaml -> commit -> (by default) push
                                      -> kick off an initial backfill run.
  - GET  /api/status?instrument=KEY   -> one big JSON snapshot for that instrument.
  - POST /api/run                     -> starts run.ps1 -Instrument KEY in the background.
  - GET  /api/run/output?instrument=KEY&since=N -> incremental console output.
  - POST /api/run/stop?instrument=KEY -> kills that instrument's in-progress run.

Each instrument has its own run-tracking state (see `_RUNS`), so two different
stocks really can run at the same time; the same stock twice cannot.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

import nse_symbols  # noqa: E402
from predictor.config import CONFIG, load_config, load_instruments_registry  # noqa: E402
from predictor.calendar import market_is_open, now_ist  # noqa: E402

PORT = 8787
STATIC_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# run.ps1 process management - one independent state machine per instrument
# ---------------------------------------------------------------------------

_RUN_LOCK = threading.Lock()
_RUNS: dict[str, dict] = {}

STAGES = [
    ("git pull (collector data)", "Downloading data collected while your computer was off"),
    ("refresh raw data", "Fetching the latest market data"),
    ("build dataset (consolidate + label + features)", "Organizing the data and calculating patterns"),
    ("paper-trade log", "Checking how yesterday's predictions turned out"),
    ("retrain (walk-forward primary + meta)", "Teaching the model with the newest data"),
    ("backtest", "Testing how the model would have performed"),
]


def _run_state(key: str) -> dict:
    return _RUNS.setdefault(key, {
        "running": False, "run_id": None, "proc": None, "lines": [],
        "exit_code": None, "started_at": None, "finished_at": None, "options": None,
    })


def _run_worker(key: str, args: list[str]) -> None:
    state = _run_state(key)
    try:
        proc = subprocess.Popen(
            args, cwd=str(CONFIG.paths.root), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        state["proc"] = proc
        for line in proc.stdout:  # type: ignore[union-attr]
            state["lines"].append(line.rstrip("\n"))
        proc.wait()
        state["exit_code"] = proc.returncode
    except Exception as exc:  # pragma: no cover - defensive
        state["lines"].append(f"[dashboard] failed to launch run.ps1: {exc}")
        state["exit_code"] = -1
    finally:
        state["running"] = False
        state["finished_at"] = datetime.now().isoformat()


def start_run(instrument: str, options: dict) -> tuple[str | None, str | None]:
    with _RUN_LOCK:
        state = _run_state(instrument)
        if state["running"]:
            return None, f"A run for {instrument} is already in progress."
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        state.update(running=True, run_id=run_id, proc=None, lines=[], exit_code=None,
                      started_at=datetime.now().isoformat(), finished_at=None, options=options)

    args = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(CONFIG.paths.root / "run.ps1"), "-Instrument", instrument]
    if options.get("skipPull"):
        args.append("-SkipPull")
    if options.get("skipBackfill"):
        args.append("-SkipBackfill")
    if options.get("tune"):
        args += ["-Tune", str(int(options["tune"]))]
    if options.get("k"):
        args += ["-K", str(float(options["k"]))]
    if options.get("holdout"):
        args += ["-Holdout", str(int(options["holdout"]))]

    threading.Thread(target=_run_worker, args=(instrument, args), daemon=True).start()
    return run_id, None


def stop_run(instrument: str) -> bool:
    state = _RUNS.get(instrument)
    if state and state.get("proc") is not None and state["running"]:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(state["proc"].pid)],
                        capture_output=True, text=True)
        return True
    return False


def run_snapshot(instrument: str, since: int) -> dict:
    state = _run_state(instrument)
    lines = state["lines"]
    completed_stages = [label for label, _ in STAGES
                        if any(f"========== {label} ==========" in ln for ln in lines)]
    current_stage = (completed_stages[-1] if completed_stages else "starting") if state["running"] else None

    return {
        "instrument": instrument,
        "running": state["running"],
        "run_id": state["run_id"],
        "started_at": state["started_at"],
        "finished_at": state["finished_at"],
        "exit_code": state["exit_code"],
        "lines": lines[since:],
        "total_lines": len(lines),
        "current_stage": current_stage,
        "completed_stages": completed_stages,
        "stages": [{"label": lbl, "plain": plain} for lbl, plain in STAGES],
    }


# ---------------------------------------------------------------------------
# status snapshot (the read-only dashboard data) - all scoped to one instrument's
# resolved Config, fetched fresh via load_config(instrument=key) rather than the
# frozen module-level CONFIG (which is pinned to whichever instrument this process
# happened to import first).
# ---------------------------------------------------------------------------

def _safe(fn, default=None):
    try:
        return fn()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[dashboard] {fn!r} failed: {exc}", file=sys.stderr)
        return {"error": str(exc)} if default is None else default


def _load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _mtime_iso(path: Path) -> str | None:
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat() if path.exists() else None


def _scan_collected(root: Path, ticker: str) -> dict:
    coll_root = root / "collected"
    per_day = []
    total_rows = 0
    last_ingested = None
    if coll_root.exists():
        for d in sorted(coll_root.glob("date=*")):
            pq = d / "quotes.parquet"
            if not pq.exists():
                continue
            df = pd.read_parquet(pq)
            if "ticker" in df.columns:
                df = df[df["ticker"] == ticker]
            rows = len(df)
            if not rows:
                continue
            total_rows += rows
            per_day.append({"date": d.name.split("=", 1)[1], "rows": rows})
            ing_col = "ingested_at" if "ingested_at" in df.columns else (
                "timestamp" if "timestamp" in df.columns else None)
            if ing_col:
                ing = df[ing_col].max()
                if last_ingested is None or ing > last_ingested:
                    last_ingested = ing
    return {
        "ticker": ticker,
        "days": len(per_day),
        "total_rows": total_rows,
        "per_day": per_day[-21:],
        "first_date": per_day[0]["date"] if per_day else None,
        "last_date": per_day[-1]["date"] if per_day else None,
        "last_ingested_at": last_ingested.isoformat() if last_ingested is not None else None,
    }


def _dataset_info(cfg) -> dict | None:
    return _load_json(cfg.paths.processed / "dataset_meta.json")


def _model_info(cfg) -> dict | None:
    card = _load_json(cfg.paths.models_dir / "model_card.json")
    if not card:
        return None
    hc = card.get("metrics", {}).get("high_confidence", {})
    n_fired = hc.get("n_fired") or 0
    precision = hc.get("precision")
    if card.get("fire_threshold") is None:
        verdict = "Not enough history yet to trust a confidence filter - it will not make live calls until it has more data. That's expected for a newly added stock."
    elif not n_fired:
        verdict = "Still learning - it isn't confident enough yet to make a call. It needs more trading days of data."
    elif precision is not None and precision >= 0.55:
        verdict = f"Making confident calls, right ~{precision:.0%} of the time so far."
    else:
        verdict = f"Making a few confident calls, but only right ~{precision:.0%} of the time so far - treat with caution."
    card["plain_verdict"] = verdict
    card["mtime"] = _mtime_iso(cfg.paths.models_dir / "model_card.json")
    return card


def _backtest_info(cfg) -> dict | None:
    bt = _load_json(cfg.paths.reports_dir / "backtest.json")
    if not bt:
        return None
    h = bt.get("headline", {})
    n_trades = h.get("n_trades") or 0
    win_rate = h.get("win_rate")
    if not n_trades:
        verdict = "No simulated trades yet."
    elif win_rate is not None and win_rate >= 0.5:
        verdict = f"Would have won {win_rate:.0%} of {n_trades} simulated trades."
    else:
        verdict = f"Would have won only {win_rate:.0%} of {n_trades} simulated trades so far - not reliable yet."
    bt["plain_verdict"] = verdict
    bt["mtime"] = _mtime_iso(cfg.paths.reports_dir / "backtest.json")
    return bt


def _paper_trading_info(cfg) -> dict | None:
    """Mirrors predictor.papertrade.summarize(), but against a per-instrument `cfg`
    instead of that module's frozen module-level CONFIG (which is pinned to one
    instrument and would give the wrong answer for every other one)."""
    log_path = cfg.paths.reports_dir / "paper_trades.parquet"
    if not log_path.exists():
        return None
    log_df = pd.read_parquet(log_path)
    if log_df.empty:
        return {"status": "no paper trades logged yet"}

    oos = log_df[~log_df["in_sample"].fillna(False)]
    fired = oos[oos["fired"].fillna(False)]
    n_days = int(oos["day"].nunique()) if len(oos) else 0

    def _dir_prec(df: pd.DataFrame) -> dict:
        d = df[df["primary_pred"] != 0]
        return {"n": len(d), "precision": float(d["correct"].mean()) if len(d) else None}

    out = {
        "rows_total": len(log_df),
        "rows_out_of_sample": len(oos),
        "oos_date_range": [str(oos["day"].min()), str(oos["day"].max())] if len(oos) else None,
        "oos_directional_precision": _dir_prec(oos),
        "fired": {
            "n": len(fired),
            "precision": float(fired["correct"].mean()) if len(fired) else None,
            "per_day": len(fired) / n_days if n_days else 0.0,
            "hit_up": int(((fired["primary_pred"] == 1) & fired["correct"]).sum()),
            "hit_down": int(((fired["primary_pred"] == -1) & fired["correct"]).sum()),
            "avg_ret_dir_adj": float((fired["primary_pred"] * fired["ret_at_touch"]).mean())
            if len(fired) else None,
        },
    }
    bt_path = cfg.paths.reports_dir / "backtest.json"
    if bt_path.exists():
        h = json.loads(bt_path.read_text(encoding="utf-8")).get("headline", {})
        out["backtest_reference"] = {
            "directional_precision": h.get("directional_precision"),
            "trades_per_day": h.get("trades_per_day"),
        }
    return out


def _scheduled_task_info(instrument_key: str) -> dict:
    candidates = [f"PredictorDaily-{instrument_key}"]
    if instrument_key == "NIFTY50":
        candidates.append("PredictorDaily")
    for name in candidates:
        try:
            out = subprocess.run(
                ["schtasks", "/query", "/tn", name, "/fo", "LIST", "/v"],
                capture_output=True, text=True, timeout=10,
            )
        except Exception:
            continue
        if out.returncode != 0:
            continue
        info = {}
        for line in out.stdout.splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                info[k.strip()] = v.strip()
        return {
            "scheduled": True,
            "task_name": name,
            "next_run": info.get("Next Run Time"),
            "last_run": info.get("Last Run Time"),
            "last_result": info.get("Last Result"),
            "status": info.get("Scheduled Task State"),
        }
    return {"scheduled": False}


def _last_collector_commit(root: Path) -> dict | None:
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%cI|%s", "--", "collected"],
            cwd=str(root), capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            iso, _, msg = out.stdout.strip().partition("|")
            return {"time": iso, "message": msg}
    except Exception:
        pass
    return None


def _log_tail(cfg, n: int = 60) -> list[str]:
    path = cfg.paths.logs_dir / "predictor.log"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.readlines()
    return [ln.rstrip("\n") for ln in lines[-n:]]


def build_status(instrument: str) -> dict:
    cfg = load_config(instrument=instrument)
    return {
        "instrument": {"key": cfg.instrument.name, "display_name": cfg.instrument.display_name,
                       "benchmark_key": cfg.instrument.benchmark_key},
        "server_time": datetime.now().isoformat(),
        "market_open": _safe(lambda: bool(market_is_open()), False),
        "now_ist": _safe(lambda: now_ist().isoformat(), None),
        "collector": _safe(lambda: _scan_collected(cfg.paths.root, cfg.instrument.yf_ticker), {}),
        "dataset": _safe(lambda: _dataset_info(cfg), None),
        "model": _safe(lambda: _model_info(cfg), None),
        "backtest": _safe(lambda: _backtest_info(cfg), None),
        "paper_trading": _safe(lambda: _paper_trading_info(cfg), None),
        "scheduled_task": _safe(lambda: _scheduled_task_info(cfg.instrument.name), {"scheduled": False}),
        "last_collector_commit": _safe(lambda: _last_collector_commit(cfg.paths.root), None),
        "log_tail": _safe(lambda: _log_tail(cfg, 60), []),
        "run": run_snapshot(instrument, 0),
    }


# ---------------------------------------------------------------------------
# instrument registry: list / search / register
# ---------------------------------------------------------------------------

def _sanitize_key(raw: str) -> str:
    key = re.sub(r"[^A-Za-z0-9]", "", raw or "").upper()
    if not key:
        raise ValueError("that doesn't look like a valid NSE symbol")
    return key


def _append_instrument_entry(root: Path, entry: dict) -> None:
    path = root / "instruments.yaml"
    block = yaml.safe_dump(entry, default_flow_style=False, sort_keys=False)
    lines = block.strip("\n").split("\n")
    indented = ["  - " + lines[0]] + [f"    {ln}" for ln in lines[1:]]
    text = path.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text + "\n".join(indented) + "\n", encoding="utf-8")


def register_instrument(payload: dict) -> dict:
    symbol_raw = (payload.get("symbol") or "").strip()
    if not symbol_raw:
        raise ValueError("symbol is required")
    key = _sanitize_key(symbol_raw)
    display_name = (payload.get("name") or key).strip()
    benchmark_key = (payload.get("benchmark_key") or "NIFTY50").strip().upper()
    push = bool(payload.get("push", True))
    root = CONFIG.paths.root

    existing = load_instruments_registry(root)
    if any(e["key"] == key for e in existing):
        raise ValueError(f"{key} is already registered")
    benchmark_entry = next((e for e in existing if e["key"] == benchmark_key), None)
    if benchmark_entry is None:
        raise ValueError(f"benchmark instrument '{benchmark_key}' is not registered")

    yf_ticker = f"{key}.NS"
    import yfinance as yf
    hist = yf.Ticker(yf_ticker).history(period="5d")
    if hist is None or hist.empty:
        raise ValueError(f"no market data found for {yf_ticker} - refusing to register a dead ticker")

    entry = {
        "key": key,
        "display_name": display_name,
        "yf_ticker": yf_ticker,
        "benchmark_key": benchmark_key,
        "benchmark_ticker": benchmark_entry["yf_ticker"],
        "option_symbol": None,
        "added_at": datetime.now().strftime("%Y-%m-%d"),
    }
    _append_instrument_entry(root, entry)

    log_lines = [f"validated {yf_ticker} ({len(hist)} recent bars found)",
                 f"appended to instruments.yaml: {key} ({display_name}), benchmark={benchmark_key}"]
    try:
        subprocess.run(["git", "add", "instruments.yaml"], cwd=root, check=True,
                        capture_output=True, text=True)
        commit = subprocess.run(
            ["git", "commit", "-m", f"register instrument: {key} ({display_name})"],
            cwd=root, capture_output=True, text=True,
        )
        log_lines.append(commit.stdout.strip() or commit.stderr.strip() or "commit created")
        if commit.returncode != 0:
            raise RuntimeError("git commit failed: " + (commit.stderr.strip() or commit.stdout.strip()))
        if push:
            pull = subprocess.run(["git", "pull", "--rebase", "--autostash"],
                                  cwd=root, capture_output=True, text=True)
            log_lines.append(pull.stdout.strip() or pull.stderr.strip() or "pull: up to date")
            pushed = subprocess.run(["git", "push", "origin", "main"],
                                    cwd=root, capture_output=True, text=True)
            log_lines.append(pushed.stdout.strip() or pushed.stderr.strip() or "pushed")
            if pushed.returncode != 0:
                raise RuntimeError("git push failed: " + pushed.stderr.strip())
        else:
            log_lines.append("push skipped (push=false)")
    except Exception as exc:
        log_lines.append(f"ERROR: {exc}")
        return {"entry": entry, "log": log_lines, "pushed": False, "error": str(exc)}

    return {"entry": entry, "log": log_lines, "pushed": push}


# ---------------------------------------------------------------------------
# HTTP plumbing
# ---------------------------------------------------------------------------

_MIME = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".js": "application/javascript; charset=utf-8"}


class Handler(BaseHTTPRequestHandler):
    server_version = "PredictorDashboard/2.0"

    def log_message(self, fmt, *args):  # quieter console
        pass

    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, rel_path: str) -> None:
        path = (STATIC_DIR / rel_path.lstrip("/")).resolve()
        if STATIC_DIR not in path.parents and path != STATIC_DIR:
            self.send_error(403)
            return
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", _MIME.get(path.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        if parsed.path in ("/", "/index.html"):
            self._send_static("index.html")
        elif parsed.path == "/api/instruments":
            self._send_json({"instruments": load_instruments_registry(CONFIG.paths.root)})
        elif parsed.path == "/api/instruments/search":
            q = qs.get("q", [""])[0]
            try:
                results = nse_symbols.search(CONFIG.paths.reference, q)
                self._send_json({"results": results})
            except Exception as exc:
                self._send_json({"results": [], "error": str(exc)})
        elif parsed.path == "/api/status":
            instrument = qs.get("instrument", ["NIFTY50"])[0]
            self._send_json(build_status(instrument))
        elif parsed.path == "/api/run/output":
            instrument = qs.get("instrument", ["NIFTY50"])[0]
            since = int(qs.get("since", ["0"])[0])
            self._send_json(run_snapshot(instrument, since))
        elif parsed.path in ("/style.css", "/app.js"):
            self._send_static(parsed.path)
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            payload = {}

        if parsed.path == "/api/run":
            instrument = payload.get("instrument", "NIFTY50")
            run_id, err = start_run(instrument, payload)
            if err:
                self._send_json({"ok": False, "error": err}, status=409)
            else:
                self._send_json({"ok": True, "run_id": run_id})
        elif parsed.path == "/api/run/stop":
            instrument = payload.get("instrument") or qs.get("instrument", ["NIFTY50"])[0]
            ok = stop_run(instrument)
            self._send_json({"ok": ok})
        elif parsed.path == "/api/instruments":
            try:
                result = register_instrument(payload)
                if result.get("error"):
                    self._send_json({"ok": False, **result}, status=502)
                else:
                    self._send_json({"ok": True, **result})
                    start_run(result["entry"]["key"], {})
            except ValueError as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=400)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)
        else:
            self.send_error(404)


def main() -> None:
    CONFIG.paths.ensure()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    url = f"http://127.0.0.1:{PORT}/"
    print(f"Predictor dashboard running at {url}")
    print("Press Ctrl+C to stop.")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
