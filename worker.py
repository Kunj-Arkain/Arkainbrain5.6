#!/usr/bin/env python3
"""
ARKAINBRAIN — Subprocess Worker

Runs pipeline/recon jobs in a separate process to avoid:
- Import deadlocks (crewai module locks)
- Thread-safety issues with OpenAI clients
- GIL contention on CPU-bound simulation

Usage (called by web_app.py, not directly):
    python worker.py pipeline <job_id> '<json_params>'
    python worker.py recon <job_id> <state_name>
"""

import json
import os
import sqlite3
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ── Suppress CrewAI tracing prompts in subprocess ──
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"
os.environ["DO_NOT_TRACK"] = "1"
os.environ["LITELLM_LOG"] = "ERROR"  # Suppress litellm info/debug logs

# ── Suppress noisy library loggers that bypass stdout capture ──
import logging
for _noisy in ("litellm", "httpx", "httpcore", "openai", "urllib3", "crewai"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ── Pre-create CrewAI config to prevent tracing prompt entirely ──
# CrewAI checks ~/.crewai/ for stored preferences. If missing, it asks.
_crewai_dirs = [
    Path.home() / ".crewai",
    Path("/tmp/crewai_storage"),
]
for _d in _crewai_dirs:
    _d.mkdir(parents=True, exist_ok=True)
    _cfg = _d / "config.json"
    if not _cfg.exists():
        _cfg.write_text(json.dumps({"tracing_enabled": False, "tracing_disabled": True}))
    # Also write the db3 format some versions use
    _db = _d / "crewai_config.db"
    if not _db.exists():
        try:
            import sqlite3 as _sq
            _c = _sq.connect(str(_db))
            _c.execute("CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT)")
            _c.execute("INSERT OR REPLACE INTO config VALUES ('tracing_enabled', 'false')")
            _c.commit()
            _c.close()
        except Exception:
            pass

os.environ["CREWAI_STORAGE_DIR"] = "/tmp/crewai_storage"

# ── Redirect stdin to prevent any interactive prompts ──
sys.stdin = open(os.devnull, "r")

# ── OpenAI SDK retry: exponential backoff on 429s ──
os.environ.setdefault("OPENAI_MAX_RETRIES", "5")
os.environ.setdefault("OPENAI_TIMEOUT", "120")

from dotenv import load_dotenv
load_dotenv()

DB_PATH = os.getenv("DB_PATH", "arkainbrain.db")
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Global pipeline timeout (seconds) ──
# If the entire pipeline exceeds this, the worker self-terminates.
# Override: PIPELINE_TIMEOUT_SECONDS=5400 (default: 1 hour)
PIPELINE_TIMEOUT = int(os.getenv("PIPELINE_TIMEOUT_SECONDS", "3600"))


def _start_watchdog(job_id: str, timeout: int):
    """Watchdog thread that kills the process if the pipeline exceeds timeout."""
    import threading, signal

    def _timeout_handler():
        try:
            db = sqlite3.connect(DB_PATH)
            db.execute(
                "UPDATE jobs SET status='failed', error=? WHERE id=? AND status='running'",
                (f"Pipeline timed out after {timeout}s", job_id),
            )
            db.commit()
            db.close()
        except Exception:
            pass
        print(f"\n[WATCHDOG] Pipeline {job_id} exceeded {timeout}s — forcing exit")
        os._exit(1)  # Hard exit — no cleanup, no deadlock

    timer = threading.Timer(timeout, _timeout_handler)
    timer.daemon = True
    timer.start()
    return timer


class JobLogger:
    """Per-job logging: explicit log() calls go to Railway stdout + log file.
    All captured stdout/stderr (CrewAI, litellm, etc.) goes ONLY to log file.
    This prevents Railway's 500 logs/sec rate limit."""

    def __init__(self, job_id: str):
        self.job_id = job_id
        self.log_path = LOG_DIR / f"{job_id}.log"
        self.log_file = open(self.log_path, "w", buffering=1)  # line-buffered
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr

    def log(self, msg: str):
        """Write a status line to BOTH Railway logs and the job log file."""
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self._original_stdout.write(line + "\n")
        self._original_stdout.flush()
        self.log_file.write(line + "\n")

    def capture_output(self):
        """Redirect stdout/stderr to log file ONLY (not Railway stdout).
        Live log viewer reads the file via SSE — users still see everything."""
        sys.stdout = _LogFileWriter(self.log_file)
        sys.stderr = _LogFileWriter(self.log_file)

    def close(self):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        self.log_file.close()


class _LogFileWriter:
    """File-like writer that sends ALL output to the log file only.
    Railway stdout is never touched — prevents 500 logs/sec rate limit."""

    def __init__(self, log_file):
        self._log = log_file

    def write(self, data):
        if data:
            try:
                self._log.write(data)
                self._log.flush()
            except (ValueError, IOError):
                pass

    def flush(self):
        try:
            self._log.flush()
        except (ValueError, IOError):
            pass

    def isatty(self):
        return False

    def fileno(self):
        raise OSError("log-only stream has no fileno")

    @property
    def encoding(self):
        return 'utf-8'

    @property
    def errors(self):
        return 'strict'

    def readable(self):
        return False

    def writable(self):
        return True

    def seekable(self):
        return False


def update_db(job_id: str, **kw):
    """Update job in SQLite (concurrency-safe with WAL mode)."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    sets = ",".join(f"{k}=?" for k in kw)
    conn.execute(f"UPDATE jobs SET {sets} WHERE id=?", list(kw.values()) + [job_id])
    conn.commit()
    conn.close()


def setup_openai_retry():
    """Configure OpenAI SDK and litellm for rate-limit retries with backoff."""
    # CrewAI uses the OpenAI SDK directly (not litellm).
    # The SDK reads OPENAI_MAX_RETRIES env var for auto-retry on 429s.
    os.environ.setdefault("OPENAI_MAX_RETRIES", "5")
    os.environ.setdefault("OPENAI_TIMEOUT", "120")

    # Also configure litellm if present (used by some tool calls)
    try:
        import litellm
        litellm.num_retries = 5
        litellm.request_timeout = 120
        litellm.set_verbose = False       # Suppress print()-based debug output
        litellm.suppress_debug_info = True
    except ImportError:
        pass

    # Monkey-patch OpenAI client defaults for maximum resilience
    try:
        import openai
        openai.default_headers = {**(openai.default_headers or {})}
        # Increase default max_retries from 2 to 5
        if hasattr(openai, '_default_max_retries'):
            openai._default_max_retries = 5
    except (ImportError, AttributeError):
        pass


def run_pipeline(job_id: str, params_json: str):
    """Run the full slot pipeline."""
    logger = JobLogger(job_id)
    logger.capture_output()  # Route all print output (including CrewAI) to log file
    setup_openai_retry()
    update_db(job_id, status="running", current_stage="Initializing")
    logger.log(f"Pipeline {job_id} starting (timeout: {PIPELINE_TIMEOUT}s)")

    # Start watchdog — kills process if pipeline exceeds timeout
    watchdog = _start_watchdog(job_id, PIPELINE_TIMEOUT)

    try:
        p = json.loads(params_json)

        # ── Auto State Recon for unknown US states ──
        KNOWN_JURISDICTIONS = {
            "uk", "malta", "ontario", "new jersey", "curacao", "curaçao",
            "sweden", "spain", "gibraltar", "isle of man", "alderney",
            "denmark", "italy", "portugal", "france", "germany",
            "michigan", "pennsylvania", "west virginia", "connecticut",
        }
        US_STATES = {
            "alabama","alaska","arizona","arkansas","california","colorado",
            "connecticut","delaware","florida","georgia","hawaii","idaho",
            "illinois","indiana","iowa","kansas","kentucky","louisiana","maine",
            "maryland","massachusetts","michigan","minnesota","mississippi",
            "missouri","montana","nebraska","nevada","new hampshire","new jersey",
            "new mexico","new york","north carolina","north dakota","ohio",
            "oklahoma","oregon","pennsylvania","rhode island","south carolina",
            "south dakota","tennessee","texas","utah","vermont","virginia",
            "washington","west virginia","wisconsin","wyoming",
        }

        if p.get("enable_recon", False):
            states_needing_recon = [
                m for m in p["target_markets"]
                if m.strip().lower() in US_STATES and m.strip().lower() not in KNOWN_JURISDICTIONS
            ]
            for state in states_needing_recon:
                try:
                    update_db(job_id, current_stage=f"State Recon: {state}")
                    logger.log(f"Running recon for {state}")
                    from flows.state_recon import run_recon
                    run_recon(state, auto=True, job_id=job_id)
                    logger.log(f"Recon complete for {state}")
                except Exception as e:
                    logger.log(f"WARN: State recon failed for {state}: {e}")

        # ── Build game input ──
        from models.schemas import GameIdeaInput, Volatility, FeatureType

        feats = []
        for f in p.get("requested_features", []):
            try:
                feats.append(FeatureType(f))
            except ValueError:
                pass

        gi = GameIdeaInput(
            theme=p["theme"],
            target_markets=p["target_markets"],
            volatility=Volatility(p["volatility"]),
            target_rtp=p["target_rtp"],
            grid_cols=p["grid_cols"],
            grid_rows=p["grid_rows"],
            ways_or_lines=str(p["ways_or_lines"]),
            max_win_multiplier=p["max_win_multiplier"],
            art_style=p["art_style"],
            requested_features=feats,
            competitor_references=p.get("competitor_references", []),
            special_requirements=p.get("special_requirements", ""),
        )

        # ── Run pipeline ──
        update_db(job_id, current_stage="Pipeline executing")
        logger.log("Pipeline executing — agents starting")
        interactive = p.get("interactive", False)
        if interactive:
            os.environ["HITL_ENABLED"] = "true"

        from flows.pipeline import SlotStudioFlow
        flow = SlotStudioFlow(auto_mode=not interactive)
        flow.state.game_idea = gi
        flow.state.job_id = job_id
        fs = flow.kickoff()

        od = getattr(fs, "output_dir", None) if hasattr(fs, "output_dir") else None
        watchdog.cancel()  # Pipeline completed — cancel the watchdog
        update_db(
            job_id,
            status="complete",
            output_dir=str(od) if od else None,
            completed_at=datetime.now().isoformat(),
        )
        logger.log(f"Pipeline {job_id} COMPLETE → {od}")

    except Exception as e:
        update_db(job_id, status="failed", error=str(e)[:500])
        logger.log(f"Pipeline {job_id} FAILED: {e}")
        traceback.print_exc()  # Goes to captured stderr → log file
    finally:
        watchdog.cancel()
        logger.close()


def run_recon_job(job_id: str, state_name: str):
    """Run state recon."""
    logger = JobLogger(job_id)
    logger.capture_output()  # Route all print output (including CrewAI) to log file
    setup_openai_retry()
    update_db(job_id, status="running", current_stage=f"Researching {state_name}...")
    logger.log(f"Recon {job_id} starting for {state_name}")

    try:
        from flows.state_recon import run_recon
        result = run_recon(state_name, auto=True, job_id=job_id)
        od = getattr(result, "output_dir", None) if result else None
        update_db(
            job_id,
            status="complete",
            output_dir=str(od) if od else None,
            completed_at=datetime.now().isoformat(),
        )
        logger.log(f"Recon {job_id} COMPLETE → {od}")

    except Exception as e:
        update_db(job_id, status="failed", error=str(e)[:500])
        logger.log(f"Recon {job_id} FAILED: {e}")
        traceback.print_exc()  # Goes to captured stderr → log file
    finally:
        logger.close()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python worker.py [pipeline|recon] <job_id> <params>")
        sys.exit(1)

    job_type = sys.argv[1]
    job_id = sys.argv[2]

    if job_type == "pipeline":
        params_json = sys.argv[3] if len(sys.argv) > 3 else "{}"
        run_pipeline(job_id, params_json)
    elif job_type == "recon":
        state_name = sys.argv[3] if len(sys.argv) > 3 else "unknown"
        run_recon_job(job_id, state_name)
    else:
        print(f"Unknown job type: {job_type}")
        sys.exit(1)
