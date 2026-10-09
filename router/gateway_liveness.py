"""Inert-hook alert for the gateway: plugin code deployed without a restart.

Hermes loads plugins once per process, so a file edited after the gateway
started is NOT running. ``check`` compares the gateway process start with the
newest plugin-file mtime and, when the code is newer, logs a warning and appends
an ``inert_hook`` row; after a restart the process start is newer, so it is
silent. The first ``on_kanban_dispatch_tick`` after boot also appends one
``boot_heartbeat`` row, proving the hook path is alive. Observer only: never raises.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from .durable_decision_log import routes_path
except ImportError:  # pragma: no cover - flat layout used by the test harness
    from router.durable_decision_log import routes_path

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_STATE: Dict[str, bool] = {"heartbeat_done": False}
_PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def liveness_path() -> Path:
    return routes_path().with_name("gateway_liveness.jsonl")


def process_start_ts() -> float:
    """Epoch start of this process (/proc), falling back to now."""
    try:
        with open("/proc/self/stat", "rb") as fh:
            ticks = int(fh.read().rsplit(b")", 1)[1].split()[19])
        with open("/proc/stat", "rb") as fh:
            btime = next(
                int(x.split()[1]) for x in fh.read().splitlines() if x.startswith(b"btime"))
        return btime + ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, StopIteration, IndexError):  # pragma: no cover
        return time.time()


def plugin_mtime(root: Optional[Path] = None) -> float:
    """Newest mtime among the plugin's own .py/.yaml files (0.0 if none)."""
    root = root or _PLUGIN_ROOT
    newest = 0.0
    for pattern in ("*.py", "*.yaml", "router/*.py"):
        for f in root.glob(pattern):
            try:
                newest = max(newest, f.stat().st_mtime)
            except OSError:  # pragma: no cover - raced deletion
                pass
    return newest


def _append(row: Dict[str, Any]) -> None:
    try:
        path = liveness_path()
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("could not persist gateway liveness row: %s", exc)


def check(started: Optional[float] = None, mtime: Optional[float] = None) -> Dict[str, Any]:
    """Return the verdict; alert (log + journal row) only when code is newer."""
    started = process_start_ts() if started is None else started
    mtime = plugin_mtime() if mtime is None else mtime
    inert = mtime > started
    verdict = {"inert": inert, "process_started_at": started, "code_mtime": mtime}
    if inert:
        logger.warning(
            "hermes-smart-router: plugin files are newer than this gateway process by "
            "%.0fs — hooks are running OLD code; restart the gateway.", mtime - started)
        _append({"schema": "gateway-liveness/1", "ts": time.time(), "event": "inert_hook",
                 **verdict})
    return verdict


def on_dispatch_tick(**_kwargs: Any) -> None:
    """``on_kanban_dispatch_tick``: heartbeat once per boot, plus the inert check."""
    try:
        with _LOCK:
            if _STATE["heartbeat_done"]:
                return
            _STATE["heartbeat_done"] = True
        verdict = check()
        _append({"schema": "gateway-liveness/1", "ts": time.time(),
                 "event": "boot_heartbeat", **verdict})
    except Exception:
        logger.debug("gateway liveness tick failed", exc_info=True)
