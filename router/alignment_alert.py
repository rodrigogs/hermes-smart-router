"""Alignment alerts (F9): deliver through channels the host already has.

Each channel in ``alert.channels`` uses the ``send_message`` target syntax
(``telegram``, ``telegram:<chat>``, ``discord:#ops`` ...). ``send_message`` is not
callable from a worker (F1 spike), so delivery shells out to ``hermes send``, which
reuses the host's existing credentials: no new secret is read or stored here.

If a send fails, the alert is appended to an outbox JSONL that the gateway side
drains (the gateway is long-lived and already connected). One alert per
``(task, run, verdict)`` is enforced by a persisted ledger. No transcript is
included unless ``alert.include_transcript_excerpt`` is true.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

try:
    from .paths import state_dir
except ImportError:  # pragma: no cover - flat harness
    from router.paths import state_dir

LEDGER_FILE = "alignment-alerts.json"
OUTBOX_FILE = "alignment-outbox.jsonl"
MAX_LEDGER = 500
SEND_TIMEOUT = 30
DEFAULT_ON = ("block", "adjust")
EXCERPT_CHARS = 500

_LOCK = threading.Lock()


def category(verdict: str, mode: str) -> str:
    return "shadow_block" if mode == "shadow" and verdict == "block" else verdict


def _ledger_path() -> Path:
    return state_dir() / LEDGER_FILE


def _read_ledger() -> List[str]:
    try:
        data = json.loads(_ledger_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [str(x) for x in data] if isinstance(data, list) else []


def _claim(key: str) -> bool:
    """Atomically record ``key``; False if already alerted."""
    with _LOCK:
        seen = _read_ledger()
        if key in seen:
            return False
        seen.append(key)
        seen = seen[-MAX_LEDGER:]
        path = _ledger_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(seen), encoding="utf-8")
        os.replace(tmp, path)
        return True


def format_message(entry: Mapping[str, Any], alert_cfg: Mapping[str, Any]) -> str:
    lines = [
        f"Alignment {entry.get('verdict')} on task {entry.get('task_id')} "
        f"(run {entry.get('run_id')}, mode {entry.get('mode')}, "
        f"confidence {entry.get('confidence')})"
    ]
    if alert_cfg.get("include_reasons", True):
        lines += [f"- {r}" for r in (entry.get("reasons") or [])]
    if alert_cfg.get("include_transcript_excerpt", False):
        ev = " | ".join(str(e) for e in (entry.get("evidence") or []))
        if ev:
            lines.append("evidence: " + ev[:EXCERPT_CHARS])
    return "\n".join(lines)


def _hermes_send(target: str, text: str) -> bool:
    proc = subprocess.run(
        ["hermes", "send", "-t", target, "-q", text],
        capture_output=True,
        text=True,
        timeout=SEND_TIMEOUT,
        check=False,
    )
    return proc.returncode == 0


def _outbox(target: str, text: str, key: str) -> None:
    path = state_dir() / OUTBOX_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"key": key, "target": target, "text": text}) + "\n")


def send_alert(
    config: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    sender: Optional[Callable[[str, str], bool]] = None,
) -> Dict[str, str]:
    """Return ``{channel: "sent"|"outbox"}``; empty when nothing was due. Never raises."""
    try:
        alert_cfg = config.get("alert") or {}
        channels = [c for c in (alert_cfg.get("channels") or []) if isinstance(c, str) and c]
        cat = category(str(entry.get("verdict")), str(entry.get("mode")))
        if (
            not channels
            or entry.get("failed_open")
            or cat not in (alert_cfg.get("on") or DEFAULT_ON)
        ):
            return {}
        key = f"{entry.get('task_id')}|{entry.get('run_id')}|{cat}"
        if not _claim(key):
            return {}
        text = format_message(entry, alert_cfg)
        send = sender or _hermes_send
        out: Dict[str, str] = {}
        for ch in channels:
            try:
                ok = bool(send(ch, text))
            except Exception:  # noqa: BLE001 - fall back to the outbox
                ok = False
            if not ok:
                _outbox(ch, text, key)
            out[ch] = "sent" if ok else "outbox"
        return out
    except Exception:  # noqa: BLE001 - alerting must never break the judge thread
        return {}
