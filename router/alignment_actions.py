"""Session-alignment action executors (F8): adjust, block, comment, ack override.

Only ``mode: enforce`` acts. ``shadow`` and ``alert`` return an ``Outcome`` with
``acted=False``. A ``continue`` verdict (or any verdict the gate downgrades to
continue) returns before the kanban DB is opened, so it never touches the card.

Adjust channel, best first:
  steer        upstream ``AIAgent.steer`` (F2). Not landed, so never chosen unless the
               caller says ``upstream_steer=True``.
  pre_tool_call  one-shot: the next non-kanban tool call is vetoed with the steer text
               (the model sees it as the tool result). Costs one wasted call.
  pre_llm_call   context injected before the next model call; used in goal-mode, where
               a turn may end without any tool call.
Block: ``block_task(kind="needs_input", expected_run_id=...)`` plus a comment, then every
following non-kanban tool call is vetoed so the worker winds down. ``kanban_*`` tools are
never vetoed, so the worker can still complete/comment/heartbeat.

Pending directives live in a JSON file (re-read on every hook call), so a plugin reload
loses nothing. Every function here is fail-open: an error means "no action".
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

try:  # package import (installed plugin) or flat import (dev harness)
    from .paths import state_dir
except ImportError:  # pragma: no cover - exercised only by the flat harness
    from router.paths import state_dir

logger = logging.getLogger(__name__)

PENDING_FILE = "alignment-pending.json"
ACK_MARKER = "[alignment-ack]"
AUTHOR = "alignment"
MAX_SESSIONS = 200
_LOCK = threading.Lock()


@dataclass(frozen=True)
class Outcome:
    action: str  # continue | adjust | block
    acted: bool
    channel: str = ""  # steer | pre_tool_call | pre_llm_call | "" (block / none)
    reason: str = ""  # why it did not act (or what it did)


# --------------------------------------------------------------------------- gate
def _num(cfg: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(cfg.get(key, default))
    except (TypeError, ValueError):
        return default


def gate(verdict: Any, config: Mapping[str, Any], adjusts_so_far: int) -> str:
    """Effective action after the code-side thresholds (the model is not trusted)."""
    if getattr(verdict, "failed_open", "") or verdict.verdict not in ("adjust", "block"):
        return "continue"
    th = config.get("thresholds") or {}
    if th.get("require_evidence", True) and not verdict.evidence:
        return "continue"
    conf = float(verdict.confidence or 0.0)
    if verdict.verdict == "adjust":
        if conf < _num(th, "adjust_min_confidence", 0.6) or not verdict.steer_message.strip():
            return "continue"
        if adjusts_so_far >= int(_num(th, "adjust_limit", 2)):
            # a further drift becomes a block proposal, never a third steer
            return "block" if conf >= _num(th, "block_min_confidence", 0.85) else "continue"
        return "adjust"
    if conf < _num(th, "block_min_confidence", 0.85):
        return "continue"
    if adjusts_so_far < 1 and not th.get("allow_direct_block", False):
        # block needs an earlier adjust that did not resolve it; fall back to adjusting
        if verdict.steer_message.strip() and conf >= _num(th, "adjust_min_confidence", 0.6):
            return "adjust"
        return "continue"
    return "block"


def choose_channel(deliver: str, *, upstream_steer: bool, goal_mode: bool) -> str:
    if deliver == "steer":
        return "steer" if upstream_steer else ("pre_llm_call" if goal_mode else "pre_tool_call")
    if deliver == "pre_tool_call":
        return "pre_tool_call"
    # auto
    if upstream_steer:
        return "steer"
    return "pre_llm_call" if goal_mode else "pre_tool_call"


# --------------------------------------------------------------------------- pending store
def _path() -> Path:
    return state_dir() / PENDING_FILE


def _read() -> Dict[str, Any]:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write(data: Dict[str, Any]) -> None:
    sessions = data.setdefault("sessions", {})
    while len(sessions) > MAX_SESSIONS:
        sessions.pop(next(iter(sessions)))
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def adjust_count(task_id: str) -> int:
    return int((_read().get("adjusts") or {}).get(task_id, 0))


def _update(session_id: str, task_id: str, **fields: Any) -> None:
    with _LOCK:
        data = _read()
        sessions = data.setdefault("sessions", {})
        cur = sessions.pop(session_id, {})
        cur.update(fields)
        sessions[session_id] = cur
        if fields.get("kind") == "adjust":
            adj = data.setdefault("adjusts", {})
            adj[task_id] = int(adj.get(task_id, 0)) + 1
        _write(data)


# --------------------------------------------------------------------------- kanban surface
def load_kanban() -> Any:  # pragma: no cover - needs the host runtime
    from hermes_cli import kanban_db as kb
    return kb


def _acknowledged(kb: Any, conn: Any, task_id: str) -> bool:
    """A human ``[alignment-ack]`` comment suppresses actions for this card."""
    for c in kb.list_comments(conn, task_id):
        if ACK_MARKER in (c.body or "") and getattr(c, "author", "") != AUTHOR:
            return True
    return False


def _comment_text(verdict: Any, kind: str) -> str:
    lines = [f"[alignment] {kind} (confidence {verdict.confidence:.2f})"]
    lines += [f"- {r}" for r in verdict.reasons]
    for e in verdict.evidence:
        lines.append(f"  evidence[{e.get('message_index')}]: {e.get('quote', '')}")
    if kind == "adjust" and verdict.steer_message:
        lines.append(f"steer: {verdict.steer_message}")
    lines.append(f"Reply with {ACK_MARKER} on this card to suppress further alignment actions.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- executor
def apply(
    verdict: Any,
    config: Mapping[str, Any],
    *,
    task_id: str,
    run_id: str,
    session_id: str,
    goal_mode: bool = False,
    upstream_steer: bool = False,
    kb: Any = None,
    board: Optional[str] = None,
) -> Outcome:
    """Carry out a verdict. Never raises; returns what happened."""
    try:
        action = gate(verdict, config, adjust_count(task_id))
        if action == "continue":
            return Outcome("continue", False, reason="no_action")  # DB never opened
        if config.get("mode", "shadow") != "enforce":
            return Outcome(action, False, reason="mode_" + str(config.get("mode", "shadow")))
        kb = kb or load_kanban()
        actions = config.get("actions") or {}
        conn = kb.connect(board=board or os.environ.get("HERMES_KANBAN_BOARD"))
        try:
            if _acknowledged(kb, conn, task_id):
                return Outcome(action, False, reason="acknowledged")
            if action == "adjust":
                spec = actions.get("adjust") or {}
                channel = choose_channel(
                    str(spec.get("deliver", "auto")),
                    upstream_steer=upstream_steer, goal_mode=goal_mode,
                )
                _update(session_id, task_id, kind="adjust", channel=channel,
                        steer=verdict.steer_message, delivered=False, blocked=False)
                if spec.get("comment", True):
                    kb.add_comment(conn, task_id, AUTHOR, _comment_text(verdict, "adjust"))
                return Outcome("adjust", True, channel=channel)
            spec = actions.get("block") or {}
            if not spec.get("kanban_block", True):
                return Outcome("block", False, reason="kanban_block_disabled")
            reason = "; ".join(verdict.reasons) or "session drifted from the card"
            ok = kb.block_task(
                conn, task_id, reason=f"[alignment] {reason}", kind="needs_input",
                expected_run_id=int(run_id) if str(run_id).isdigit() else None,
            )
            if not ok:  # CAS lost: the run already ended or changed
                return Outcome("block", False, reason="cas_lost")
            _update(session_id, task_id, kind="block", blocked=True, steer="")
            if spec.get("comment", True):
                kb.add_comment(conn, task_id, AUTHOR, _comment_text(verdict, "block"))
            return Outcome("block", True)
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - fail open
        logger.warning("hermes-smart-router: alignment action failed: %s", exc)
        return Outcome("continue", False, reason="error")


# --------------------------------------------------------------------------- hook bodies
def _is_kanban_tool(name: Any) -> bool:
    return str(name or "").startswith("kanban_")


def pre_tool_call(config: Mapping[str, Any], **kw: Any) -> Optional[Dict[str, str]]:
    """Veto one non-kanban tool call to deliver a steer, or all of them after a block."""
    try:
        if config.get("mode") != "enforce" or _is_kanban_tool(kw.get("tool_name")):
            return None
        sid = str(kw.get("session_id") or "")
        st = (_read().get("sessions") or {}).get(sid)
        if not st:
            return None
        if st.get("blocked"):
            return {"action": "block", "message": (
                "This card was blocked by the session-alignment check and needs human "
                "input. Stop work; summarize where you are with a kanban comment.")}
        if st.get("kind") == "adjust" and st.get("channel") == "pre_tool_call" \
                and not st.get("delivered"):
            _update(sid, "", delivered=True)
            return {"action": "block", "message": (
                "[alignment] Course correction (this call was not run; retry it after "
                "adjusting): " + str(st.get("steer", "")))}
    except Exception as exc:  # noqa: BLE001
        logger.warning("hermes-smart-router: alignment pre_tool_call failed: %s", exc)
    return None


def pre_llm_call(config: Mapping[str, Any], **kw: Any) -> Optional[Dict[str, str]]:
    """Goal-mode channel: inject the steer as context once."""
    try:
        if config.get("mode") != "enforce":
            return None
        sid = str(kw.get("session_id") or "")
        st = (_read().get("sessions") or {}).get(sid)
        if st and st.get("kind") == "adjust" and st.get("channel") == "pre_llm_call" \
                and not st.get("delivered"):
            _update(sid, "", delivered=True)
            return {"context": "[alignment] Course correction: " + str(st.get("steer", ""))}
    except Exception as exc:  # noqa: BLE001
        logger.warning("hermes-smart-router: alignment pre_llm_call failed: %s", exc)
    return None
