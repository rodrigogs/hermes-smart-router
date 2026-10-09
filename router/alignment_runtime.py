"""Session-alignment hook wiring (F7): observer, daemon-thread evaluation, shadow mode.

``observe_post_api_request`` is the ``post_api_request`` callback body. It does the
cheap part inline (read counters, load persisted state, run the pure trigger
engine) and hands everything slow (package build, judge call, log write) to a
daemon thread, so the agent loop never waits on the judge. It never raises.

State is a JSON file under ``router.paths.state_dir()`` and is re-read on every
call, so there is no module state for a plugin reload to lose. Only the judge
breaker lives in memory (it is a cooldown, not a ledger).

``mode: shadow`` judges and logs; no action is ever taken. Executors (adjust,
block, comment) belong to F8, so until then every mode only records.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

try:  # package import (installed plugin) or flat import (dev harness)
    from . import alignment, alignment_alert, alignment_judge, alignment_package
    from .breaker import BreakerState
    from .model_lint import load_cache
    from .paths import hermes_root, state_dir
except ImportError:  # pragma: no cover - exercised only by the flat harness
    from router import alignment, alignment_alert, alignment_judge, alignment_package
    from router.breaker import BreakerState
    from router.model_lint import load_cache
    from router.paths import hermes_root, state_dir

logger = logging.getLogger(__name__)

STATE_FILE = "alignment-state.json"
DEFAULT_LOG = "alignment.jsonl"
MAX_SESSIONS = 200

_LOCK = threading.Lock()
_BREAKER = BreakerState({})
_HISTORY: Dict[str, Any] = {}  # session_id -> last conversation_history seen pre-call


def stash_history(session_id: str, history: Any) -> None:
    """``pre_api_request`` carries the transcript; ``post_api_request`` does not."""
    if not session_id or not isinstance(history, list):
        return
    with _LOCK:
        _HISTORY.pop(session_id, None)
        _HISTORY[session_id] = history
        while len(_HISTORY) > 32:
            _HISTORY.pop(next(iter(_HISTORY)))


def _state_path() -> Path:
    return state_dir() / STATE_FILE


def _log_path(config: Mapping[str, Any]) -> Path:
    name = (config.get("log") or {}).get("path") or DEFAULT_LOG
    p = Path(name)
    return p if p.is_absolute() else state_dir() / p


def _read_all() -> Dict[str, Any]:
    try:
        data = json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def load_state(session_id: str) -> alignment.AlignmentState:
    raw = _read_all().get(session_id)
    if not isinstance(raw, dict):
        return alignment.AlignmentState()
    return alignment.AlignmentState(
        fired=frozenset(tuple(k) for k in raw.get("fired", []) if len(k) == 3),
        evaluations=int(raw.get("evaluations", 0)),
        last_action_iteration=raw.get("last_action_iteration"),
        disabled_until=raw.get("disabled_until"),
    )


def save_state(session_id: str, st: alignment.AlignmentState) -> None:
    allp = _read_all()
    allp.pop(session_id, None)
    allp[session_id] = {
        "fired": sorted(list(k) for k in st.fired),
        "evaluations": st.evaluations,
        "last_action_iteration": st.last_action_iteration,
        "disabled_until": st.disabled_until,
    }
    while len(allp) > MAX_SESSIONS:
        allp.pop(next(iter(allp)))
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(allp), encoding="utf-8")
    os.replace(tmp, path)


def append_log(config: Mapping[str, Any], entry: Dict[str, Any]) -> None:
    path = _log_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def _in_scope(config: Mapping[str, Any], kanban_task: Optional[str]) -> bool:
    scopes = config.get("scopes") or ["kanban"]
    return "kanban" in scopes and bool(kanban_task)


def _evaluate(
    ctx: Any, config: Mapping[str, Any], decision: alignment.Decision,
    counters: alignment.Counters, worker_provider: Optional[str], kanban_task: str,
    history: Any, now: float,
) -> None:
    """Thread body: package, judge, log, record. Never raises."""
    try:
        judge_cfg = config.get("judge") or {}
        chain = alignment_judge.parse_chain(judge_cfg.get("chain"))
        try:
            chain = alignment_judge.usable_chain(
                chain, load_cache(hermes_root() / "provider_models_cache.json")
            ) or chain
        except Exception:  # noqa: BLE001 - cache is advisory
            pass
        privacy = config.get("privacy") or {}
        pkg = alignment_package.build_package(
            list(history or []),
            max_input_tokens=int(judge_cfg.get("max_input_tokens", 120000)),
            redact_secrets=bool(privacy.get("redact", True)),
        )
        verdict = alignment_judge.judge(
            ctx, pkg["text"], chain=chain, worker_provider=worker_provider,
            breaker=_BREAKER, now=now,
            require_distinct=bool(judge_cfg.get("require_distinct_provider", True)),
            timeout_seconds=float(judge_cfg.get("timeout_seconds", 90)),
        )
        acted = verdict.verdict != "continue" and not verdict.failed_open
        if acted:  # the observer already recorded the rung; only start the cooldown
            with _LOCK:
                cur = load_state(counters.session_id)
                save_state(counters.session_id, alignment.AlignmentState(
                    cur.fired, cur.evaluations, counters.iteration, cur.disabled_until,
                ))
        mode = config.get("mode", "shadow")
        entry = {
            "ts": now, "session_id": counters.session_id, "task_id": kanban_task,
            "run_id": counters.run_id, "iteration": counters.iteration,
            "rung": decision.rung, "pct": decision.pct, "mode": mode,
            "verdict": verdict.verdict, "confidence": verdict.confidence,
            "reasons": verdict.reasons, "evidence": verdict.evidence,
            "failed_open": verdict.failed_open,
            "judge": f"{verdict.provider}/{verdict.model}",
            "tokens": pkg.get("tokens"), "action_taken": False,
        }
        append_log(config, entry)
        alignment_alert.send_alert(config, entry)
    except Exception as exc:  # noqa: BLE001 - a daemon thread must not leak
        logger.warning("hermes-smart-router: alignment evaluation failed: %s", exc)


def observe_post_api_request(
    ctx: Any,
    config: Mapping[str, Any],
    *,
    spawn: Optional[Callable[..., Any]] = None,
    now: Optional[float] = None,
    **kw: Any,
) -> Optional[threading.Thread]:
    """Cheap inline check; the evaluation itself runs on a daemon thread.

    Returns the started thread (or None) so tests can join it.
    """
    try:
        if not isinstance(config, Mapping) or not config.get("enabled", False):
            return None
        kanban_task = os.environ.get("HERMES_KANBAN_TASK", "")
        if not _in_scope(config, kanban_task):
            return None
        session_id = str(kw.get("session_id") or "")
        if not session_id:
            return None
        counters = alignment.Counters(
            session_id=session_id,
            run_id=os.environ.get("HERMES_KANBAN_RUN_ID", ""),
            iteration=int(kw.get("api_call_count") or 0),
            max_iterations=kw.get("max_iterations"),
            budget_used=kw.get("iteration_budget_used"),
            budget_max=kw.get("iteration_budget_max"),
            profile=os.environ.get("HERMES_PROFILE") or None,
        )
        ts = time.time() if now is None else now
        with _LOCK:
            decision = alignment.evaluate(counters, config, load_state(session_id), ts)
            history = _HISTORY.get(session_id)
            if decision.fire:
                # Mark fired BEFORE the thread runs so the next call cannot re-fire
                # the rung while the judge is still thinking.
                save_state(session_id, alignment.record(
                    load_state(session_id), decision, iteration=counters.iteration, acted=False,
                ))
        if not decision.fire:
            return None
        thread = (spawn or threading.Thread)(
            target=_evaluate,
            args=(ctx, config, decision, counters, kw.get("provider"), kanban_task,
                  history, ts),
            name="alignment-eval", daemon=True,
        )
        thread.start()
        return thread
    except Exception as exc:  # noqa: BLE001 - an observer must never break a turn
        logger.warning("hermes-smart-router: alignment observer failed: %s", exc)
        return None
