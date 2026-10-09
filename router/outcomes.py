"""Per-card outcome journal: what happened to each routed ``(task_id, run_id)``.

The router decides at ``pre_kanban_dispatch`` and, before this module, never
learned whether the card it routed succeeded. Three Hermes lifecycle hooks carry
the answer — ``kanban_task_completed``, ``kanban_task_blocked`` and
``on_kanban_worker_exited`` — and each is appended here as one JSON line beside
the route trace. :func:`summarize` joins them to the decisions on
``(task_id, run_id)`` and reports success / blocked / rate_limited rates per tier.

Writing is best-effort by contract (a hook must never break the dispatcher), and
the reader never raises.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from .durable_decision_log import routes_path
except ImportError:  # pragma: no cover - flat layout used by the test harness
    from router.durable_decision_log import routes_path

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_MAX_BYTES = 2 * 1024 * 1024

COMPLETED = "completed"
BLOCKED = "blocked"
RATE_LIMITED = "rate_limited"
UNTRACKED = "no_outcome"


def outcomes_path() -> Path:
    """Journal beside the route trace, so writer and reader converge on one file."""
    return routes_path().with_name("outcomes.jsonl")


def record_outcome(
    task_id: Any, run_id: Any, outcome: str, source: str, **extra: Any,
) -> bool:
    """Append one outcome row; False (never raises) when it could not be written."""
    if not isinstance(task_id, str) or not task_id:
        return False
    row = {
        "schema": "route-outcome/1", "ts": time.time(), "task_id": task_id,
        "run_id": run_id, "outcome": str(outcome), "source": source,
    }
    row.update({k: v for k, v in extra.items() if isinstance(v, (str, int, float, bool))})
    path = outcomes_path()
    try:
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if path.stat().st_size + len(line) > _MAX_BYTES:
                    path.replace(path.with_suffix(".jsonl.1"))
            except OSError:
                pass  # absent: nothing to rotate
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning("could not persist route outcome: %s", exc)
        return False
    return True


def read_outcomes() -> List[Dict[str, Any]]:
    """Rows oldest→newest (rotated backup first); corrupt lines are skipped."""
    rows: List[Dict[str, Any]] = []
    base = outcomes_path()
    for path in (base.with_suffix(".jsonl.1"), base):
        try:
            raw = path.read_bytes().splitlines()
        except OSError:
            continue
        for line in raw:
            try:
                obj = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                continue
            if isinstance(obj, dict) and obj.get("schema") == "route-outcome/1":
                rows.append(obj)
    return rows


def tier_of(entry: Dict[str, Any]) -> str:
    """Tier a decision belongs to: the classifier tier if recorded, else the declared model."""
    for step in entry.get("steps") or []:
        inputs = step.get("in") if isinstance(step, dict) else None
        tier = inputs.get("tier") if isinstance(inputs, dict) else None
        if isinstance(tier, str) and tier:
            return tier
    out = entry.get("output")
    model = out.get("model") if isinstance(out, dict) else None
    return model if isinstance(model, str) and model else "unknown"


def _key(task_id: Any, run_id: Any) -> Optional[tuple]:
    if not isinstance(task_id, str) or not task_id:
        return None
    if not isinstance(run_id, (str, int, float, bool, type(None))):
        return None
    return (task_id, run_id)


def summarize(
    decisions: Iterable[Dict[str, Any]], outcomes: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    """Success / blocked / rate_limited rates per tier.

    The LAST outcome row per ``(task_id, run_id)`` wins. Decisions with no
    outcome row are counted as ``no_outcome`` and EXCLUDED from the rate
    denominator: an in-flight card is not a failure. Rates are None when a tier
    has no resolved card.
    """
    final: Dict[tuple, str] = {}
    for row in outcomes:
        key = _key(row.get("task_id"), row.get("run_id"))
        if key is not None:
            final[key] = str(row.get("outcome"))
    tiers: Dict[str, Dict[str, Any]] = {}
    for entry in decisions:
        key = _key(entry.get("task_id"), entry.get("run_id"))
        if key is None:
            continue
        bucket = tiers.setdefault(tier_of(entry), {"cards": 0, "counts": {}})
        bucket["cards"] += 1
        outcome = final.get(key, UNTRACKED)
        bucket["counts"][outcome] = bucket["counts"].get(outcome, 0) + 1
    result: Dict[str, Any] = {}
    for tier, bucket in sorted(tiers.items()):
        counts = bucket["counts"]
        resolved = bucket["cards"] - counts.get(UNTRACKED, 0)
        result[tier] = {
            "cards": bucket["cards"], "resolved": resolved, "counts": counts,
            "success_rate": _rate(counts, COMPLETED, resolved),
            "blocked_rate": _rate(counts, BLOCKED, resolved),
            "rate_limited_rate": _rate(counts, RATE_LIMITED, resolved),
        }
    return {"tiers": result, "outcomes_path": str(outcomes_path())}


def _rate(counts: Dict[str, int], name: str, resolved: int) -> Optional[float]:
    return round(counts.get(name, 0) / resolved, 4) if resolved else None
