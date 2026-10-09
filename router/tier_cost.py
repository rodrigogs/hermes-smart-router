"""Cost and latency per tier over the last N days.

Joins three sources on ``(task_id, run_id)``:

* the route trace, for the tier each card was routed to (``outcomes.tier_of``);
* kanban ``task_runs``, for run latency (``ended_at - started_at``);
* Hermes ``sessions`` rows of kanban workers (title ``Work kanban task <id>``), for
  spend. A session belongs to the run whose time window contains its start.

Cost is the session's recorded ``actual``/``estimated`` figure; when none was recorded
and tokens exist, ``agent.usage_pricing.estimate_usage_cost`` fills it in (that module
also decides what is ``subscription_included``). Subscription-included spend is NEVER
summed into ``cost_usd``: it has its own bucket so a flat-rate model does not read as
free-and-cheap next to a metered one. Unknown cost is counted, not treated as zero.
Readers never raise.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

try:
    from .outcomes import tier_of
    from .paths import hermes_root
except ImportError:  # pragma: no cover - flat layout used by the test harness
    from router.outcomes import tier_of
    from router.paths import hermes_root

logger = logging.getLogger(__name__)

INCLUDED = "subscription_included"
DEFAULT_DAYS = 7
_TASK_RE = re.compile(r"\b(t_[0-9a-f]{6,})\b")
_SLACK_S = 120.0  # a session starts a moment after its run is claimed


def _kanban_dbs() -> List[Path]:
    root = hermes_root()
    paths = [Path(p) for p in sorted(glob.glob(str(root / "kanban" / "boards" / "*" / "kanban.db")))]
    paths.append(root / "kanban.db")
    explicit = os.environ.get("HERMES_KANBAN_DB")
    if explicit:
        paths.append(Path(explicit))
    seen, out = set(), []
    for p in paths:
        if p.exists() and str(p) not in seen:
            seen.add(str(p))
            out.append(p)
    return out


def _ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2)
    conn.row_factory = sqlite3.Row
    return conn


def read_runs(since: float) -> List[Dict[str, Any]]:
    """Kanban runs started since ``since`` across every board."""
    rows: List[Dict[str, Any]] = []
    for path in _kanban_dbs():
        try:
            conn = _ro(path)
            try:
                cur = conn.execute(
                    "SELECT id, task_id, started_at, ended_at, outcome FROM task_runs "
                    "WHERE started_at >= ?", (since,))
                rows.extend(dict(r) for r in cur)
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.warning("could not read kanban runs from %s: %s", path, exc)
    return rows


def read_sessions(since: float) -> List[Dict[str, Any]]:
    """Kanban-worker sessions started since ``since`` from the Hermes state DB."""
    path = hermes_root() / "state.db"
    if not path.exists():
        return []
    try:
        conn = _ro(path)
        try:
            cur = conn.execute(
                "SELECT id, model, title, started_at, billing_provider, billing_base_url, "
                "billing_mode, estimated_cost_usd, actual_cost_usd, cost_status, input_tokens, "
                "output_tokens, cache_read_tokens, cache_write_tokens FROM sessions "
                "WHERE source = 'kanban' AND started_at >= ?", (since,))
            return [dict(r) for r in cur]
        finally:
            conn.close()
    except sqlite3.Error as exc:
        logger.warning("could not read sessions from %s: %s", path, exc)
        return []


def _pricing() -> Optional[Any]:
    try:
        from agent import usage_pricing  # type: ignore
        return usage_pricing
    except Exception:  # absent on CI / foreign layout: stored cost only
        return None


def session_cost(sess: Dict[str, Any], pricing: Optional[Any] = None) -> Dict[str, Any]:
    """``{"included": bool, "usd": float|None}`` for one session row (None = unknown)."""
    mode = sess.get("billing_mode")
    provider = sess.get("billing_provider") or ""
    model = sess.get("model") or ""
    pricing = pricing if pricing is not None else _pricing()
    if pricing is not None and not mode:
        try:
            mode = pricing.resolve_billing_route(
                model, provider=provider, base_url=sess.get("billing_base_url")).billing_mode
        except Exception:
            mode = None
    if mode == INCLUDED or sess.get("cost_status") == "included":
        return {"included": True, "usd": 0.0}
    for field in ("actual_cost_usd", "estimated_cost_usd"):
        val = sess.get(field)
        if isinstance(val, (int, float)) and val > 0:
            return {"included": False, "usd": float(val)}
    tokens = [int(sess.get(k) or 0) for k in (
        "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")]
    if pricing is not None and any(tokens):
        try:
            usage = pricing.CanonicalUsage(
                input_tokens=tokens[0], output_tokens=tokens[1],
                cache_read_tokens=tokens[2], cache_write_tokens=tokens[3])
            res = pricing.estimate_usage_cost(
                model, usage, provider=provider or None, base_url=sess.get("billing_base_url"))
            if res.status == "included":
                return {"included": True, "usd": 0.0}
            if res.amount_usd is not None:
                return {"included": False, "usd": float(res.amount_usd)}
        except Exception:
            pass
    return {"included": False, "usd": None}


def _task_of(title: Any) -> Optional[str]:
    m = _TASK_RE.search(title) if isinstance(title, str) else None
    return m.group(1) if m else None


def _percentile(vals: List[float], pct: float) -> Optional[float]:
    if not vals:
        return None
    vals = sorted(vals)
    return round(vals[int(round(pct * (len(vals) - 1)))], 1)


def _int(value: Any) -> Any:
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _owning_run(candidates: List[Dict[str, Any]], started: float) -> Optional[Dict[str, Any]]:
    best = None
    for r in candidates:
        end = r.get("ended_at") or float("inf")
        if r["started_at"] - _SLACK_S <= started <= end + _SLACK_S:
            if best is None or r["started_at"] > best["started_at"]:
                best = r
    return best


def summarize(
    decisions: Iterable[Dict[str, Any]],
    runs: Iterable[Dict[str, Any]],
    sessions: Iterable[Dict[str, Any]],
    *, days: int = DEFAULT_DAYS, now: Optional[float] = None,
    pricing: Optional[Any] = None,
) -> Dict[str, Any]:
    """Per-tier cost (metered / subscription_included / unknown) and run latency."""
    now = time.time() if now is None else now
    since = now - days * 86400
    runs = [r for r in runs if (r.get("started_at") or 0) >= since]
    runs_by_task: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        runs_by_task.setdefault(r["task_id"], []).append(r)

    tier_for: Dict[tuple, str] = {}
    for d in decisions:
        tid, rid = d.get("task_id"), d.get("run_id")
        if isinstance(tid, str) and tid and isinstance(rid, (int, str)):
            tier_for[(tid, _int(rid))] = tier_of(d)

    tiers: Dict[str, Dict[str, Any]] = {}

    def bucket(tier: str) -> Dict[str, Any]:
        return tiers.setdefault(tier, {
            "runs": 0, "latencies": [], "sessions": 0, "cost_usd": 0.0,
            "cost_unknown_sessions": 0, "subscription_included": {"sessions": 0, "models": {}},
        })

    for run in runs:
        tier = tier_for.get((run["task_id"], run["id"]))
        if tier is None:
            continue
        b = bucket(tier)
        b["runs"] += 1
        if run.get("ended_at") and run["ended_at"] >= run["started_at"]:
            b["latencies"].append(float(run["ended_at"] - run["started_at"]))

    unattributed = 0
    for sess in sessions:
        started = sess.get("started_at") or 0
        if started < since:
            continue
        tid = _task_of(sess.get("title"))
        run = _owning_run(runs_by_task.get(tid or "", []), started)
        tier = tier_for.get((run["task_id"], run["id"])) if run else None
        if tier is None:
            unattributed += 1
            continue
        b = bucket(tier)
        b["sessions"] += 1
        cost = session_cost(sess, pricing)
        if cost["included"]:
            inc = b["subscription_included"]
            inc["sessions"] += 1
            model = sess.get("model") or "unknown"
            inc["models"][model] = inc["models"].get(model, 0) + 1
        elif cost["usd"] is None:
            b["cost_unknown_sessions"] += 1
        else:
            b["cost_usd"] += cost["usd"]

    out: Dict[str, Any] = {}
    for tier, b in sorted(tiers.items()):
        lat = b.pop("latencies")
        b["cost_usd"] = round(b["cost_usd"], 6)
        b["latency_s"] = {
            "n": len(lat), "avg": round(sum(lat) / len(lat), 1) if lat else None,
            "p50": _percentile(lat, 0.5), "p95": _percentile(lat, 0.95),
        }
        out[tier] = b
    return {"days": days, "tiers": out, "unattributed_sessions": unattributed}


def report(
    decisions: Iterable[Dict[str, Any]], days: int = DEFAULT_DAYS,
    read_runs_fn: Callable[[float], List[Dict[str, Any]]] = read_runs,
    read_sessions_fn: Callable[[float], List[Dict[str, Any]]] = read_sessions,
) -> Dict[str, Any]:
    """Read the live sources and summarize; the sidecar's ``GET /tier-cost`` body."""
    now = time.time()
    since = now - days * 86400
    return summarize(decisions, read_runs_fn(since), read_sessions_fn(since), days=days, now=now)
