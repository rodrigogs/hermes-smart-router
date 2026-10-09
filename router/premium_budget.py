"""Copilot premium-request budget, modelled from the route trace (shadow mode).

Hermes sends ``x-initiator: user`` on the first call of every user turn, and Copilot
bills that as one premium request times the model multiplier. A kanban worker and a
``delegate_profile`` child (``hermes -p X chat -q``) each start a fresh turn, so each
routed decision with source ``kanban`` or ``delegate`` costs >= 1 premium request when
the elo that runs is a Copilot elo. Tool calls inside the turn do not bill.

Nothing here changes routing: it only counts. ``report`` is the body of
``GET /premium-budget`` and never raises.

Sources (read 2026-10-09):
* https://docs.github.com/en/copilot/concepts/billing/copilot-requests  (allowances)
* https://docs.github.com/en/copilot/reference/copilot-billing/request-based-billing-legacy/
  model-multipliers-for-annual-plans  (multipliers)
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Optional

try:
    from .decision_log import source_of
    from .outcomes import tier_of
except ImportError:  # pragma: no cover - flat layout used by the test harness
    from router.decision_log import source_of
    from router.outcomes import tier_of

PROVIDER = "copilot"
# Per worker or delegated child: one user-initiated turn.
BILLED_SOURCES = ("kanban", "delegate")
REQUESTS_PER_TURN = 1
DEFAULT_DAYS = 7

#: Monthly allowance of the LEGACY request-based plans. Resets on the 1st, 00:00 UTC;
#: extra requests cost $0.04 each. Usage-based plans (after 2026-06-01) have no
#: multipliers and no request allowance, so this budget applies only to legacy annual
#: Pro / Pro+ subscribers. Which plan this install holds is NOT verified.
QUOTA = {"pro": 300, "pro_plus": 1500, "overage_usd_per_request": 0.04}

#: Published multipliers for annual plans. A model absent here is UNPUBLISHED: it is
#: counted at 1.0 and reported under ``unpublished_multiplier`` rather than guessed.
MULTIPLIERS: Dict[str, float] = {"gpt-5.4": 6.0}
DEFAULT_MULTIPLIER = 1.0


def _head(entry: Dict[str, Any]) -> tuple:
    out = entry.get("output") if isinstance(entry.get("output"), dict) else {}
    return (out.get("attempted_model") or out.get("model") or "",
            out.get("attempted_provider") or out.get("provider") or "")


def _copilot_hops(entry: Dict[str, Any]) -> list:
    out = entry.get("output") if isinstance(entry.get("output"), dict) else {}
    hops = [{"model": out.get("model"), "provider": out.get("provider")}]
    hops += [h for h in (out.get("fallback") or []) if isinstance(h, dict)]
    return [h.get("model") for h in hops if h.get("provider") == PROVIDER and h.get("model")]


def _weight(model: str) -> float:
    return MULTIPLIERS.get(model, DEFAULT_MULTIPLIER) * REQUESTS_PER_TURN


def summarize(
    decisions: Iterable[Dict[str, Any]], *, days: int = DEFAULT_DAYS,
    now: Optional[float] = None,
) -> Dict[str, Any]:
    """Premium requests per tier over the last ``days`` days.

    ``premium_requests`` counts decisions whose head elo is on Copilot (what is billed
    when the primary serves). ``exposure_requests`` also counts decisions that merely
    list a Copilot hop in the fallback chain (what is billed if the chain falls
    through to it) — an upper bound, kept separate so it is never read as spend.
    """
    now = time.time() if now is None else now
    since = now - days * 86400
    tiers: Dict[str, Dict[str, Any]] = {}
    unpublished = set()
    total_decisions = 0
    for d in decisions:
        ts = d.get("ts") if isinstance(d, dict) else None
        if not isinstance(ts, (int, float)) or ts < since:
            continue
        total_decisions += 1
        src = source_of(d)
        if src == "unknown" and d.get("task_id"):
            src = "kanban"  # pre-`source` traces: a task_id means a kanban dispatch
        if src not in BILLED_SOURCES:
            continue
        tier = tier_of(d)
        b = tiers.setdefault(tier, {
            "turns": 0, "premium_requests": 0.0, "exposure_requests": 0.0, "models": {}})
        b["turns"] += 1
        model, provider = _head(d)
        if provider == PROVIDER:
            w = _weight(model)
            b["premium_requests"] += w
            b["models"][model] = b["models"].get(model, 0) + 1
            if model not in MULTIPLIERS:
                unpublished.add(model)
        hops = _copilot_hops(d)
        if hops:
            b["exposure_requests"] += max(_weight(m) for m in hops)
            unpublished.update(m for m in hops if m not in MULTIPLIERS)
    total = round(sum(b["premium_requests"] for b in tiers.values()), 2)
    per_month = round(total * 30 / days, 1) if days else None
    return {
        "days": days, "mode": "shadow", "decisions_in_window": total_decisions,
        "tiers": {t: {**b, "premium_requests": round(b["premium_requests"], 2),
                      "exposure_requests": round(b["exposure_requests"], 2)}
                  for t, b in sorted(tiers.items())},
        "total_premium_requests": total,
        "projected_per_30_days": per_month,
        "quota": QUOTA,
        "multipliers": dict(MULTIPLIERS),
        "unpublished_multiplier": sorted(unpublished),
    }


def report(decisions: Iterable[Dict[str, Any]], days: int = DEFAULT_DAYS) -> Dict[str, Any]:
    return summarize(decisions, days=days)
