"""Session-alignment trigger engine (§4 of the alignment-hook research note).

Pure core: given (counters, config, state, clock) it answers which rung of the
threshold ladder, if any, wins right now. No IO, no module state, no clock read
(the caller passes ``now``), no model calls. Persisting ``AlignmentState`` and
acting on the verdict belong to the callers (F7/F8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Mapping, Optional, Tuple

_DEFAULT_PCT: Tuple[float, ...] = (60.0, 85.0)
_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "pct": _DEFAULT_PCT,
    "min_iterations": 12,
    "goal_turn_pct": 70.0,
    "fallback_max_iterations": 90,
    "cooldown_iterations": 10,
    "max_evaluations_per_session": 3,
}
_RESOLVED_KEYS = tuple(_DEFAULTS)
GOAL_RUNG = "goal_turn"


@dataclass(frozen=True)
class Counters:
    session_id: str
    run_id: str
    iteration: int  # api_call_count of this run
    tier: Optional[str] = None
    profile: Optional[str] = None
    budget_used: Optional[int] = None
    budget_max: Optional[int] = None
    max_iterations: Optional[int] = None
    goal_turn: Optional[int] = None  # set only in goal-mode
    goal_max_turns: Optional[int] = None


@dataclass(frozen=True)
class AlignmentState:
    """What survives a restart. ``fired`` holds (session, run, rung) keys."""

    fired: FrozenSet[Tuple[str, str, str]] = frozenset()
    evaluations: int = 0
    last_action_iteration: Optional[int] = None  # last non-continue verdict
    disabled_until: Optional[float] = None  # breaker, compared to ``now``


@dataclass(frozen=True)
class Decision:
    fire: bool
    reason: str
    rung: Optional[str] = None  # "60", "85" or GOAL_RUNG
    key: Optional[Tuple[str, str, str]] = None
    pct: Optional[float] = None
    # Lower rungs crossed in the same step: the caller records them as fired
    # too, so a jump from 50% to 90% never replays the 60% rung afterwards.
    superseded: Tuple[Tuple[str, str, str], ...] = field(default_factory=tuple)


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    return value if isinstance(value, Mapping) else {}


def resolve(config: Mapping[str, Any], tier: Optional[str], profile: Optional[str]) -> Dict[str, Any]:
    """Effective knobs, precedence profile > tier > global > built-in default."""
    trigger = _section(config, "trigger")
    overrides = _section(config, "overrides")
    layers = (
        {"enabled": config.get("enabled", True)},
        trigger,
        _section(_section(overrides, "tiers"), tier) if tier else {},
        _section(_section(overrides, "profiles"), profile) if profile else {},
    )
    out = dict(_DEFAULTS)
    for layer in layers:
        for key in _RESOLVED_KEYS:
            if key in layer:
                out[key] = layer[key]
    out["pct"] = tuple(sorted(float(p) for p in out["pct"]))
    return out


def _rung_name(pct: float) -> str:
    return str(int(pct)) if float(pct).is_integer() else str(pct)


def _ratio(counters: Counters, knobs: Mapping[str, Any]) -> Tuple[int, int]:
    """(used, max) for the x% test: budget first, then api calls over a ceiling."""
    if counters.budget_max:
        return int(counters.budget_used or 0), int(counters.budget_max)
    return counters.iteration, int(counters.max_iterations or knobs["fallback_max_iterations"])


def evaluate(
    counters: Counters,
    config: Mapping[str, Any],
    state: AlignmentState,
    now: float,
) -> Decision:
    """Pick the winning rung, or explain why none fires."""
    knobs = resolve(config, counters.tier, counters.profile)
    if not config.get("enabled", False) or not knobs["enabled"]:
        return Decision(False, "disabled")
    if state.disabled_until is not None and now < state.disabled_until:
        return Decision(False, "breaker_open")
    if state.evaluations >= knobs["max_evaluations_per_session"]:
        return Decision(False, "max_evaluations")
    if counters.iteration < knobs["min_iterations"]:
        return Decision(False, "below_min_iterations")
    last = state.last_action_iteration
    if last is not None and counters.iteration - last < knobs["cooldown_iterations"]:
        return Decision(False, "cooldown")

    def key(rung: str) -> Tuple[str, str, str]:
        return (counters.session_id, counters.run_id, rung)

    used, ceiling = _ratio(counters, knobs)
    pct_now = 100.0 * used / ceiling if ceiling > 0 else 0.0
    crossed = [
        p for p in knobs["pct"]
        if pct_now >= p and key(_rung_name(p)) not in state.fired
    ]
    if crossed:
        win = crossed[-1]
        return Decision(
            True, "ladder", _rung_name(win), key(_rung_name(win)), pct_now,
            tuple(key(_rung_name(p)) for p in crossed[:-1]),
        )

    if counters.goal_turn is not None and counters.goal_max_turns:
        goal_pct = 100.0 * counters.goal_turn / counters.goal_max_turns
        if goal_pct >= knobs["goal_turn_pct"] and key(GOAL_RUNG) not in state.fired:
            return Decision(True, "goal_turn", GOAL_RUNG, key(GOAL_RUNG), goal_pct)
    return Decision(False, "no_rung")


def record(state: AlignmentState, decision: Decision, *, iteration: int, acted: bool) -> AlignmentState:
    """State after a fired decision was evaluated (shadow firings count too).

    ``acted`` is False for a ``continue`` verdict, which must not start a cooldown.
    """
    if not decision.fire:
        return state
    return AlignmentState(
        fired=state.fired | {decision.key, *decision.superseded},
        evaluations=state.evaluations + 1,
        last_action_iteration=iteration if acted else state.last_action_iteration,
        disabled_until=state.disabled_until,
    )
