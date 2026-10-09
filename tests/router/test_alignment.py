"""Trigger engine: ladder, hysteresis, cooldown, resolution, goal turn, purity."""

import ast
import inspect

from router import alignment as mod
from router.alignment import AlignmentState, Counters, Decision, evaluate, record, resolve

CFG = {"enabled": True, "trigger": {"pct": [60, 85], "min_iterations": 12}}


def c(it, **kw):
    kw.setdefault("max_iterations", 100)
    return Counters("s", "r1", it, **kw)


def test_master_switch_off_and_missing():
    assert evaluate(c(99), {}, AlignmentState(), 0).reason == "disabled"
    assert evaluate(c(99), {"enabled": False}, AlignmentState(), 0).reason == "disabled"


def test_ladder_first_and_second_rung():
    d = evaluate(c(60), CFG, AlignmentState(), 0)
    assert d.fire and d.rung == "60" and d.key == ("s", "r1", "60") and d.pct == 60.0
    st = record(AlignmentState(), d, iteration=60, acted=True)
    assert evaluate(c(65), CFG, st, 0).reason == "cooldown"
    d2 = evaluate(c(85), CFG, st, 0)
    assert d2.fire and d2.rung == "85"


def test_below_threshold_and_min_iterations():
    assert evaluate(c(30), CFG, AlignmentState(), 0).reason == "no_rung"
    cfg = {"enabled": True, "trigger": {"pct": [10], "min_iterations": 12}}
    assert evaluate(c(11), cfg, AlignmentState(), 0).reason == "below_min_iterations"


def test_jump_fires_highest_and_supersedes_lower():
    d = evaluate(c(90), CFG, AlignmentState(), 0)
    assert d.rung == "85" and d.superseded == (("s", "r1", "60"),)
    st = record(AlignmentState(), d, iteration=90, acted=False)
    assert st.last_action_iteration is None
    assert evaluate(c(99), CFG, st, 0).reason == "no_rung"


def test_hysteresis_each_rung_once_and_new_run_refires():
    st = record(AlignmentState(), evaluate(c(60), CFG, AlignmentState(), 0), iteration=60, acted=False)
    assert evaluate(c(61), CFG, st, 0).reason == "no_rung"
    assert evaluate(Counters("s", "r2", 60, max_iterations=100), CFG, st, 0).fire


def test_restart_does_not_refire_rung():
    st = record(AlignmentState(), evaluate(c(60), CFG, AlignmentState(), 0), iteration=60, acted=True)
    reloaded = AlignmentState(
        fired=frozenset(st.fired), evaluations=st.evaluations,
        last_action_iteration=st.last_action_iteration,
    )
    assert reloaded == st
    assert not evaluate(c(60), CFG, reloaded, 0).fire
    # Even with the cooldown long gone, the persisted key blocks the rung.
    assert evaluate(c(80), CFG, reloaded, 0).reason == "no_rung"


def test_cooldown_expires():
    st = AlignmentState(last_action_iteration=58)
    cfg = {"enabled": True, "trigger": {"pct": [60], "cooldown_iterations": 10}}
    assert evaluate(c(65), cfg, st, 0).reason == "cooldown"
    assert evaluate(c(67), cfg, st, 0).reason == "cooldown"
    assert evaluate(c(68), cfg, st, 0).fire


def test_max_evaluations():
    assert evaluate(c(90), CFG, AlignmentState(evaluations=3), 0).reason == "max_evaluations"


def test_breaker_uses_injected_clock():
    st = AlignmentState(disabled_until=100.0)
    assert evaluate(c(90), CFG, st, 99.0).reason == "breaker_open"
    assert evaluate(c(90), CFG, st, 100.0).fire


def test_t1_disabled_by_tier_override():
    cfg = {**CFG, "overrides": {"tiers": {"T1": {"enabled": False}}}}
    assert evaluate(c(90, tier="T1"), cfg, AlignmentState(), 0).reason == "disabled"
    assert evaluate(c(90, tier="T2"), cfg, AlignmentState(), 0).fire
    assert evaluate(c(90), cfg, AlignmentState(), 0).fire


def test_resolution_profile_beats_tier_beats_global():
    cfg = {
        "enabled": True,
        "trigger": {"pct": [60, 85]},
        "overrides": {
            "tiers": {"T4": {"pct": [50, 80]}},
            "profiles": {"coder": {"pct": [70]}, "other": {"enabled": False}},
        },
    }
    assert resolve(cfg, None, None)["pct"] == (60.0, 85.0)
    assert resolve(cfg, "T4", None)["pct"] == (50.0, 80.0)
    assert resolve(cfg, "T4", "coder")["pct"] == (70.0,)
    assert resolve(cfg, "T9", "coder")["pct"] == (70.0,)
    assert resolve(cfg, None, "other")["enabled"] is False
    assert resolve({"trigger": "junk", "overrides": []}, "T1", "x")["pct"] == (60.0, 85.0)


def test_fractional_rung_name():
    cfg = {"enabled": True, "trigger": {"pct": [62.5], "min_iterations": 1}}
    assert evaluate(c(70), cfg, AlignmentState(), 0).rung == "62.5"


def test_denominator_precedence():
    cfg = {"enabled": True, "trigger": {"pct": [50], "min_iterations": 1, "fallback_max_iterations": 20}}
    # budget wins over api_call_count
    d = evaluate(Counters("s", "r", 5, budget_used=50, budget_max=100, max_iterations=5), cfg, AlignmentState(), 0)
    assert d.fire and d.pct == 50.0
    d = evaluate(Counters("s", "r", 5, budget_max=100), cfg, AlignmentState(), 0)
    assert not d.fire
    # max_iterations then fallback
    assert evaluate(Counters("s", "r", 10, max_iterations=20), cfg, AlignmentState(), 0).fire
    assert evaluate(Counters("s", "r", 10), cfg, AlignmentState(), 0).fire
    assert not evaluate(Counters("s", "r", 9), cfg, AlignmentState(), 0).fire
    zero = {"enabled": True, "trigger": {"pct": [1], "min_iterations": 0, "fallback_max_iterations": 0}}
    assert evaluate(Counters("s", "r", 5), zero, AlignmentState(), 0).reason == "no_rung"


def test_goal_turn_trigger_once():
    cfg = {"enabled": True, "trigger": {"pct": [95], "goal_turn_pct": 70, "min_iterations": 1}}
    g = lambda t: c(5, goal_turn=t, goal_max_turns=10)
    assert evaluate(g(6), cfg, AlignmentState(), 0).reason == "no_rung"
    d = evaluate(g(7), cfg, AlignmentState(), 0)
    assert d.fire and d.reason == "goal_turn" and d.rung == "goal_turn" and d.pct == 70.0
    st = record(AlignmentState(), d, iteration=5, acted=False)
    assert not evaluate(g(9), cfg, st, 0).fire
    assert evaluate(c(5, goal_turn=3), cfg, AlignmentState(), 0).reason == "no_rung"


def test_record_ignores_non_fire():
    st = AlignmentState()
    assert record(st, Decision(False, "x"), iteration=1, acted=True) is st
    d = evaluate(c(90), CFG, AlignmentState(disabled_until=5.0), 0)
    assert d.reason == "breaker_open"


def test_record_preserves_breaker_and_counts():
    st = AlignmentState(disabled_until=9.0, last_action_iteration=3)
    d = Decision(True, "ladder", "60", ("s", "r", "60"))
    out = record(st, d, iteration=70, acted=False)
    assert out.disabled_until == 9.0 and out.last_action_iteration == 3 and out.evaluations == 1


def test_module_is_pure():
    tree = ast.parse(inspect.getsource(mod))
    called, imported = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            called.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
        elif isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not called & {"now", "utcnow", "today", "monotonic", "time", "fromtimestamp", "open", "print"}
    assert imported <= {"__future__", "dataclasses", "typing"}
