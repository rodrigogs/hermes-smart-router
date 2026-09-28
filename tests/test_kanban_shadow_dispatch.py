"""Tests for the kanban-dispatch SHADOW routing (pre_kanban_dispatch hook).

The hook records what the capability router WOULD choose for each dispatched
card — title+body, no classifier, profile-constrained — without writing the
model/provider field. The exit gate (shadow_gate_rate) measures how often
Stage 0 alone falls through on REAL cards.
"""

import json
import sys
import time
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "delegate_profile_plugin", REPO_ROOT / "__init__.py"
)
assert _spec is not None and _spec.loader is not None, "could not load plugin spec"
dp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dp)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trace_home(tmp_path, monkeypatch):
    """Point the durable trace (and breaker state) at a throwaway HERMES_HOME."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _card(title, body="", model_override=None, provider_override=None):
    return types.SimpleNamespace(
        title=title,
        body=body,
        model_override=model_override,
        provider_override=provider_override,
    )


def _read_trace() -> list:
    """All durable trace entries, parsed from disk."""
    from router.durable_decision_log import routes_path
    entries = []
    p = routes_path()
    if not p.exists():
        return entries
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            entries.append(json.loads(line))
    return entries


# ---------------------------------------------------------------------------
# _kanban_task_text — title + body is the routing input
# ---------------------------------------------------------------------------

def test_kanban_task_text_joins_title_and_body():
    text = dp._kanban_task_text(_card("Fix the bug", "The crash happens on save"))
    assert text == "Fix the bug\n\nThe crash happens on save"


def test_kanban_task_text_title_only_when_no_body():
    assert dp._kanban_task_text(_card("Fix the bug")) == "Fix the bug"


def test_kanban_task_text_duck_types_missing_fields():
    assert dp._kanban_task_text(types.SimpleNamespace(title="t")) == "t"


# ---------------------------------------------------------------------------
# _KanbanShadowLog — the shadow stamp and the profile constraint
# ---------------------------------------------------------------------------

def test_shadow_log_stamps_the_persisted_entry(trace_home):
    log = dp._KanbanShadowLog()
    log.record("hard_rule", {"profile": "coder", "model": "T4"},
               task_preview="task")
    entry = _read_trace()
    assert len(entry) == 1
    assert entry[0]["shadow"] is True
    assert entry[0]["cause"] == "hard_rule"


def test_shadow_log_keeps_a_matching_profile_decision_untouched(trace_home):
    log = dp._KanbanShadowLog(allowed_profile="coder")
    log.record("has_code_rule", {"profile": "coder", "model": "glm-4.7"},
               task_preview="rename a var")
    entry = _read_trace()[0]
    assert entry["cause"] == "has_code_rule"
    assert entry["output"]["model"] == "glm-4.7"


def test_shadow_log_keeps_the_model_half_when_the_role_is_out_of_scope(trace_home):
    """The role axis is not this path's to move; the model axis is.

    Until 2026-08-26 this dropped the model half too, and 135 of 158 measured
    decisions therefore contributed nothing.
    """
    log = dp._KanbanShadowLog(allowed_profile="coder")
    log.record(
        "review-request", {"profile": "reviewer", "model": "gpt-5.5"},
        task_preview="Review this PR",
        steps=[{"stage": "rules", "in": {}, "out": {"profile": "reviewer"},
                "cause": "keyword_match"}],
    )
    entry = _read_trace()[0]
    assert entry["cause"] == "role_out_of_scope"
    # The model half is what this path can apply, so it stays.
    assert entry["output"]["model"] == "gpt-5.5"
    # And the role the policy wanted stays too: the operator can tell "chose this
    # model" from "also wanted another role, which this path never moves".
    assert entry["output"]["profile"] == "reviewer"


def test_shadow_log_without_assignee_constrains_nothing(trace_home):
    log = dp._KanbanShadowLog(allowed_profile=None)
    log.record("keyword_match", {"profile": "reviewer", "model": "x"})
    assert _read_trace()[0]["cause"] == "keyword_match"


# ---------------------------------------------------------------------------
# _kanban_role_out_of_scope — the ONE role question
# ---------------------------------------------------------------------------

def test_role_in_scope_when_the_decision_names_the_same_role():
    assert dp._kanban_role_out_of_scope({"profile": "coder"}, "coder") is False


def test_role_out_of_scope_when_the_decision_names_another_role():
    assert dp._kanban_role_out_of_scope({"profile": "reviewer"}, "coder") is True


def test_role_in_scope_when_the_caller_fixed_no_role():
    assert dp._kanban_role_out_of_scope({"profile": "reviewer"}, None) is False


def test_role_in_scope_when_the_decision_names_no_role():
    assert dp._kanban_role_out_of_scope({"model": "glm-4.7"}, "coder") is False


# ---------------------------------------------------------------------------
# _kanban_live_override — what a live hook may hand the dispatcher
# ---------------------------------------------------------------------------

def test_live_override_uses_the_plan_head_when_a_chain_is_attached():
    decision = {
        "profile": "coder", "model": "glm-5.3", "provider": "zai",
        "chain": [{"model": "gpt-5.6-luna", "provider": "openai-codex"}],
    }
    assert dp._kanban_live_override(decision, "coder") == {
        "model": "gpt-5.6-luna", "provider": "openai-codex",
    }


def test_live_override_refuses_a_chain_head_without_provider():
    decision = {
        "profile": "coder", "model": "glm-5.3", "provider": "zai",
        "chain": [{"model": "gpt-5.6-luna"}],
    }
    assert dp._kanban_live_override(decision, "coder") is None


def test_live_override_falls_back_to_the_declared_pair_without_a_chain():
    decision = {"profile": "coder", "model": "glm-4.7", "provider": "zai"}
    assert dp._kanban_live_override(decision, "coder") == {
        "model": "glm-4.7", "provider": "zai",
    }


def test_live_override_refuses_a_declared_model_without_provider():
    decision = {"profile": "coder", "model": "glm-4.7"}
    assert dp._kanban_live_override(decision, "coder") is None


def test_live_override_refuses_a_decision_with_no_model():
    decision = {"profile": "coder", "action": "classify"}
    assert dp._kanban_live_override(decision, "coder") is None


def test_live_override_hands_the_model_over_when_the_role_is_out_of_scope():
    """The role was the operator's to choose and stays theirs; the model is the
    one axis this hook can apply, so a role mismatch no longer voids it."""
    decision = {
        "profile": "reviewer", "model": "gpt-5.5", "provider": "openai-codex",
    }
    assert dp._kanban_live_override(decision, "coder") == {
        "model": "gpt-5.5", "provider": "openai-codex",
    }


def test_live_override_without_an_assignee_constrains_nothing():
    decision = {
        "profile": "reviewer", "model": "gpt-5.5", "provider": "openai-codex",
    }
    assert dp._kanban_live_override(decision, None) == {
        "model": "gpt-5.5", "provider": "openai-codex",
    }


# ---------------------------------------------------------------------------
# _read_kanban_task — board DB read, guarded like every hermes_cli access
# ---------------------------------------------------------------------------

def test_read_kanban_task_degrades_without_hermes_cli(monkeypatch):
    monkeypatch.setitem(sys.modules, "hermes_cli", None)
    assert dp._read_kanban_task("t1", None) is None


def test_read_kanban_task_returns_the_card(monkeypatch):
    card = _card("Fix the bug", "body text")
    fake_cli = types.ModuleType("hermes_cli")
    fake_kb = types.ModuleType("hermes_cli.kanban_db")

    class _Conn:
        def close(self):
            pass

    fake_kb.connect = lambda board=None: _Conn()
    fake_kb.get_task = lambda conn, task_id: card
    fake_cli.kanban_db = fake_kb
    monkeypatch.setitem(sys.modules, "hermes_cli", fake_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_db", fake_kb)
    assert dp._read_kanban_task("t1", "board") is card


def test_read_kanban_task_empty_id_short_circuits():
    assert dp._read_kanban_task("", None) is None


def test_read_kanban_task_survives_post_split_core(monkeypatch):
    """Post-split ``kanban_db`` without ``connect``: the sub-module owns it."""
    card = _card("Fix the bug", "body text")
    fake_cli = types.ModuleType("hermes_cli")
    fake_kb = types.ModuleType("hermes_cli.kanban_db")
    fake_connect = types.ModuleType("hermes_cli.kanban_db_connect")

    class _Conn:
        def close(self):
            pass

    fake_kb.connect = None  # compat map dropped
    fake_kb.get_task = lambda conn, task_id: card
    fake_connect.connect = lambda board=None: _Conn()
    fake_cli.kanban_db = fake_kb
    monkeypatch.setitem(sys.modules, "hermes_cli", fake_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_db", fake_kb)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_db_connect", fake_connect)
    assert dp._read_kanban_task("t1", "board") is card


def test_read_kanban_task_returns_none_when_the_board_read_raises(monkeypatch):
    fake_cli = types.ModuleType("hermes_cli")
    fake_kb = types.ModuleType("hermes_cli.kanban_db")

    class _Conn:
        def close(self):
            pass

    def _boom(conn, task_id):
        raise RuntimeError("db locked")

    fake_kb.connect = lambda board=None: _Conn()
    fake_kb.get_task = _boom
    fake_cli.kanban_db = fake_kb
    monkeypatch.setitem(sys.modules, "hermes_cli", fake_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_db", fake_kb)
    assert dp._read_kanban_task("t1", "board") is None


# ---------------------------------------------------------------------------
# _on_pre_kanban_dispatch — shadow consumer
# ---------------------------------------------------------------------------

def test_shadow_hook_records_the_decision_and_never_writes_the_field(
    trace_home, monkeypatch,
):
    """Happy path: card routes through Stage 0, trace gets a shadow entry."""
    monkeypatch.setattr(
        dp, "_read_kanban_task",
        lambda task_id, board: _card(
            "Rename getCwd in src/utils.py", "small mechanical change",
        ),
    )
    result = dp._on_pre_kanban_dispatch(
        task_id="t1", profile_name="trama-engineer", board="default",
        assignee="coder", run_id=1,
    )
    # SHADOW: the hook never returns a model/provider dict.
    assert result is None
    entries = _read_trace()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["shadow"] is True
    assert entry["task"] == "Rename getCwd in src/utils.py\n\nsmall mechanical change"
    # The live policy routes this card deterministically (Stage 0, no classifier).
    assert entry["cause"] not in ("", None)
    assert "steps" in entry


def test_shadow_hook_records_the_model_half_of_a_reviewer_rule(trace_home, monkeypatch):
    """review-request wants a reviewer; the card stays a coder card and still
    gets the model the policy chose for that task shape."""
    monkeypatch.setattr(
        dp, "_read_kanban_task",
        lambda task_id, board: _card("Review this PR for security issues"),
    )
    dp._on_pre_kanban_dispatch(
        task_id="t2", profile_name="x", board="default",
        assignee="coder", run_id=2,
    )
    entry = _read_trace()[0]
    assert entry["cause"] == "role_out_of_scope"
    assert entry["output"]["profile"] == "reviewer"
    assert entry["output"]["model"]


def test_shadow_hook_is_silent_when_router_is_disabled(trace_home, monkeypatch):
    monkeypatch.setattr(
        dp, "_read_kanban_task",
        lambda task_id, board: _card("Rename getCwd in src/utils.py"),
    )
    monkeypatch.setattr(dp, "_load_router_config", lambda: {"enabled": False})
    assert dp._on_pre_kanban_dispatch(task_id="t3", assignee="coder") is None
    assert _read_trace() == []


def test_live_hook_routes_records_and_returns_the_decision(trace_home, monkeypatch):
    """shadow.enabled: false is LIVE mode, not off: same routing and trace,
    but the hook now returns the model decision for the dispatcher to apply."""
    monkeypatch.setattr(
        dp, "_read_kanban_task",
        lambda task_id, board: _card(
            "Rename getCwd in src/utils.py", "small mechanical change",
        ),
    )
    monkeypatch.setattr(
        dp, "_load_router_config",
        lambda: {"enabled": True, "shadow": {"enabled": False}},
    )
    result = dp._on_pre_kanban_dispatch(
        task_id="t4", profile_name="trama-engineer", board="default",
        assignee="coder", run_id=4,
    )
    # Live: both fields set — the head of the (declared-order) chain, which for
    # a fail-safe decision is the declared pair itself.
    assert result == {"model": "deepseek-v4-pro", "provider": "deepseek"}
    entries = _read_trace()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["shadow"] is False
    assert entry["cause"] == "fail_safe_strong"


def test_live_hook_applies_the_model_of_a_rule_that_named_another_role(trace_home, monkeypatch):
    """LIVE mode and the trace agree, as they must: the model half drives the
    dispatch and the trace says the role half was out of scope."""
    monkeypatch.setattr(
        dp, "_read_kanban_task",
        lambda task_id, board: _card("Review this PR for security issues"),
    )
    monkeypatch.setattr(
        dp, "_load_router_config",
        lambda: {
            "enabled": True,
            "shadow": {"enabled": False},
            "rules": [
                {"id": "review-request", "status": "stable",
                 "when": {"keywords": {"contains": "review"}},
                 "then": {"profile": "reviewer", "model": "T1"}},
            ],
            "tiers": {"T1": {"model": "glm-4.7", "provider": "zai"}},
        },
    )
    result = dp._on_pre_kanban_dispatch(
        task_id="t6", profile_name="x", board="default",
        assignee="coder", run_id=6,
    )
    assert result == {"model": "glm-4.7", "provider": "zai"}
    entry = _read_trace()[0]
    assert entry["shadow"] is False
    assert entry["cause"] == "role_out_of_scope"
    assert entry["output"]["profile"] == "reviewer"
    assert entry["output"]["model"] == "glm-4.7"


def test_live_entries_never_count_in_the_gate(trace_home, monkeypatch):
    """shadow_gate_rate counts ``shadow is True`` only — a live-only trace is
    an empty sample even when the live decision fell through Stage 0."""
    monkeypatch.setattr(
        dp, "_read_kanban_task",
        lambda task_id, board: _card("Some ambiguous task with no rule"),
    )
    monkeypatch.setattr(
        dp, "_load_router_config",
        lambda: {"enabled": True, "shadow": {"enabled": False}},
    )
    result = dp._on_pre_kanban_dispatch(task_id="t7", assignee="coder")
    assert result is not None  # live: the fail-safe decision was returned
    entries = _read_trace()
    assert len(entries) == 1
    assert entries[0]["shadow"] is False
    # The only entry is live — the sample is empty, so the gate reads unmet.
    assert dp.shadow_gate_rate() is None
    assert dp._shadow_gate_ok() is False


def test_shadow_hook_never_breaks_dispatch_when_routing_raises(trace_home, monkeypatch):
    """A broken router call is a no-op — byte-identical to no subscriber."""
    import router.adapter as adapter_mod

    def _boom(*args, **kwargs):
        raise RuntimeError("route exploded")

    monkeypatch.setattr(
        dp, "_read_kanban_task",
        lambda task_id, board: _card("Rename getCwd in src/utils.py"),
    )
    monkeypatch.setattr(adapter_mod, "route", _boom)
    assert dp._on_pre_kanban_dispatch(task_id="t9", assignee="coder") is None
    assert _read_trace() == []


def test_shadow_hook_is_silent_when_the_config_load_raises(trace_home, monkeypatch):
    def _boom():
        raise RuntimeError("yaml broken")

    monkeypatch.setattr(dp, "_load_router_config", _boom)
    assert dp._on_pre_kanban_dispatch(task_id="t8", assignee="coder") is None
    assert _read_trace() == []


def test_shadow_hook_never_breaks_dispatch_on_failure(trace_home, monkeypatch):
    """A broken card read is a no-op, byte-identical to having no subscriber."""
    monkeypatch.setattr(dp, "_read_kanban_task", lambda task_id, board: None)
    assert dp._on_pre_kanban_dispatch(task_id="t5", assignee="coder") is None
    assert _read_trace() == []


# ---------------------------------------------------------------------------
# The exit gate — no_classifier + fallthrough on REAL cards
# ---------------------------------------------------------------------------

def _shadow_entry(cause, reason=None, shadow=True):
    steps = []
    if reason is not None:
        steps.append({"stage": "fail_safe", "in": {"reason": reason},
                      "out": {}, "cause": "fail_safe_strong"})
    entry = {"ts": 0.0, "cause": cause, "output": {}, "rule_id": None,
             "task": "", "steps": steps}
    if shadow:
        entry["shadow"] = True
    return entry


def test_gate_empty_trace_is_not_met(trace_home):
    assert dp.shadow_gate_rate() is None
    assert dp._shadow_gate_ok() is False


def test_gate_counts_only_shadow_entries(trace_home):
    from router.durable_decision_log import routes_path
    path = routes_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        # A delegate-tool entry (no shadow key) must not be counted.
        json.dumps(_shadow_entry("fail_safe_strong", "no_classifier", shadow=False)),
        json.dumps(_shadow_entry("has_code_rule", shadow=True)),
        json.dumps(_shadow_entry("fail_safe_strong", "no_classifier", shadow=True)),
        json.dumps(_shadow_entry("fail_safe_strong", "fallthrough", shadow=True)),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # 2 of 3 shadow entries fell through.
    assert dp.shadow_gate_rate() == pytest.approx(2 / 3)
    assert dp._shadow_gate_ok() is False


def test_gate_zero_fallthrough_is_met(trace_home):
    from router.durable_decision_log import routes_path
    path = routes_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(_shadow_entry("has_code_rule", shadow=True)),
        json.dumps(_shadow_entry("keyword_match", shadow=True)),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert dp.shadow_gate_rate() == 0.0
    assert dp._shadow_gate_ok() is True


def test_gate_boundary_at_the_agreed_limit(trace_home):
    from router.durable_decision_log import routes_path
    path = routes_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # 1 of 5 shadow entries fell through == exactly the agreed 0.20 limit.
    lines = [
        json.dumps(_shadow_entry("fail_safe_strong", "no_classifier", shadow=True)),
    ] + [
        json.dumps(_shadow_entry("has_code_rule", shadow=True)) for _ in range(4)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert dp.shadow_gate_rate() == pytest.approx(0.20)
    assert dp._shadow_gate_ok() is True


def test_gate_role_out_of_scope_still_counts_as_fallthrough(trace_home):
    """A rewritten cause never hides a fallthrough: the steps keep the truth."""
    from router.durable_decision_log import routes_path
    path = routes_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = _shadow_entry("role_out_of_scope", "no_classifier", shadow=True)
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    assert dp.shadow_gate_rate() == 1.0


def test_gate_profile_ignored_still_counts_as_fallthrough(trace_home):
    """Traces written before 2026-08-26 carry the retired cause and still count."""
    from router.durable_decision_log import routes_path
    path = routes_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = _shadow_entry("profile_ignored", "no_classifier", shadow=True)
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    assert dp.shadow_gate_rate() == 1.0


def test_gate_ignores_steps_that_are_not_a_fail_safe_fallthrough(trace_home):
    """A step that is not a fail_safe stage (or not a dict) is not a fallthrough."""
    from router.durable_decision_log import routes_path
    path = routes_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = _shadow_entry("has_code_rule", shadow=True)
    entry["steps"] = [{"stage": "rules", "in": {}, "out": {}, "cause": "hard_rule"}]
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    assert dp.shadow_gate_rate() == 0.0


def test_gate_ok_honors_an_explicit_rate():
    assert dp._shadow_gate_ok(0.0) is True
    assert dp._shadow_gate_ok(0.5) is False


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_register_subscribes_the_pre_kanban_dispatch_hook(monkeypatch):
    class Ctx:
        def __init__(self):
            self.tools = []
            self.hooks = []

        def dispatch_tool(self, name, args):
            return "{}:{}".format(name, args["goal"])

        def register_tool(self, **kwargs):
            self.tools.append(kwargs)

        def register_hook(self, *args, **kwargs):
            self.hooks.append((args, kwargs))

    monkeypatch.setattr(dp, "_get_active_profile_name", lambda: "parent")
    ctx = Ctx()
    dp.register(ctx)
    names = [args[0] for args, _ in ctx.hooks]
    assert "pre_kanban_dispatch" in names
    assert "post_tool_call" in names

def test_a_card_of_any_role_gets_the_model_the_policy_chose(trace_home, monkeypatch):
    """The measured regression, as a test.

    On 2026-08-26 every card of this board ran as ``trama-engineer`` while every
    rule of the shipped policy named ``coder`` or ``reviewer``: 135 of 158
    decisions were recorded as refused and contributed nothing. A role nobody
    wrote a rule for must still get the model its task shape earned.
    """
    monkeypatch.setattr(
        dp, "_read_kanban_task",
        lambda task_id, board: _card("Rename getCwd in src/utils.py"),
    )
    monkeypatch.setattr(
        dp, "_load_router_config",
        lambda: {
            "enabled": True,
            "shadow": {"enabled": False},
            "rules": [
                {"id": "standard-implementation", "status": "stable",
                 "when": {"has_code": {"eq": True}},
                 "then": {"profile": "coder", "model": "T2"}},
            ],
            "tiers": {"T2": {"model": "glm-5.3", "provider": "zai"}},
        },
    )
    result = dp._on_pre_kanban_dispatch(
        task_id="t99", profile_name="x", board="capability-router",
        assignee="trama-engineer", run_id=99,
    )
    assert result == {"model": "glm-5.3", "provider": "zai"}
    assert _read_trace()[0]["cause"] == "role_out_of_scope"


def test_a_role_scoped_rule_does_not_fire_for_another_role(trace_home, monkeypatch):
    """The protection the old veto gave, kept where it belongs: on the input side.

    A reviewer-tuned row that buys the strongest tier must not fire for a card
    that is not a reviewer. It says so in ``when``, and the row is simply not the
    one that matches.
    """
    monkeypatch.setattr(
        dp, "_read_kanban_task",
        lambda task_id, board: _card("Please audit this diff for security holes"),
    )
    monkeypatch.setattr(
        dp, "_load_router_config",
        lambda: {
            "enabled": True,
            "shadow": {"enabled": False},
            "rules": [
                {"id": "adversarial-review", "status": "stable",
                 "when": {"keywords": {"contains": "audit"},
                          "assignee": {"eq": "reviewer"}},
                 "then": {"profile": "reviewer", "model": "T4"}},
                {"id": "standard-implementation", "status": "stable",
                 "when": {"has_code": {"eq": True}},
                 "then": {"profile": "coder", "model": "T2"}},
            ],
            "tiers": {
                "T4": {"model": "gpt-5.6-terra", "provider": "openai-codex"},
                "T2": {"model": "glm-5.3", "provider": "zai"},
            },
        },
    )
    result = dp._on_pre_kanban_dispatch(
        task_id="t100", profile_name="x", board="capability-router",
        assignee="coder", run_id=100,
    )
    assert result != {"model": "gpt-5.6-terra", "provider": "openai-codex"}

    # The same text, for a reviewer, DOES buy the strongest tier.
    result = dp._on_pre_kanban_dispatch(
        task_id="t101", profile_name="x", board="capability-router",
        assignee="reviewer", run_id=101,
    )
    assert result == {"model": "gpt-5.6-terra", "provider": "openai-codex"}


# ---------------------------------------------------------------------------
# Shadow mode MEASURES; it must not spend the breaker's probe slot
# ---------------------------------------------------------------------------

def _seed_expired_open(home, model, provider):
    """Write a breaker entry that is OPEN with its cooldown already past.

    The shape where `is_blocked` is at its least query-like: it transitions to
    HALF_OPEN, consumes the probe, and persists — so this is the state a
    measurement must be able to observe without changing.
    """
    state_dir = home / "hermes-smart-router" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / "breaker-state.json"
    past = time.time() - 200_000
    path.write_text(json.dumps({
        "version": 1,
        "entries": {f"{model}@{provider}": {
            "state": "OPEN",
            "failure_events": [{"kind": "ttfb_stall", "ts": past, "weight": 3}] * 2,
            "cooldown_until": past + 60,
            "backoff_seconds": 60.0,
            "last_failure_kind": "ttfb_stall",
        }},
    }), encoding="utf-8")
    return path


def _shipped_with_breaker():
    """The shipped policy with the auto-breaker on (it ships on)."""
    import yaml
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "router.example.yaml")
        .read_text(encoding="utf-8")
    )
    config["blocklist"]["auto_breaker"]["enabled"] = True
    return config


def test_a_shadow_card_does_not_spend_the_breakers_probe_slot(
    trace_home, monkeypatch,
):
    """Measuring took rails out of rotation, permanently.

    The hook runs a full `route()` and only then checks the mode, so in shadow
    mode — the SHIPPED DEFAULT — every card produced a real decision against the
    real breaker for a card it never dispatches. `is_blocked` flips an
    expired-OPEN rail to HALF_OPEN, consumes its single probe and PERSISTS that;
    HALF_OPEN is left only by a RECORDED outcome, and a shadow card records none.
    So the rail was excluded for good, by a measurement.

    Asserted on the bytes AND on the consequence: a later reader must still see
    the probe available.
    """
    config = _shipped_with_breaker()
    tier = config["tiers"]["T1"]
    model, provider = tier["model"], tier["provider"]
    state = _seed_expired_open(trace_home, model, provider)
    before = state.read_bytes()

    monkeypatch.setattr(dp, "_load_router_config", lambda: config)
    monkeypatch.setattr(
        dp, "_read_kanban_task",
        lambda task_id, board: _card("Rename getCwd in src/utils.py"),
    )

    for i in range(3):
        assert dp._on_pre_kanban_dispatch(
            task_id=f"shadow-{i}", profile_name="x", board="default",
            assignee="coder", run_id=i,
        ) is None

    assert state.read_bytes() == before, (
        "a shadow card rewrote the breaker state it was only supposed to measure"
    )
    # The consequence, not just the bytes: the probe is still there for the real
    # decision that will actually test the rail.
    from router.blocklist import Blocklist
    assert Blocklist(config).would_block(model, provider) is False
    assert Blocklist(config).is_blocked(model, provider) is False

    # And the shadow trace still recorded a decision — the measurement is intact.
    entries = [e for e in _read_trace() if e.get("shadow") is True]
    assert len(entries) == 3
    assert all(e.get("cause") for e in entries)


def test_a_live_card_may_spend_the_probe_because_it_really_dispatches(
    trace_home, monkeypatch,
):
    """The other side of the same rule: live mode IS a decision.

    It returns a model the dispatcher applies, so it is entitled to the probe —
    and must persist the transition, or two concurrent live cards both probe a
    recovering rail.
    """
    config = _shipped_with_breaker()
    config["shadow"] = {"enabled": False}
    tier = config["tiers"]["T1"]
    model, provider = tier["model"], tier["provider"]
    state = _seed_expired_open(trace_home, model, provider)
    before = state.read_bytes()

    monkeypatch.setattr(dp, "_load_router_config", lambda: config)
    monkeypatch.setattr(
        dp, "_read_kanban_task",
        lambda task_id, board: _card("Rename getCwd in src/utils.py"),
    )
    dp._on_pre_kanban_dispatch(
        task_id="live-1", profile_name="x", board="default",
        assignee="coder", run_id=1,
    )
    assert state.read_bytes() != before, (
        "a live decision must persist the probe it consumed"
    )


class TestTheObservingBlocklist:
    """The proxy's own contract: same answers, no spend, everything else forwards."""

    @staticmethod
    def _real(home):
        from router.blocklist import Blocklist
        config = _shipped_with_breaker()
        _seed_expired_open(home, "glm-5.3-flash", "zai")
        return Blocklist(config), config

    def test_is_blocked_answers_without_consuming_the_probe(self, trace_home):
        real, config = self._real(trace_home)
        observing = dp._ObservingBlocklist(real)

        # Ten looks, no spend.
        for _ in range(10):
            assert observing.is_blocked("glm-5.3-flash", "zai") is False
        from router.blocklist import Blocklist
        assert Blocklist(config).is_blocked("glm-5.3-flash", "zai") is False, (
            "the proxy consumed the probe a real decision needed"
        )

    def test_a_still_cooling_rail_still_reads_as_blocked(self, trace_home):
        """The proxy must not turn "blocked" into "clean" — only "do not spend"."""
        from router.blocklist import Blocklist
        config = _shipped_with_breaker()
        state_dir = trace_home / "hermes-smart-router" / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        now = time.time()
        (state_dir / "breaker-state.json").write_text(json.dumps({
            "version": 1,
            "entries": {"glm-5.3-flash@zai": {
                "state": "OPEN",
                "failure_events": [{"kind": "ttfb_stall", "ts": now, "weight": 3}] * 2,
                "cooldown_until": now + 86_400, "backoff_seconds": 60.0,
                "last_failure_kind": "ttfb_stall"}},
        }), encoding="utf-8")
        observing = dp._ObservingBlocklist(Blocklist(config))
        assert observing.is_blocked("glm-5.3-flash", "zai") is True

    def test_a_manual_ban_reads_through_unchanged(self, trace_home):
        from router.blocklist import Blocklist
        config = _shipped_with_breaker()
        config["blocklist"]["manual_ban"] = [
            {"model": "gpt-5.5", "provider": "", "reason": "test"}
        ]
        observing = dp._ObservingBlocklist(Blocklist(config))
        assert observing.is_blocked("gpt-5.5", "openai-codex") is True

    def test_every_other_member_forwards_untouched(self, trace_home):
        """Only ``is_blocked`` is special; the proxy must stay a drop-in."""
        real, _config = self._real(trace_home)
        observing = dp._ObservingBlocklist(real)

        # __getattr__ forwarding: methods, and the same answers as the real one.
        assert observing.fallback_chain() == real.fallback_chain()
        assert observing.manual_bans() == real.manual_bans()
        assert observing.breaker_enabled() is real.breaker_enabled()
        assert observing.fallback_for("glm-5.3-flash") == (
            real.fallback_for("glm-5.3-flash")
        )
        assert observing.would_block("gpt-5.5", "") is real.would_block("gpt-5.5", "")
        with pytest.raises(AttributeError):
            observing.no_such_member
