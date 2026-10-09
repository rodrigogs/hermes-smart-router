"""F8: adjust / block / comment executors against a fake kanban DB."""

from types import SimpleNamespace

import pytest

from router import alignment_actions as aa
from router.alignment_judge import Verdict

EV = [{"message_index": 1, "quote": "q"}]
CFG = {"mode": "enforce", "thresholds": {"allow_direct_block": True, "adjust_limit": 99}}


class FakeKB:
    def __init__(self, comments=(), block_ok=True):
        self.calls, self.comments, self.block_ok = [], list(comments), block_ok
        self.closed = 0

    def connect(self, board=None):
        self.calls.append(("connect", board))
        return SimpleNamespace(close=lambda: setattr(self, "closed", self.closed + 1))

    def list_comments(self, conn, task_id):
        self.calls.append(("list_comments", task_id))
        return [SimpleNamespace(author=a, body=b) for a, b in self.comments]

    def add_comment(self, conn, task_id, author, body):
        self.calls.append(("add_comment", task_id, author, body))
        return 1

    def block_task(self, conn, task_id, **kw):
        self.calls.append(("block_task", task_id, kw))
        return self.block_ok

    def writes(self):
        return [c for c in self.calls if c[0] in ("add_comment", "block_task")]


def V(verdict="adjust", conf=0.9, steer="do Y", ev=EV, failed=""):
    return Verdict(verdict, conf, ["drift"], steer, list(ev), failed_open=failed)


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def run(v, cfg=CFG, kb=None, **kw):
    kb = kb or FakeKB()
    out = aa.apply(v, cfg, task_id="t_1", run_id="7", session_id="s1", kb=kb, **kw)
    return out, kb


# ---- continue never touches the card (mutation-style: every continue-like input)
@pytest.mark.parametrize("v", [
    V("continue", 0.99),
    V("adjust", 0.1),                       # below threshold -> downgraded
    V("adjust", 0.9, ev=[]),                # no evidence
    V("adjust", 0.9, steer=" "),            # no steer text
    V("block", 0.5),                        # below block threshold
    V("adjust", 0.9, failed="timeout"),     # judge failed open
])
def test_continue_never_touches_card(v):
    kb = FakeKB()
    out = aa.apply(v, CFG, task_id="t_1", run_id="7", session_id="s1", kb=kb)
    assert out.action == "continue" and not out.acted
    assert kb.calls == []  # not even connect()


def test_continue_guard_mutation_is_detected(monkeypatch):
    """If gate() ever let continue through, the no-call assertion above must fail."""
    monkeypatch.setattr(aa, "gate", lambda *a: "adjust")
    kb = FakeKB()
    aa.apply(V("continue", 0.99), CFG, task_id="t_1", run_id="7", session_id="s1", kb=kb)
    assert kb.calls  # mutated gate reaches the DB: the guard is what protects the card


def test_shadow_and_alert_modes_never_act():
    for mode in ("shadow", "alert"):
        out, kb = run(V("block"), {**CFG, "mode": mode})
        assert out.action == "block" and not out.acted and kb.calls == []


def test_block_uses_needs_input_cas_and_comments():
    out, kb = run(V("block"))
    assert out.acted and out.action == "block"
    kind, tid, kw = [c for c in kb.calls if c[0] == "block_task"][0]
    assert tid == "t_1" and kw["kind"] == "needs_input" and kw["expected_run_id"] == 7
    assert "drift" in kw["reason"]
    assert any(c[0] == "add_comment" and "[alignment] block" in c[3] for c in kb.calls)
    assert kb.closed == 1


def test_block_cas_lost_no_comment():
    out, kb = run(V("block"), kb=FakeKB(block_ok=False))
    assert not out.acted and out.reason == "cas_lost"
    assert not any(c[0] == "add_comment" for c in kb.calls)


def test_block_needs_prior_adjust_unless_direct():
    cfg = {"mode": "enforce"}
    out, _ = run(V("block"), cfg)  # no prior adjust -> becomes an adjust
    assert out.action == "adjust"
    out, _ = run(V("block"), cfg)  # one adjust recorded -> block allowed
    assert out.action == "block"


def test_adjust_limit_turns_third_drift_into_block():
    cfg = {"mode": "enforce", "thresholds": {"adjust_limit": 1}}
    assert run(V("adjust"), cfg)[0].action == "adjust"
    assert run(V("adjust", 0.9), cfg)[0].action == "block"
    assert run(V("adjust", 0.7), cfg)[0].action == "continue"  # below block confidence


def test_ack_override_suppresses_actions():
    kb = FakeKB(comments=[("rodrigo", "ok [alignment-ack]")])
    out, kb = run(V("block"), kb=kb)
    assert not out.acted and out.reason == "acknowledged" and kb.writes() == []


def test_own_comment_quoting_ack_is_not_an_ack():
    out, kb = run(V("adjust"), kb=FakeKB(comments=[("alignment", "reply [alignment-ack]")]))
    assert out.acted


def test_adjust_delivery_channels():
    out, kb = run(V("adjust"))
    assert out.channel == "pre_tool_call"
    assert any(c[0] == "add_comment" and "steer: do Y" in c[3] for c in kb.calls)
    assert run(V("adjust"), goal_mode=True)[0].channel == "pre_llm_call"
    assert run(V("adjust"), upstream_steer=True)[0].channel == "steer"
    cfg = {**CFG, "actions": {"adjust": {"deliver": "pre_tool_call", "comment": False}}}
    out, kb = run(V("adjust"), cfg, goal_mode=True)
    assert out.channel == "pre_tool_call" and not any(c[0] == "add_comment" for c in kb.calls)
    cfg["actions"]["adjust"]["deliver"] = "steer"
    assert run(V("adjust"), cfg)[0].channel == "pre_tool_call"
    assert run(V("adjust"), cfg, goal_mode=True)[0].channel == "pre_llm_call"
    assert run(V("adjust"), cfg, upstream_steer=True)[0].channel == "steer"


def test_kanban_block_disabled():
    cfg = {**CFG, "actions": {"block": {"kanban_block": False}}}
    out, kb = run(V("block"), cfg)
    assert not out.acted and kb.writes() == []


def test_non_numeric_run_id_passes_none():
    kb = FakeKB()
    aa.apply(V("block"), CFG, task_id="t_1", run_id="", session_id="s1", kb=kb)
    assert [c for c in kb.calls if c[0] == "block_task"][0][2]["expected_run_id"] is None


def test_errors_fail_open():
    class Boom(FakeKB):
        def connect(self, board=None):
            raise RuntimeError("db down")
    out, _ = run(V("block"), kb=Boom())
    assert out.action == "continue" and not out.acted and out.reason == "error"


# ---- hooks
def test_pre_tool_call_one_shot_steer_and_never_kanban_tools():
    run(V("adjust"))
    kw = {"session_id": "s1"}
    assert aa.pre_tool_call(CFG, tool_name="kanban_comment", **kw) is None
    first = aa.pre_tool_call(CFG, tool_name="terminal", **kw)
    assert first["action"] == "block" and "do Y" in first["message"]
    assert aa.pre_tool_call(CFG, tool_name="terminal", **kw) is None  # one-shot
    assert aa.pre_tool_call(CFG, tool_name="terminal", session_id="other") is None


def test_pre_tool_call_kanban_tool_does_not_consume_steer():
    run(V("adjust"))
    aa.pre_tool_call(CFG, tool_name="kanban_heartbeat", session_id="s1")
    assert aa.pre_tool_call(CFG, tool_name="terminal", session_id="s1")


def test_after_block_vetoes_everything_but_kanban():
    run(V("block"))
    for name in ("kanban_complete", "kanban_block", "kanban_comment"):
        assert aa.pre_tool_call(CFG, tool_name=name, session_id="s1") is None
    for _ in range(2):
        assert aa.pre_tool_call(CFG, tool_name="terminal", session_id="s1")["action"] == "block"


def test_pre_tool_call_inert_outside_enforce():
    run(V("adjust"))
    assert aa.pre_tool_call({"mode": "shadow"}, tool_name="terminal", session_id="s1") is None
    assert aa.pre_llm_call({"mode": "shadow"}, session_id="s1") is None


def test_pre_llm_call_goal_mode_context_once():
    run(V("adjust"), goal_mode=True)
    assert aa.pre_tool_call(CFG, tool_name="terminal", session_id="s1") is None
    ctx = aa.pre_llm_call(CFG, session_id="s1")
    assert "do Y" in ctx["context"]
    assert aa.pre_llm_call(CFG, session_id="s1") is None


def test_hooks_fail_open(monkeypatch):
    monkeypatch.setattr(aa, "_read", lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert aa.pre_tool_call(CFG, tool_name="t", session_id="s1") is None
    assert aa.pre_llm_call(CFG, session_id="s1") is None


def test_bad_threshold_value_uses_default():
    cfg = {"mode": "enforce", "thresholds": {"adjust_min_confidence": "x"}}
    assert aa.gate(V("adjust", 0.7), cfg, 0) == "adjust"


def test_block_without_prior_adjust_or_steer_continues():
    assert aa.gate(V("block", 0.9, steer=" "), {"mode": "enforce"}, 0) == "continue"


def test_pending_store_evicts_oldest_sessions(monkeypatch):
    monkeypatch.setattr(aa, "MAX_SESSIONS", 2)
    for s in ("a", "b", "c"):
        aa._update(s, "t_1", kind="x")
    assert list(aa._read()["sessions"]) == ["b", "c"]


def test_block_without_comment():
    cfg = dict(CFG, actions={"block": {"comment": False}})
    out, kb = run(V("block"), cfg)
    assert out.action == "block" and out.acted
    assert [c[0] for c in kb.writes()] == ["block_task"]
