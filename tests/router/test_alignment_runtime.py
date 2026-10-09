"""F7: observer wiring, daemon thread, shadow mode, persisted state."""

import importlib
import json
import threading
from types import SimpleNamespace

import pytest

from router import alignment_runtime as rt

CFG = {
    "enabled": True,
    "mode": "shadow",
    "trigger": {"min_iterations": 1, "pct": [50], "fallback_max_iterations": 10},
    "judge": {"chain": [{"model": "m", "provider": "other"}]},
}


class LLM:
    def __init__(self, parsed=None, exc=None):
        self.parsed, self.exc, self.calls = parsed, exc, []

    def complete_structured(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return SimpleNamespace(parsed=self.parsed)


CONTINUE = {"verdict": "continue", "confidence": 0.9, "reasons": [], "steer_message": "",
            "evidence": []}
BLOCK = {"verdict": "block", "confidence": 0.95, "reasons": ["drift"], "steer_message": "",
         "evidence": []}


@pytest.fixture(autouse=True)
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_1")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "7")
    monkeypatch.setenv("HERMES_PROFILE", "coder")
    rt._HISTORY.clear()
    rt._BREAKER = rt.BreakerState({})
    return tmp_path


def lines(tmp_path):
    p = tmp_path / "hermes-smart-router" / "state" / "alignment.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def observe(llm, cfg=CFG, n=6, **kw):
    rt.stash_history("s1", [{"role": "user", "content": "do X"}])
    return rt.observe_post_api_request(
        SimpleNamespace(llm=llm), cfg, session_id="s1", api_call_count=n,
        provider="worker", now=1000.0, **kw)


class Inert:
    def __init__(self, **kw):
        self.kw = kw

    def start(self):
        pass


def test_fires_on_daemon_thread_and_logs(env):
    llm = LLM(CONTINUE)
    t = observe(llm)
    assert t.daemon and t.name == "alignment-eval"
    t.join(5)
    (row,) = lines(env)
    assert row["verdict"] == "continue" and row["mode"] == "shadow"
    assert row["action_taken"] is False and row["task_id"] == "t_1"
    assert llm.calls[0]["provider"] == "other"


def test_never_inline(env):
    llm = LLM(CONTINUE)
    t = observe(llm, spawn=Inert)
    assert t is not None and llm.calls == [] and t.kw["daemon"] is True


def test_shadow_block_verdict_acts_on_nothing(env):
    observe(LLM(BLOCK)).join(5)
    (row,) = lines(env)
    assert row["verdict"] == "block" and row["action_taken"] is False


def test_state_persisted_and_survives_reload(env):
    observe(LLM(CONTINUE)).join(5)
    st = rt.load_state("s1")
    assert st.evaluations == 1 and ("s1", "7", "50") in st.fired
    importlib.reload(rt)  # plugin reload: module state is gone, the file is not
    assert rt.load_state("s1") == st
    assert observe(LLM(CONTINUE)) is None  # rung not replayed


def test_acted_verdict_starts_cooldown(env):
    observe(LLM(BLOCK)).join(5)
    assert rt.load_state("s1").last_action_iteration == 6


def test_rung_marked_before_thread_runs(env):
    observe(LLM(), spawn=Inert)
    assert rt.load_state("s1").evaluations == 1
    assert observe(LLM(), spawn=Inert) is None


@pytest.mark.parametrize("cfg", [None, {}, {"enabled": False}, {**CFG, "scopes": ["chat"]}])
def test_inert(env, cfg):
    assert observe(LLM(), cfg=cfg) is None


def test_inert_outside_kanban(env, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    assert observe(LLM()) is None


def test_inert_without_session(env):
    assert rt.observe_post_api_request(None, CFG, api_call_count=9) is None


def test_below_threshold_no_fire(env):
    assert observe(LLM(), n=0) is None


def test_observer_never_raises(env):
    assert rt.observe_post_api_request(None, CFG, session_id="s", api_call_count="x") is None


def test_judge_failure_fails_open_and_logs(env):
    observe(LLM(exc=RuntimeError("boom"))).join(5)
    (row,) = lines(env)
    assert row["verdict"] == "continue" and row["failed_open"]


def test_eval_thread_swallows_errors(env, monkeypatch):
    def boom(*a, **k):
        raise ValueError("x")

    monkeypatch.setattr(rt.alignment_package, "build_package", boom)
    observe(LLM(CONTINUE)).join(5)
    assert lines(env) == []


def test_models_cache_error_is_advisory(env, monkeypatch):
    def boom(p):
        raise OSError("no")

    monkeypatch.setattr(rt, "load_cache", boom)
    llm = LLM(CONTINUE)
    observe(llm).join(5)
    assert llm.calls


def test_models_cache_filters_chain(env):
    (env / "provider_models_cache.json").write_text(json.dumps({"other": {"models": ["m"]}}))
    llm = LLM(CONTINUE)
    observe(llm).join(5)
    assert llm.calls


def test_stash_history_bounds_and_guards():
    rt.stash_history("", [])
    rt.stash_history("a", "nope")
    assert rt._HISTORY == {}
    for i in range(40):
        rt.stash_history(f"s{i}", [])
    assert len(rt._HISTORY) == 32 and "s0" not in rt._HISTORY


def test_state_file_bounded_and_corrupt_tolerated(env):
    for i in range(rt.MAX_SESSIONS + 5):
        rt.save_state(f"s{i}", rt.alignment.AlignmentState())
    assert len(rt._read_all()) == rt.MAX_SESSIONS
    rt._state_path().write_text("{nope")
    assert rt.load_state("s1") == rt.alignment.AlignmentState()
    rt._state_path().write_text("[1]")
    assert rt._read_all() == {}


def test_log_path_absolute_and_custom(env, tmp_path):
    cfg = {"log": {"path": str(tmp_path / "x.jsonl")}}
    rt.append_log(cfg, {"a": 1})
    assert (tmp_path / "x.jsonl").exists()


def test_threading_module_used():
    assert rt.threading is threading


def test_save_breaker_swallows_errors(env, monkeypatch):
    monkeypatch.setattr(rt, "state_dir", lambda: (_ for _ in ()).throw(OSError("x")))
    rt.save_breaker(0.0)
