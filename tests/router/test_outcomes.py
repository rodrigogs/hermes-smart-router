"""Outcome journal: hooks record (task_id, run_id) outcomes; /outcomes rates per tier."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import router.outcomes as oc

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("dp_outcomes_plugin", REPO_ROOT / "__init__.py")
dp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dp)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_ROUTE_TRACE_FILE", raising=False)
    return tmp_path


def _dec(task, run, tier="T2", model="m"):
    return {"task_id": task, "run_id": run, "output": {"model": model},
            "steps": [{"stage": "classify", "in": {"tier": tier}}]}


# --- hooks write the outcome (the mutation guard: no write => these fail) -----

def test_completed_hook_records_the_outcome_for_task_and_run(home):
    dp._on_kanban_task_completed(task_id="t1", run_id=7, assignee="coder")
    rows = oc.read_outcomes()
    assert [(r["task_id"], r["run_id"], r["outcome"], r["source"]) for r in rows] == [
        ("t1", 7, "completed", "kanban_task_completed")]


def test_blocked_hook_records_the_outcome(home):
    dp._on_kanban_task_blocked(task_id="t1", run_id=7, reason="need input")
    assert [(r["outcome"], r["source"]) for r in oc.read_outcomes()] == [
        ("blocked", "kanban_task_blocked")]


def test_worker_exit_records_rate_limited_from_outcome_or_exit_kind(home):
    dp._on_kanban_worker_exited(task_id="a", run_id=1, outcome="rate_limited",
                                exit_kind="x", exit_code=75)
    dp._on_kanban_worker_exited(task_id="b", run_id=1, exit_kind="rate_limited")
    dp._on_kanban_worker_exited(task_id="c", run_id=1, outcome="crashed", exit_kind="signal",
                                exit_code=9)
    dp._on_kanban_worker_exited(task_id="d", run_id=1, exit_kind="clean")
    dp._on_kanban_worker_exited(task_id="e", run_id=1)
    assert [r["outcome"] for r in oc.read_outcomes()] == [
        "rate_limited", "rate_limited", "crashed", "clean", "exited"]
    assert oc.read_outcomes()[0]["exit_code"] == 75


def test_hooks_never_raise_when_the_journal_cannot_be_written(home, monkeypatch):
    monkeypatch.setattr(oc, "record_outcome", lambda *a, **k: 1 / 0)
    dp._on_kanban_task_completed(task_id="t1", run_id=1)  # must not raise


def test_package_import_branch_is_taken_when_loaded_as_package(home, monkeypatch):
    monkeypatch.setattr(dp, "_LOADED_AS_PACKAGE", True)
    dp._on_kanban_task_completed(task_id="t1", run_id=1)  # relative import fails -> swallowed
    assert oc.read_outcomes() == []


def test_register_subscribes_the_three_hooks():
    seen = {}

    class Ctx:
        def register_hook(self, name, fn):
            seen[name] = fn

        def register_tool(self, **_k):
            pass

        def register_middleware(self, *_a):
            pass

    dp._REGISTERED_CTX.clear()
    dp.register(Ctx())
    assert seen["kanban_task_completed"] is dp._on_kanban_task_completed
    assert seen["kanban_task_blocked"] is dp._on_kanban_task_blocked
    assert seen["on_kanban_worker_exited"] is dp._on_kanban_worker_exited


# --- journal ------------------------------------------------------------------

def test_record_rejects_a_missing_task_id(home):
    assert oc.record_outcome("", 1, "completed", "s") is False
    assert oc.record_outcome(None, 1, "completed", "s") is False
    assert oc.read_outcomes() == []


def test_record_drops_non_scalar_extras_and_returns_true(home):
    assert oc.record_outcome("t", 1, "completed", "s", keep="x", drop={"a": 1}) is True
    row = oc.read_outcomes()[0]
    assert row["keep"] == "x" and "drop" not in row


def test_record_swallows_oserror(home, monkeypatch):
    monkeypatch.setattr(oc, "outcomes_path", lambda: home / "f" / "outcomes.jsonl")
    (home / "f").write_text("a file where the directory should be")
    assert oc.record_outcome("t", 1, "completed", "s") is False


def test_record_rotates_past_the_cap(home, monkeypatch):
    monkeypatch.setattr(oc, "_MAX_BYTES", 10)
    oc.record_outcome("t1", 1, "completed", "s")
    oc.record_outcome("t2", 1, "blocked", "s")
    assert oc.outcomes_path().with_suffix(".jsonl.1").exists()
    assert [r["task_id"] for r in oc.read_outcomes()] == ["t1", "t2"]


def test_read_skips_corrupt_foreign_and_non_utf8_lines(home):
    p = oc.outcomes_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    good = json.dumps({"schema": "route-outcome/1", "task_id": "t", "run_id": 1,
                       "outcome": "completed"}).encode()
    p.write_bytes(b"{bad\n\xff\xfe\n" + json.dumps({"schema": "x"}).encode() + b"\n[1]\n"
                  + good + b"\n")
    assert len(oc.read_outcomes()) == 1


# --- tier + summary -----------------------------------------------------------

def test_tier_of_prefers_classifier_step_then_model_then_unknown():
    assert oc.tier_of(_dec("t", 1, tier="T3")) == "T3"
    assert oc.tier_of({"steps": ["junk", {"in": "junk"}, {"in": {"tier": ""}}],
                       "output": {"model": "glm"}}) == "glm"
    assert oc.tier_of({"steps": None, "output": {}}) == "unknown"
    assert oc.tier_of({"output": "junk"}) == "unknown"


def test_summarize_rates_per_tier_join_on_task_and_run():
    decisions = [_dec("a", 1), _dec("b", 1), _dec("c", 1), _dec("d", 1),
                 _dec("e", 1, tier="T4"), _dec("a", 2),  # a/2 has no outcome
                 {"task_id": "", "run_id": 1}, _dec("f", [1])]  # unjoinable
    outcomes = [
        {"task_id": "a", "run_id": 1, "outcome": "crashed"},
        {"task_id": "a", "run_id": 1, "outcome": "completed"},  # last wins
        {"task_id": "b", "run_id": 1, "outcome": "blocked"},
        {"task_id": "c", "run_id": 1, "outcome": "rate_limited"},
        {"task_id": "d", "run_id": 1, "outcome": "completed"},
        {"task_id": "a", "run_id": 3, "outcome": "blocked"},  # other run: not joined
        {"task_id": "", "run_id": 1, "outcome": "blocked"},
        {"task_id": "f", "run_id": [1], "outcome": "blocked"},
    ]
    t2 = oc.summarize(decisions, outcomes)["tiers"]["T2"]
    assert (t2["cards"], t2["resolved"]) == (5, 4)
    assert t2["counts"]["no_outcome"] == 1
    assert t2["success_rate"] == 0.5
    assert t2["blocked_rate"] == 0.25
    assert t2["rate_limited_rate"] == 0.25
    t4 = oc.summarize(decisions, outcomes)["tiers"]["T4"]
    assert t4["resolved"] == 0 and t4["success_rate"] is None


# --- sidecar endpoint ---------------------------------------------------------

def test_outcomes_endpoint_reports_rates_end_to_end(home):
    from tests.router.test_one_sidecar import _app, _auth
    rp = home / "hermes-smart-router" / "state" / "routes.jsonl"
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(_dec("t1", 4)) + "\n", encoding="utf-8")
    dp._on_kanban_task_completed(task_id="t1", run_id=4)
    app = _app(home)
    assert app.dispatch("GET", "/outcomes", {})[0] == 401
    status, body = app.dispatch("GET", "/outcomes", _auth())
    assert status == 200
    assert body["tiers"]["T2"]["success_rate"] == 1.0
    assert app.dispatch("POST", "/outcomes", _auth(), body={})[0] == 405
