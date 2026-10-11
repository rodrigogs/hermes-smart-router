"""Per-tier cost/latency: subscription_included in its own bucket, unknown never zero."""

from __future__ import annotations

import sqlite3
import types
from decimal import Decimal

import router.tier_cost as tc

NOW = 1_000_000.0


def _dec(task, run, tier):
    return {"task_id": task, "run_id": run, "output": {"model": "m"},
            "steps": [{"stage": "classify", "in": {"tier": tier}}]}


def _run(rid, task, start, end):
    return {"id": rid, "task_id": task, "started_at": start, "ended_at": end, "outcome": "completed"}


def _sess(task, start, **kw):
    row = {"title": f"Work kanban task {task}", "started_at": start, "model": "m",
           "billing_mode": None, "estimated_cost_usd": 0.0, "actual_cost_usd": None,
           "cost_status": "unknown"}
    row.update(kw)
    return row


def test_included_has_own_bucket_and_metered_sums():
    out = tc.summarize(
        [_dec("t_aaaaaa", 1, "T2"), _dec("t_bbbbbb", 2, "T2")],
        [_run(1, "t_aaaaaa", NOW - 100, NOW - 40), _run(2, "t_bbbbbb", NOW - 50, NOW - 20)],
        [_sess("t_aaaaaa", NOW - 99, billing_mode="subscription_included", model="gpt"),
         _sess("t_bbbbbb", NOW - 49, actual_cost_usd=0.25)],
        days=1, now=NOW, pricing=None)
    t2 = out["tiers"]["T2"]
    assert t2["cost_usd"] == 0.25
    assert t2["subscription_included"] == {"sessions": 1, "models": {"gpt": 1}}
    assert t2["runs"] == 2 and t2["latency_s"]["avg"] == 45.0 and t2["latency_s"]["p95"] == 60.0


def test_unknown_cost_is_counted_not_zero_and_window_excludes_old():
    out = tc.summarize(
        [_dec("t_aaaaaa", 1, "T1"), _dec("t_old000", 9, "T1")],
        [_run(1, "t_aaaaaa", NOW - 10, None), _run(9, "t_old000", NOW - 10 * 86400, NOW - 10 * 86400 + 5)],
        [_sess("t_aaaaaa", NOW - 9), _sess("t_nomatch", NOW - 9)],
        days=1, now=NOW, pricing=None)
    t1 = out["tiers"]["T1"]
    assert t1["runs"] == 1 and t1["cost_unknown_sessions"] == 1 and t1["cost_usd"] == 0.0
    assert t1["latency_s"] == {"n": 0, "avg": None, "p50": None, "p95": None}
    assert out["unattributed_sessions"] == 1


def test_session_without_decision_run_is_unattributed():
    out = tc.summarize([], [_run(1, "t_aaaaaa", NOW - 10, NOW)], [_sess("t_aaaaaa", NOW - 9)],
                       days=1, now=NOW, pricing=None)
    assert out["tiers"] == {} and out["unattributed_sessions"] == 1


class _Pricing:
    class _Route:
        billing_mode = "official_models_api"

    class CanonicalUsage:
        def __init__(self, **kw):
            self.kw = kw

    def resolve_billing_route(self, model, provider=None, base_url=None):
        return self._Route()

    def estimate_usage_cost(self, model, usage, provider=None, base_url=None):
        return types.SimpleNamespace(status="estimated", amount_usd=Decimal("1.5"))


def test_usage_pricing_fills_missing_cost_from_tokens():
    sess = _sess("t_a", 0, input_tokens=10, output_tokens=5)
    assert tc.session_cost(sess, _Pricing()) == {"included": False, "usd": 1.5}


def test_usage_pricing_included_status_and_failures():
    class Inc(_Pricing):
        def estimate_usage_cost(self, *a, **k):
            return types.SimpleNamespace(status="included", amount_usd=Decimal(0))

    class Boom(_Pricing):
        def resolve_billing_route(self, *a, **k):
            raise RuntimeError

        def estimate_usage_cost(self, *a, **k):
            raise RuntimeError

    class NoAmount(_Pricing):
        def estimate_usage_cost(self, *a, **k):
            return types.SimpleNamespace(status="unknown", amount_usd=None)

    s = _sess("t_a", 0, input_tokens=1)
    assert tc.session_cost(s, Inc()) == {"included": True, "usd": 0.0}
    assert tc.session_cost(s, Boom()) == {"included": False, "usd": None}
    assert tc.session_cost(s, NoAmount()) == {"included": False, "usd": None}
    assert tc.session_cost(_sess("t_a", 0, cost_status="included"), None) == {"included": True, "usd": 0.0}
    assert tc.session_cost(_sess("t_a", 0), _Pricing()) == {"included": False, "usd": None}


def test_readers_against_real_sqlite(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    assert tc.read_runs(0) == [] and tc.read_sessions(0) == []
    board = tmp_path / "kanban" / "boards" / "b"
    board.mkdir(parents=True)
    db = sqlite3.connect(board / "kanban.db")
    db.execute("CREATE TABLE task_runs (id INTEGER, task_id TEXT, started_at INTEGER, ended_at INTEGER, outcome TEXT)")
    db.execute("INSERT INTO task_runs VALUES (4,'t_aaaaaa',100,160,'completed')")
    db.commit(); db.close()
    st = sqlite3.connect(tmp_path / "state.db")
    st.execute("CREATE TABLE sessions (id TEXT, model TEXT, title TEXT, started_at REAL, source TEXT, "
               "billing_provider TEXT, billing_base_url TEXT, billing_mode TEXT, estimated_cost_usd REAL, "
               "actual_cost_usd REAL, cost_status TEXT, input_tokens INT, output_tokens INT, "
               "cache_read_tokens INT, cache_write_tokens INT)")
    st.execute("INSERT INTO sessions VALUES ('s','m','Work kanban task t_aaaaaa',101,'kanban',NULL,NULL,NULL,0.5,NULL,'estimated',0,0,0,0)")
    st.execute("INSERT INTO sessions VALUES ('c','m','chat',101,'cli',NULL,NULL,NULL,9,NULL,'x',0,0,0,0)")
    st.commit(); st.close()
    assert [r["id"] for r in tc.read_runs(0)] == [4]
    assert [r["id"] for r in tc.read_sessions(0)] == ["s"]
    rep = tc.report([_dec("t_aaaaaa", 4, "T3")], days=36500)
    assert rep["tiers"]["T3"]["cost_usd"] == 0.5


def test_readers_survive_corrupt_dbs(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "kanban.db"))
    (tmp_path / "kanban.db").write_text("not sqlite")
    (tmp_path / "state.db").write_text("not sqlite")
    assert tc.read_runs(0) == [] and tc.read_sessions(0) == []


def test_pricing_loader_absent_returns_none(monkeypatch):
    import sys

    package = types.ModuleType("agent")
    monkeypatch.setitem(sys.modules, "agent", package)
    monkeypatch.delitem(sys.modules, "agent.usage_pricing", raising=False)
    assert tc._pricing() is None


def test_helpers():
    assert tc._task_of("Work kanban task t_8846edad #2") == "t_8846edad"
    assert tc._task_of(None) is None
    assert tc._int("x") == "x" and tc._int("3") == 3
    assert tc._percentile([], 0.5) is None


def test_tier_cost_endpoint(tmp_path, monkeypatch):
    from tests.router.test_one_sidecar import _app, _auth
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    app = _app(tmp_path)
    assert app.dispatch("GET", "/tier-cost", {})[0] == 401
    status, body = app.dispatch("GET", "/tier-cost", _auth())
    assert status == 200 and body["days"] == 7 and body["tiers"] == {}
    assert app.dispatch("GET", "/tier-cost", _auth(), query={"days": ["bad"]})[1]["days"] == 7
    assert app.dispatch("POST", "/tier-cost", _auth(), body={})[0] == 405


def test_pricing_module_resolves_when_runtime_importable(monkeypatch):
    import sys
    fake = types.ModuleType("agent.usage_pricing")
    pkg = types.ModuleType("agent")
    pkg.usage_pricing = fake
    monkeypatch.setitem(sys.modules, "agent", pkg)
    monkeypatch.setitem(sys.modules, "agent.usage_pricing", fake)
    assert tc._pricing() is fake


def test_owning_run_prefers_latest_overlapping_and_skips_outside():
    early = _run(1, "t_aaaaaa", NOW - 500, NOW - 400)
    late = _run(2, "t_aaaaaa", NOW - 100, None)
    outside = _run(3, "t_aaaaaa", NOW + 5000, NOW + 6000)
    assert tc._owning_run([late, early, outside], NOW - 50)["id"] == 2
    assert tc._owning_run([early, late], NOW - 450)["id"] == 1
    assert tc._owning_run([outside], NOW - 50) is None


def test_invalid_decisions_and_old_sessions_are_ignored():
    bad = [{"task_id": "", "run_id": 1}, {"task_id": "t_aaaaaa", "run_id": None},
           _dec("t_aaaaaa", 1, "T3")]
    out = tc.summarize(bad, [_run(1, "t_aaaaaa", NOW - 10, NOW)],
                       [_sess("t_aaaaaa", NOW - 5 * 86400)], days=1, now=NOW, pricing=None)
    assert out["tiers"]["T3"]["sessions"] == 0 and out["unattributed_sessions"] == 0


def test_owning_run_keeps_newer_when_older_also_overlaps():
    newer = _run(2, "t_aaaaaa", NOW - 100, None)
    older = _run(1, "t_aaaaaa", NOW - 200, None)
    assert tc._owning_run([newer, older], NOW - 50)["id"] == 2
