"""Shadow premium-request count: 1 per worker/child, multiplier-weighted, per tier."""

from __future__ import annotations

import router.premium_budget as pb

NOW = 1_000_000.0


def _d(tier, model, provider, source="kanban", ts=NOW - 10, fallback=()):
    return {"ts": ts, "source": source,
            "output": {"model": model, "provider": provider, "attempted_model": model,
                       "attempted_provider": provider, "fallback": list(fallback)},
            "steps": [{"stage": "classify", "in": {"tier": tier}}]}


def test_one_request_per_worker_and_child_weighted_by_multiplier():
    rows = [
        _d("T3", "gpt-5.4", "copilot"),
        _d("T3", "claude-sonnet-5.5", "copilot", source="delegate"),
        _d("T3", "glm-5.3", "zai"),
        _d("T3", "gpt-5.4", "copilot", source="chat"),  # chat is not a worker
    ]
    out = pb.summarize(rows, days=7, now=NOW)
    t3 = out["tiers"]["T3"]
    assert t3["turns"] == 3
    assert t3["premium_requests"] == 7.0  # 6 + 1.0 (unpublished) + 0
    assert out["unpublished_multiplier"] == ["claude-sonnet-5.5"]
    assert out["total_premium_requests"] == 7.0


def test_exposure_counts_fallback_hop_without_billing_it():
    rows = [_d("T1", "glm-5.3-flash", "zai",
               fallback=[{"model": "gpt-5.4", "provider": "copilot"}])]
    t1 = pb.summarize(rows, days=7, now=NOW)["tiers"]["T1"]
    assert t1["premium_requests"] == 0
    assert t1["exposure_requests"] == 6.0


def test_window_filter_and_empty():
    old = _d("T2", "gpt-5.4", "copilot", ts=NOW - 9 * 86400)
    out = pb.summarize([old], days=7, now=NOW)
    assert out["tiers"] == {} and out["total_premium_requests"] == 0
    assert out["mode"] == "shadow"


def test_legacy_trace_with_task_id_counts_as_kanban():
    row = _d("T2", "gpt-5.4", "copilot", source=None)
    row.pop("source")
    row["task_id"] = "t_legacy1"
    out = pb.summarize([row], days=7, now=NOW)
    assert out["tiers"]["T2"]["premium_requests"] == 6.0


def test_report_wraps_summarize():
    assert pb.report([], days=3)["tiers"] == {}


def test_premium_budget_endpoint(tmp_path, monkeypatch):
    from tests.router.test_one_sidecar import _app, _auth
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    app = _app(tmp_path)
    assert app.dispatch("GET", "/premium-budget", {})[0] == 401
    status, body = app.dispatch("GET", "/premium-budget", _auth())
    assert status == 200 and body["tiers"] == {}
    assert app.dispatch("GET", "/premium-budget", _auth(), query={"days": ["bad"]})[0] == 200
