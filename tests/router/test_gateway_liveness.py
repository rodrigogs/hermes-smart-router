"""Inert-hook alert: fires on deploy-without-restart, silent after restart, heartbeat once."""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path

import pytest

import router.gateway_liveness as gl

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("dp_liveness_plugin", REPO_ROOT / "__init__.py")
dp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dp)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_ROUTE_TRACE_FILE", raising=False)
    monkeypatch.setitem(gl._STATE, "heartbeat_done", False)
    return tmp_path


def _rows():
    try:
        return [json.loads(x) for x in gl.liveness_path().read_text().splitlines()]
    except OSError:
        return []


def test_alert_fires_when_plugin_newer_than_process(home, caplog):
    with caplog.at_level(logging.WARNING):
        v = gl.check(started=100.0, mtime=200.0)
    assert v["inert"] is True
    assert "restart the gateway" in caplog.text
    assert [r["event"] for r in _rows()] == ["inert_hook"]


def test_alert_silent_after_restart(home, caplog):
    with caplog.at_level(logging.WARNING):
        v = gl.check(started=300.0, mtime=200.0)
    assert v["inert"] is False
    assert caplog.text == ""
    assert _rows() == []


def test_first_tick_records_heartbeat_once(home, monkeypatch):
    monkeypatch.setattr(gl, "process_start_ts", lambda: 300.0)
    monkeypatch.setattr(gl, "plugin_mtime", lambda root=None: 200.0)
    dp._on_kanban_dispatch_tick(board="b")
    dp._on_kanban_dispatch_tick(board="b")
    assert [r["event"] for r in _rows()] == ["boot_heartbeat"]


def test_first_tick_on_stale_deploy_alerts_and_heartbeats(home, monkeypatch):
    monkeypatch.setattr(gl, "process_start_ts", lambda: 100.0)
    monkeypatch.setattr(gl, "plugin_mtime", lambda root=None: 200.0)
    dp._on_kanban_dispatch_tick()
    rows = _rows()
    assert [r["event"] for r in rows] == ["inert_hook", "boot_heartbeat"]
    assert rows[1]["inert"] is True


def test_real_probes_and_failures_never_raise(home, tmp_path, monkeypatch):
    assert gl.process_start_ts() > 0
    (tmp_path / "x.py").write_text("")
    assert gl.plugin_mtime(tmp_path) > 0
    assert gl.plugin_mtime(tmp_path / "none") == 0.0
    monkeypatch.setattr(gl, "check", lambda: 1 / 0)
    gl.on_dispatch_tick()
    monkeypatch.setattr(gl, "liveness_path", lambda: tmp_path / "f" / "x")
    (tmp_path / "f").write_text("file blocks mkdir")
    gl._append({"a": 1})
