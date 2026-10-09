"""F9: alert delivery, dedupe, outbox fallback, privacy default."""

import json
from types import SimpleNamespace

import pytest

from router import alignment_alert as al

ENTRY = {
    "task_id": "t_1",
    "run_id": "7",
    "mode": "alert",
    "verdict": "block",
    "confidence": 0.9,
    "reasons": ["drift"],
    "evidence": ["secret-ish line"],
}
CFG = {"alert": {"channels": ["telegram", "discord:#ops"]}}


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def outbox(tmp_path):
    p = tmp_path / "hermes-smart-router" / "state" / al.OUTBOX_FILE
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def test_sends_each_channel_once_per_key():
    calls = []
    s = lambda t, x: calls.append((t, x)) or True  # noqa: E731
    assert al.send_alert(CFG, ENTRY, sender=s) == {"telegram": "sent", "discord:#ops": "sent"}
    assert al.send_alert(CFG, ENTRY, sender=s) == {}
    assert len(calls) == 2
    assert al.send_alert(CFG, {**ENTRY, "run_id": "8"}, sender=s)


def test_no_transcript_by_default_and_opt_in():
    assert "secret-ish" not in al.format_message(ENTRY, {})
    assert "secret-ish" in al.format_message(ENTRY, {"include_transcript_excerpt": True})
    assert "drift" in al.format_message(ENTRY, {})
    assert "drift" not in al.format_message(ENTRY, {"include_reasons": False})
    assert "evidence" not in al.format_message(
        {**ENTRY, "evidence": []}, {"include_transcript_excerpt": True}
    )


def test_failure_and_exception_go_to_outbox(home):
    def s(t, x):
        if t == "telegram":
            raise OSError("no hermes")
        return False

    assert al.send_alert(CFG, ENTRY, sender=s) == {"telegram": "outbox", "discord:#ops": "outbox"}
    assert [o["target"] for o in outbox(home)] == ["telegram", "discord:#ops"]


def test_gating():
    s = lambda t, x: True  # noqa: E731
    assert al.send_alert({}, ENTRY, sender=s) == {}
    assert al.send_alert(CFG, {**ENTRY, "verdict": "continue"}, sender=s) == {}
    assert al.send_alert(CFG, {**ENTRY, "failed_open": True}, sender=s) == {}
    shadow = {**ENTRY, "mode": "shadow"}
    assert al.send_alert(CFG, shadow, sender=s) == {}
    on = {"alert": {"channels": ["telegram"], "on": ["shadow_block"]}}
    assert al.send_alert(on, shadow, sender=s) == {"telegram": "sent"}


def test_never_raises():
    assert al.send_alert(None, ENTRY) == {}


def test_corrupt_ledger_is_reset(home):
    p = home / "hermes-smart-router" / "state" / al.LEDGER_FILE
    p.parent.mkdir(parents=True)
    p.write_text("{not json")
    assert al.send_alert(CFG, ENTRY, sender=lambda t, x: True)
    p.write_text('{"a": 1}')
    assert al._read_ledger() == []


def test_hermes_send_subprocess(monkeypatch):
    seen = {}

    def run(cmd, **kw):
        seen["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(al.subprocess, "run", run)
    assert al._hermes_send("telegram", "hi") is True
    assert seen["cmd"] == ["hermes", "send", "-t", "telegram", "-q", "hi"]
