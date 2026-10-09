import json
import sqlite3

import pytest

from router import alignment_eval as ev
from router import cli


def test_wilson_known_values():
    lo, hi = ev.wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-3)
    assert hi == pytest.approx(0.9433, abs=1e-3)
    assert ev.wilson(0, 0) is None
    assert ev.wilson(0, 5)[0] == 0.0
    assert ev.wilson(5, 5)[1] == 1.0


def test_label_outcome():
    assert ev.label_outcome("completed") == "aligned"
    assert ev.label_outcome("blocked") == "drift"
    assert ev.label_outcome("crashed") == "drift"
    assert ev.label_outcome(None) is None


def test_worst_verdict_skips_failed_open():
    ents = [
        {"task_id": "t", "run_id": 1, "verdict": "adjust"},
        {"task_id": "t", "run_id": 1, "verdict": "block"},
        {"task_id": "t", "run_id": 2, "verdict": "block", "failed_open": "timeout"},
        {"task_id": "t", "run_id": 3, "verdict": "bogus"},
    ]
    assert ev.worst_verdicts(ents) == {("t", 1): "block"}


def test_score_precision_recall():
    pairs = [("block", "drift"), ("block", "aligned"), ("adjust", "drift"),
             ("continue", "drift"), ("continue", "aligned")]
    r = ev.score(pairs)
    assert r["block"]["precision"]["k"] == 1 and r["block"]["precision"]["n"] == 2
    assert r["block"]["recall"]["n"] == 3
    assert r["any"]["recall"]["k"] == 2
    assert r["adjust"]["precision"]["value"] == 1.0
    empty = ev.score([])
    assert empty["block"]["precision"]["value"] is None
    assert "n/a" in ev.render({**empty, "judged_runs": 0, "unlabelled_runs": 0, "failed_open": 0})


def _make_db(path):
    c = sqlite3.connect(path)
    c.executescript("""
    CREATE TABLE tasks (id TEXT, title TEXT, body TEXT);
    CREATE TABLE task_runs (id INTEGER PRIMARY KEY, task_id TEXT, outcome TEXT, ended_at INTEGER);
    CREATE TABLE task_comments (id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, author TEXT, body TEXT);
    INSERT INTO tasks VALUES ('t1','A','body'),('t2','B',NULL);
    INSERT INTO task_runs VALUES (1,'t1','completed',9),(2,'t2','blocked',9),(3,'t2',NULL,NULL);
    INSERT INTO task_comments (task_id,author,body) VALUES ('t1','x','c1'),('t1','x','c2'),('t1','x','c3');
    """)
    c.commit()
    c.close()


def test_report_from_disk_and_readonly(tmp_path):
    db = tmp_path / "k.db"
    _make_db(db)
    before = db.read_bytes()
    log = tmp_path / "a.jsonl"
    log.write_text("\n".join([
        json.dumps({"task_id": "t1", "run_id": 1, "verdict": "block"}),
        json.dumps({"task_id": "t2", "run_id": 2, "verdict": "adjust"}),
        json.dumps({"task_id": "t2", "run_id": 3, "verdict": "block"}),
        "not json", "[1]",
    ]))
    rep = ev.report_from_disk(log, db)
    assert rep["block"]["precision"]["n"] == 1 and rep["block"]["fp"] == 1
    assert rep["adjust"]["tp"] == 1
    assert rep["unlabelled_runs"] == 1
    assert "precision" in ev.render(rep)
    assert db.read_bytes() == before  # live board untouched


def test_report_missing_inputs(tmp_path):
    rep = ev.report_from_disk(tmp_path / "none.jsonl", tmp_path / "none.db")
    assert rep["judged_runs"] == 0
    assert ev.read_log(tmp_path / "none") == []
    assert ev.run_labels(_empty_db(tmp_path)) == {}


def _empty_db(tmp_path):
    p = tmp_path / "e.db"
    sqlite3.connect(p).close()
    return p


def test_kanban_cases_truncates_on_copy(tmp_path):
    db = tmp_path / "k.db"
    _make_db(db)
    cases = ev.kanban_cases(db)
    ids = {c["id"] for c in cases}
    assert ids == {"t1#1@60", "t1#1@85", "t2#2@60", "t2#2@85"}
    assert len([c for c in cases if c["id"] == "t1#1@60"][0]["messages"]) == 1
    assert len([c for c in cases if c["id"] == "t1#1@85"][0]["messages"]) == 2
    assert [c for c in cases if c["id"].startswith("t2")][0]["messages"] == []


def test_corpus_has_labelled_seeded_drift():
    cases = ev.load_corpus()
    assert len(cases) >= 10
    assert {c["label"] for c in cases} == {"aligned", "drift"}
    assert sum(c["label"] == "drift" for c in cases) >= 5


def test_replay_scores_judge_and_fails_open():
    cases = ev.load_corpus()

    def oracle(case):
        return "block" if case["label"] == "drift" else "continue"

    r = ev.replay(cases, oracle)
    assert r["block"]["precision"]["value"] == 1.0 and r["block"]["recall"]["value"] == 1.0

    def broken(case):
        raise RuntimeError

    r2 = ev.replay(cases, broken)
    assert r2["any"]["recall"]["k"] == 0
    assert ev.replay(cases[:1], lambda c: "junk")["any"]["precision"]["n"] == 0


def test_cli_report(tmp_path, capsys):
    db = tmp_path / "k.db"
    _make_db(db)
    log = tmp_path / "a.jsonl"
    log.write_text(json.dumps({"task_id": "t2", "run_id": 2, "verdict": "block"}))
    cli.main(["alignment", "report", "--log", str(log), "--db", str(db)])
    assert "Wilson" in capsys.readouterr().out
    cli.main(["alignment", "report", "--log", str(log), "--db", str(db), "--json"])
    assert json.loads(capsys.readouterr().out)["block"]["tp"] == 1


def test_default_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "x.db"))
    assert ev.default_db_path() == tmp_path / "x.db"
    monkeypatch.delenv("HERMES_KANBAN_DB")
    assert ev.default_db_path().name == "kanban.db"
    assert ev.default_log_path().name == "alignment.jsonl"


def test_sidecar_alignment_route(tmp_path, monkeypatch):
    from tests.router.test_one_sidecar import _app, _auth
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "none.db"))
    app = _app(tmp_path)
    assert app.dispatch("GET", "/alignment", {})[0] == 401
    status, body = app.dispatch("GET", "/alignment", _auth())
    assert status == 200 and body["judged_runs"] == 0
    assert app.dispatch("POST", "/alignment", _auth(), body={})[0] == 405


def test_panel_counts_recent_and_breaker():
    from router import alignment_eval as ev
    entries = [
        {"ts": 1, "verdict": "continue"},
        {"ts": 2, "verdict": "block", "task_id": "t1"},
        {"ts": 3, "verdict": "continue", "failed_open": "timeout"},
        {"ts": 4, "verdict": "adjust"},
    ]
    br = {"ts": 5, "entries": [{"model_key": "p/m", "state": "OPEN", "cooldown_remaining_s": 30}]}
    p = ev.panel({"enabled": True, "mode": "shadow"}, entries, br, recent=2)
    assert p["mode"] == "shadow" and p["configured"] is True
    assert p["counts"] == {"continue": 1, "adjust": 1, "block": 1, "failed_open": 1}
    assert [e["ts"] for e in p["recent"]] == [4, 3]
    assert p["breaker"]["known"] and p["breaker"]["open"][0]["model_key"] == "p/m"


def test_panel_empty_is_unconfigured():
    from router import alignment_eval as ev
    p = ev.panel({}, [], None)
    assert p["configured"] is False and p["evaluations"] == 0
    assert p["breaker"] == {"known": False, "ts": None, "open": []}


def test_worst_verdict_keeps_more_severe():
    ents = [{"task_id": "t", "run_id": 1, "verdict": "block"},
            {"task_id": "t", "run_id": 1, "verdict": "adjust"}]
    assert ev.worst_verdicts(ents) == {("t", 1): "block"}


def test_kanban_cases_skips_unlabelled(tmp_path, monkeypatch):
    db = tmp_path / "k.db"
    _make_db(db)
    monkeypatch.setattr(ev, "label_outcome", lambda o: None)
    assert ev.kanban_cases(db) == []


def test_report_from_disk_copy_error(tmp_path, monkeypatch):
    db = tmp_path / "k.db"
    _make_db(db)
    def boom(*a, **k):
        raise sqlite3.Error("locked")
    monkeypatch.setattr(ev, "readonly_copy", boom)
    assert ev.report_from_disk(tmp_path / "none.jsonl", db)["judged_runs"] == 0


def test_panel_ignores_unknown_verdict():
    p = ev.panel({}, [{"verdict": "bogus"}], None)
    assert p["counts"] == {"continue": 0, "adjust": 0, "block": 0, "failed_open": 0}
