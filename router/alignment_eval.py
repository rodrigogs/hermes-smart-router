"""Alignment evaluation (F10): report with Wilson intervals and an offline replay harness.

Ground truth is the board itself (research doc section 7): a run that ended ``done`` is
aligned; ``blocked``/``timed_out``/``crashed``/``request_changes`` is drift. A false
positive is ``block``/``adjust`` on an aligned run; a false negative is ``continue`` on a
drifted one. Precision and recall are reported per verdict with a Wilson score interval,
because one person's volume is dozens of cards and a point estimate would mislead.

Replay never touches the live board: ``readonly_copy`` snapshots the kanban DB into a temp
file and every read goes through a ``mode=ro`` connection on that copy.
"""

from __future__ import annotations

import json
import math
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

VERDICTS = ("block", "adjust")
DRIFT_OUTCOMES = frozenset({"blocked", "timed_out", "crashed", "request_changes", "reclaimed"})
ALIGNED_OUTCOMES = frozenset({"completed", "done"})
_RANK = {"continue": 0, "adjust": 1, "block": 2}
CORPUS_PATH = Path(__file__).resolve().parent / "fixtures" / "alignment_corpus.json"


def wilson(successes: int, total: int, z: float = 1.96) -> Optional[Tuple[float, float]]:
    """Wilson score interval for a proportion; None when there is no data."""
    if total <= 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _rate(k: int, n: int) -> Dict[str, Any]:
    ci = wilson(k, n)
    return {
        "k": k, "n": n,
        "value": (k / n) if n else None,
        "wilson95": [round(ci[0], 4), round(ci[1], 4)] if ci else None,
    }


def label_outcome(outcome: Optional[str]) -> Optional[str]:
    """``aligned`` / ``drift`` from a run outcome; None while unlabelled (still running)."""
    if outcome in ALIGNED_OUTCOMES:
        return "aligned"
    if outcome in DRIFT_OUTCOMES:
        return "drift"
    return None


def worst_verdicts(entries: Iterable[Mapping[str, Any]]) -> Dict[Tuple[str, Any], str]:
    """One verdict per (task, run): the most severe non-failed-open evaluation."""
    out: Dict[Tuple[str, Any], str] = {}
    for e in entries:
        v = e.get("verdict")
        if v not in _RANK or e.get("failed_open"):
            continue
        key = (str(e.get("task_id")), e.get("run_id"))
        if key not in out or _RANK[v] > _RANK[out[key]]:
            out[key] = v
    return out


def score(pairs: Sequence[Tuple[str, str]]) -> Dict[str, Any]:
    """``pairs`` = (verdict, label) per run. Precision/recall per verdict, with Wilson."""
    drift = sum(1 for _, lab in pairs if lab == "drift")
    res: Dict[str, Any] = {"runs": len(pairs), "drift_runs": drift}
    for v in VERDICTS:
        tp = sum(1 for p, lab in pairs if p == v and lab == "drift")
        fp = sum(1 for p, lab in pairs if p == v and lab == "aligned")
        res[v] = {"precision": _rate(tp, tp + fp), "recall": _rate(tp, drift),
                  "tp": tp, "fp": fp}
    caught = sum(1 for p, lab in pairs if p != "continue" and lab == "drift")
    flagged = sum(1 for p, _ in pairs if p != "continue")
    res["any"] = {"precision": _rate(caught, flagged), "recall": _rate(caught, drift)}
    return res


def read_log(path: Path) -> List[Dict[str, Any]]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            obj = json.loads(ln)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


# --- kanban (read-only) ---------------------------------------------------------

def readonly_copy(db_path: Path, dest_dir: Optional[Path] = None) -> Path:
    """Snapshot the board DB (via sqlite backup, WAL-safe) and return the copy's path."""
    dest_dir = Path(dest_dir or tempfile.mkdtemp(prefix="alignment-replay-"))
    dest = dest_dir / "kanban-copy.db"
    src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
    try:
        dst = sqlite3.connect(dest)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
    return dest


def _ro(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def run_labels(db_path: Path) -> Dict[Tuple[str, Any], str]:
    """(task_id, run_id) -> aligned/drift for every ended run, read-only."""
    conn = _ro(db_path)
    try:
        rows = conn.execute("SELECT task_id, id, outcome FROM task_runs").fetchall()
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    out = {}
    for r in rows:
        lab = label_outcome(r["outcome"])
        if lab:
            out[(str(r["task_id"]), r["id"])] = lab
    return out


def report(entries: Sequence[Mapping[str, Any]], labels: Mapping[Tuple[str, Any], str]) -> Dict[str, Any]:
    verdicts = worst_verdicts(entries)
    pairs = [(v, labels[k]) for k, v in verdicts.items() if k in labels]
    res = score(pairs)
    res["judged_runs"] = len(verdicts)
    res["unlabelled_runs"] = len(verdicts) - len(pairs)
    res["failed_open"] = sum(1 for e in entries if e.get("failed_open"))
    res["evaluations"] = len(entries)
    return res


def render(rep: Mapping[str, Any]) -> str:
    def fmt(r: Mapping[str, Any]) -> str:
        if r["value"] is None:
            return "n/a (0/0)"
        lo, hi = r["wilson95"]
        return f"{r['value']:.2f} [{lo:.2f}-{hi:.2f}] ({r['k']}/{r['n']})"
    lines = [
        f"alignment report: {rep['judged_runs']} judged runs, {rep['runs']} labelled "
        f"({rep['drift_runs']} drift), {rep['unlabelled_runs']} unlabelled, "
        f"{rep['failed_open']} failed-open evaluations",
        "Wilson 95% intervals in brackets",
    ]
    for v in (*VERDICTS, "any"):
        lines.append(f"  {v:7s} precision {fmt(rep[v]['precision'])}  recall {fmt(rep[v]['recall'])}")
    return "\n".join(lines)


def kanban_cases(
    db_path: Path, fractions: Sequence[float] = (0.6, 0.85), limit: int = 200
) -> List[Dict[str, Any]]:
    """Corpus from finished cards, truncated to ``fractions`` of their comments.

    Works on a read-only copy; the live DB is only opened by ``readonly_copy``.
    """
    copy = readonly_copy(db_path)
    try:
        conn = _ro(copy)
        try:
            runs = conn.execute(
                "SELECT task_id, id, outcome FROM task_runs WHERE ended_at IS NOT NULL "
                "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            cases = []
            for r in runs:
                lab = label_outcome(r["outcome"])
                task = conn.execute("SELECT title, body FROM tasks WHERE id=?", (r["task_id"],)).fetchone()
                if not lab or not task:
                    continue
                comments = [dict(c) for c in conn.execute(
                    "SELECT author, body FROM task_comments WHERE task_id=? ORDER BY id", (r["task_id"],))]
                for f in fractions:
                    cut = comments[: max(1, int(len(comments) * f))] if comments else []
                    cases.append({
                        "id": f"{r['task_id']}#{r['id']}@{int(f * 100)}", "label": lab,
                        "card": {"title": task["title"], "body": task["body"] or ""},
                        "messages": [{"role": "assistant", "content": c["body"]} for c in cut],
                    })
            return cases
        finally:
            conn.close()
    finally:
        shutil.rmtree(copy.parent, ignore_errors=True)


def load_corpus(path: Path = CORPUS_PATH) -> List[Dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["cases"]


def replay(
    cases: Sequence[Mapping[str, Any]], judge_fn: Callable[[Mapping[str, Any]], str]
) -> Dict[str, Any]:
    """Run ``judge_fn(case) -> verdict`` over labelled cases and score it.

    A judge that raises or returns junk counts as ``continue`` (fail-open, as in production).
    """
    pairs = []
    for case in cases:
        try:
            v = judge_fn(case)
        except Exception:  # noqa: BLE001
            v = "continue"
        pairs.append((v if v in _RANK else "continue", case["label"]))
    res = score(pairs)
    res["cases"] = len(cases)
    res["judged_runs"] = len(cases)
    res["unlabelled_runs"] = 0
    res["failed_open"] = 0
    return res


def default_log_path() -> Path:
    try:
        from .paths import state_dir
    except ImportError:  # pragma: no cover - flat harness
        from router.paths import state_dir
    return state_dir() / "alignment.jsonl"


def default_db_path() -> Path:
    import os
    try:
        from .paths import hermes_root
    except ImportError:  # pragma: no cover - flat harness
        from router.paths import hermes_root
    env = os.environ.get("HERMES_KANBAN_DB")
    return Path(env) if env else hermes_root() / "kanban.db"


def report_from_disk(log_path: Optional[Path] = None, db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Report over alignment.jsonl, labelled from a read-only snapshot of the board."""
    entries = read_log(log_path or default_log_path())
    labels: Dict[Tuple[str, Any], str] = {}
    db = Path(db_path or default_db_path())
    if db.exists():
        try:
            copy = readonly_copy(db)
        except sqlite3.Error:
            copy = None
        if copy:
            try:
                labels = run_labels(copy)
            finally:
                shutil.rmtree(copy.parent, ignore_errors=True)
    return report(entries, labels)


def panel(config: Mapping[str, Any], entries: Sequence[Mapping[str, Any]],
          breaker: Optional[Mapping[str, Any]], recent: int = 10) -> Dict[str, Any]:
    """Read-only console panel: mode, last evaluations, verdict counts, judge breaker."""
    cfg = config if isinstance(config, Mapping) else {}
    counts = {"continue": 0, "adjust": 0, "block": 0, "failed_open": 0}
    for e in entries:
        if e.get("failed_open"):
            counts["failed_open"] += 1
        elif e.get("verdict") in counts:
            counts[e["verdict"]] += 1
    keys = ("ts", "task_id", "run_id", "rung", "pct", "mode", "verdict", "confidence",
            "failed_open", "judge", "action", "reasons")
    last = [{k: e.get(k) for k in keys} for e in list(entries)[-recent:]][::-1]
    br = breaker if isinstance(breaker, Mapping) else {}
    open_entries = [b for b in (br.get("entries") or []) if isinstance(b, dict)]
    return {
        "configured": bool(cfg.get("enabled", False)),
        "mode": cfg.get("mode", "shadow") if cfg else None,
        "evaluations": len(entries),
        "counts": counts,
        "recent": last,
        "breaker": {"known": bool(br), "ts": br.get("ts"), "open": open_entries},
    }


def panel_from_disk(config: Mapping[str, Any], log_path: Optional[Path] = None) -> Dict[str, Any]:
    try:
        from .paths import state_dir
    except ImportError:  # pragma: no cover - flat harness
        from router.paths import state_dir
    try:
        br = json.loads((state_dir() / "alignment-breaker.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        br = None
    return panel(config, read_log(log_path or default_log_path()), br)
