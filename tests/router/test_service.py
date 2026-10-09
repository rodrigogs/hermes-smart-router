"""Tests for the router service shared by web surfaces (read + write paths).

Every time-dependent case pins the clock — either by passing ``at`` or by
replacing ``service._utc_now`` — so no test here depends on the hour it runs at.
A test that read the real clock would be the defect the injected-clock design
exists to prevent.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from router import breaker as breaker_mod
from router.service import RouterService, resolve_compaction_auxiliary


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "router.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "enabled": True,
                "classifier": {"model": "judge", "provider": "judge-rail"},
                "fail_safe": {
                    "profile": "coder",
                    "model": "strong",
                    "provider": "safe-rail",
                },
                "blocklist": {
                    "manual_ban": [
                        {"model": "bad", "provider": "rail", "reason": "stalls"}
                    ],
                    "fallback_chain": ["strong"],
                    "auto_breaker": {"enabled": False},
                },
                "rules": [
                    {
                        "id": "hard-verbs",
                        "status": "stable",
                        "when": {"verb_class": {"eq": "hard"}},
                        "then": {"profile": "coder", "model": "T4"},
                    }
                ],
                "default": {"action": "classify"},
                "tiers": {
                    "T1": {"model": "tiny", "provider": "cheap"},
                    "T2": {"model": "small", "provider": "cheap"},
                    "T3": {"model": "medium", "provider": "strong-rail"},
                    "T4": {
                        "model": "strong",
                        "provider": "strong-rail",
                        "fallback": [
                            {"model": "backup", "provider": "backup-rail"}
                        ],
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_status_and_policy_are_read_only_snapshots(config_path):
    service = RouterService(config_path)

    status = service.status()
    policy = service.policy()

    # `warnings` and the provenance fields are purely ADDITIVE: pulling them out
    # must leave the exact prior response. These tier models are unknown to the
    # capability registry, which is advisory only — note `valid` stays True below.
    #
    # Asserted as "the registry finding is IN there", not as "every warning is that
    # finding": lint_warnings is an advisory channel designed to grow, so an `all()`
    # over it makes any new advisory rule fail this test, which is about read-only
    # snapshots and additivity rather than about the warning set. (It did exactly
    # that when the fallback_chain drift warning was added — this fixture's chain
    # lists one of its five tier members.)
    warnings = status.pop("warnings")
    assert any(
        "unknown to the capability registry" in warning for warning in warnings
    ), warnings

    # Provenance: a standalone RouterService reports config_mtime — the config is
    # re-read per request, so its age is knowable here. process_started_at and
    # code_mtime are stamped by the sidecar at boot and injected; without that
    # injection neither may appear, or the header would show ages it invented.
    config_mtime = status.pop("config_mtime")
    assert datetime.fromisoformat(config_mtime).tzinfo is not None, config_mtime
    assert "process_started_at" not in status
    assert "code_mtime" not in status

    # `configured_models` is additive in the same sense: it describes the INSTALL (the
    # agent's config.yaml), not this policy, and it is empty here because the isolated
    # HERMES_HOME this suite runs under has no agent config — which is the honest
    # answer, and the one the console renders as "no list to choose from".
    assert status.pop("configured_models") == []
    # Same additivity, same reason: the queues describe the INSTALL, and this suite runs
    # under an isolated HERMES_HOME with no agent config, so there are none to report.
    assert status.pop("agent_queues") == []

    assert status == {
        "valid": True,
        "validation_errors": [],
        "error_targets": [],
        "enabled": True,
        "rules_count": 1,
        "tiers": ["T1", "T2", "T3", "T4"],
        "classifier": {"model": "judge", "provider": "judge-rail"},
        "breaker_enabled": False,
    }
    assert policy["rules"][0]["id"] == "hard-verbs"
    assert policy["tiers"]["T4"]["fallback"] == [
        {"model": "backup", "provider": "backup-rail"}
    ]
    assert "api_key" not in json.dumps(policy)


def test_status_reports_sidecar_injected_provenance(config_path):
    """The sidecar stamps boot provenance onto the service; /status must echo it.

    The mtimes are captured at sidecar boot (one_sidecar._code_mtime), not
    computed per request — the whole point is that a process that booted before
    the last edit keeps reporting the code it actually loaded.
    """
    service = RouterService(config_path)
    service._process_started_at = "2026-08-18T23:40:44+00:00"
    service._code_mtime = "2026-08-18T19:39:42+00:00"

    status = service.status()
    assert status["process_started_at"] == "2026-08-18T23:40:44+00:00"
    assert status["code_mtime"] == "2026-08-18T19:39:42+00:00"
    assert "config_mtime" in status


def test_status_omits_config_mtime_when_config_is_unstatable(config_path):
    """A config that cannot be stat'ed degrades, it does not crash /status.

    config_mtime is stamped from the file's mtime; when the file is gone
    (deleted after boot — an operator swapping the live policy) the field is
    omitted and the endpoint still answers. This OSError path was covered only
    by a fastapi-gated dashboard-parity test that skips on CI, so the 100%
    gate leaned on an environment CI does not have — the same shape of defect
    as the router.yaml provenance, on the same /status payload.
    """
    service = RouterService(config_path)
    config_path.unlink()  # vanished between boot and /status

    status = service.status()
    assert "config_mtime" not in status
    assert "process_started_at" not in status
    assert "valid" in status


def test_explain_is_deterministic_and_never_calls_classifier(config_path):
    service = RouterService(config_path)

    hard = service.explain("Debug a race condition")
    uncertain = service.explain("Summarize this note")

    assert hard["mode"] == "deterministic_dry_run"
    assert hard["requires_classifier"] is False
    assert hard["decision"]["cause"] == "hard_rule"
    assert hard["decision"]["output"]["fallback"][0]["provider"] == "backup-rail"
    assert uncertain["requires_classifier"] is True
    assert uncertain["decision"]["output"] == {"action": "classify"}


def test_blocklist_and_invalid_config_are_explicit(config_path, tmp_path):
    service = RouterService(config_path)
    blocklist = service.blocklist()
    assert blocklist["manual_bans"][0]["model"] == "bad"
    assert blocklist["fallback_chain"] == ["strong"]
    assert blocklist["breaker_enabled"] is False

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("enabled: [", encoding="utf-8")
    broken = RouterService(invalid)
    status = broken.status()
    assert status["valid"] is False
    assert status["enabled"] is False
    assert status["validation_errors"]


def test_blocklist_publishes_the_breakers_thresholds_and_its_criterion(config_path):
    """The half of the breaker a console could not compute for itself.

    ``breaker_cooldowns`` says who is blocked and for how long. It cannot answer
    the two questions that come next — "how close to tripping is a healthy rail?"
    and "which errors even count?" — because the threshold and the weight table
    live in config and in code, and this route published neither. A screen showing
    ``failure_count: 3`` with no threshold beside it is a number the operator has
    to read the source to interpret, and a screen that GUESSED the threshold would
    be a second authority on it (card t_1c6a002d).
    """
    policy = RouterService(config_path).blocklist()["auto_breaker"]

    # The thresholds in force. This fixture declares no auto_breaker knobs, so
    # every value here is a shipped DEFAULT — which is the point: an absent key
    # does not mean an absent policy, and reporting the raw YAML would have shown
    # an operator an empty one while a real one runs.
    assert policy["threshold"] == 5
    assert policy["window_seconds"] == 600
    assert policy["base_cooldown_seconds"] == 60
    assert policy["max_cooldown_seconds"] == 900
    assert policy["backoff_multiplier"] == 2.0

    # The criterion, read off the same table `record` uses — not a copy that could
    # drift from it. A quota exhaustion trips on its own (10 >= 5); a nonzero exit
    # needs five, and those are opposite diagnoses of the same `failure_count`.
    assert policy["failure_weights"] == dict(breaker_mod.FAILURE_WEIGHTS)
    assert policy["failure_weights"]["quota_exhausted"] >= policy["threshold"]
    assert policy["failure_weights"]["nonzero_exit"] < policy["threshold"]
    assert policy["default_weight"] == breaker_mod.DEFAULT_WEIGHT

    # Reported even though the breaker is OFF here: `breaker_enabled` is the
    # authority on whether any of it acts, and an operator deciding to turn it on
    # has to see what they would be turning on.
    assert RouterService(config_path).blocklist()["breaker_enabled"] is False

    # A COPY, not the module table. Found by mutation: handing out
    # `FAILURE_WEIGHTS` itself means any consumer that edits the payload edits the
    # criterion `record` consults — a routing change made by a reader. Same
    # invariant `capabilities_for` already holds over the registry.
    before = dict(breaker_mod.FAILURE_WEIGHTS)
    policy["failure_weights"]["ttfb_stall"] = 999
    policy["failure_weights"]["invented_kind"] = 7
    assert breaker_mod.FAILURE_WEIGHTS == before, (
        "the served weights must not be the table the breaker scores with"
    )
    assert RouterService(config_path).blocklist()[
        "auto_breaker"]["failure_weights"] == before


def test_the_breaker_policy_reports_what_the_operator_actually_set(tmp_path):
    """Declared knobs win over the defaults, and only the declared ones move."""
    path = tmp_path / "router.yaml"
    path.write_text(yaml.safe_dump({
        "enabled": True,
        "classifier": {"model": "glm-4.6", "provider": "zai"},
        "fail_safe": {"profile": "coder", "model": "glm-4.6", "provider": "zai"},
        "blocklist": {
            "manual_ban": [], "fallback_chain": [],
            "auto_breaker": {"enabled": True, "threshold": 9,
                             "window_seconds": 120},
        },
        "rules": [],
        "default": {"profile": "coder", "model": "T1"},
        "tiers": {name: {"model": "glm-4.6", "provider": "zai"}
                  for name in ("T1", "T2", "T3", "T4")},
    }, sort_keys=False), encoding="utf-8")

    served = RouterService(path).blocklist()
    assert served["breaker_enabled"] is True
    policy = served["auto_breaker"]
    assert policy["threshold"] == 9, "the operator's number, not the default 5"
    assert policy["window_seconds"] == 120
    # Untouched knobs keep the shipped values rather than becoming absent or zero.
    assert policy["base_cooldown_seconds"] == 60
    assert policy["max_cooldown_seconds"] == 900


def test_the_breaker_policy_survives_a_malformed_auto_breaker_block(tmp_path):
    """A non-mapping `auto_breaker` must report the policy that really runs.

    `Blocklist` already refuses to read a scalar as config and falls back to the
    defaults; the route has to report THOSE, because they are what `record` will
    use. Reporting nothing here would be the shape of the defect this whole card is
    about: a surface that goes quiet exactly when the config is wrong.
    """
    path = tmp_path / "router.yaml"
    path.write_text(yaml.safe_dump({
        "enabled": True,
        "classifier": {"model": "glm-4.6", "provider": "zai"},
        "fail_safe": {"profile": "coder", "model": "glm-4.6", "provider": "zai"},
        "blocklist": {"manual_ban": [], "fallback_chain": [],
                      "auto_breaker": "sim"},
        "rules": [],
        "default": {"profile": "coder", "model": "T1"},
        "tiers": {name: {"model": "glm-4.6", "provider": "zai"}
                  for name in ("T1", "T2", "T3", "T4")},
    }, sort_keys=False), encoding="utf-8")

    served = RouterService(path).blocklist()
    assert served["breaker_enabled"] is False, "a scalar cannot enable it"
    assert served["auto_breaker"]["threshold"] == 5
    assert served["auto_breaker"]["failure_weights"], "the criterion still ships"


def test_explain_rejects_empty_or_oversized_tasks(config_path):
    service = RouterService(config_path, max_task_chars=12)
    with pytest.raises(ValueError, match="required"):
        service.explain("   ")
    with pytest.raises(ValueError, match="12 characters"):
        service.explain("x" * 13)


def test_scalar_and_invalid_policy_cannot_be_explained(tmp_path):
    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("just-a-string", encoding="utf-8")
    scalar_service = RouterService(scalar)
    assert scalar_service.status()["validation_errors"] == ["router config root must be a mapping"]

    invalid = tmp_path / "incomplete.yaml"
    invalid.write_text("enabled: true", encoding="utf-8")
    with pytest.raises(ValueError, match="policy is invalid"):
        RouterService(invalid).explain("Describe a task")


def test_plan_previews_without_writing(config_path):
    service = RouterService(config_path)
    before = config_path.read_bytes()

    plan = service.plan({"default": {"action": "T1"}})

    assert plan["valid"] is True
    assert plan["base_hash"]
    assert plan["policy"]["default"] == {"action": "T1"}
    assert "default" in plan["diff"]
    # plan() must be pure: the file is byte-identical afterwards.
    assert config_path.read_bytes() == before


def test_plan_serves_the_before_document_so_a_client_can_diff_like_for_like(config_path):
    """``current`` is the file as parsed, not the projection ``/policy`` serves.

    The console renders its own structural diff — that is what lets it say "9 rules
    replace 8", which the YAML text diff cannot — and it had no BEFORE side except the
    ``/policy`` projection. That projection omits ``enabled``, ``blocklist`` and
    ``shadow`` and always carries a ``compaction`` key, so four keys read as changes on
    EVERY plan and the count said five keys changed when one did. Measured on the docker
    stack 2026-09-03 while editing the classifier's chain.
    """
    service = RouterService(config_path)
    on_disk = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    plan = service.plan({"default": {"action": "T1"}})

    assert plan["current"] == on_disk, "the BEFORE side is the file, verbatim"
    assert plan["current"] != plan["policy"], (
        "non-vacuity: this patch really does change something, so before != after"
    )
    # Every key the file has is on BOTH sides now, so a structural diff of the two
    # reports only what the patch actually changed.
    assert set(plan["current"]) == set(plan["policy"]), (
        "same key set on both sides — a diff can only report values, never phantom keys"
    )
    # And the projection is a genuinely different shape, which is why it was the wrong
    # BEFORE. `enabled` and `blocklist` are in this file and NOT in the projection, so
    # each read as an addition on every plan. (`shadow` does the same on installs that
    # declare it; this fixture does not, so it is not asserted here.)
    served = service.policy()
    for key in ("enabled", "blocklist"):
        assert key in plan["current"], f"non-vacuity: this file declares {key}"
        assert key not in served, (
            f"{key} is absent from the projection — diffed against the merged document "
            f"it read as an addition on every plan"
        )
    assert "compaction" in served and served["compaction"] is None
    assert "compaction" not in plan["current"], (
        "the projection invents a compaction key the file does not have; that is what "
        "used to read as a removal"
    )


def test_plan_rejects_unknown_tier_and_ignores_non_allowlisted_keys(config_path):
    service = RouterService(config_path)

    invalid = service.plan({"rules": [
        {"id": "x", "when": {"verb_class": {"eq": "hard"}}, "then": {"model": "T9"}}
    ]})
    assert invalid["valid"] is False
    assert any("T9" in e for e in invalid["errors"])

    # A key outside the hot allowlist is silently dropped, never written.
    plan = service.plan({"secrets": {"api_key": "leak"}, "default": {"action": "T2"}})
    assert "secrets" not in plan["policy"]
    assert plan["policy"]["default"] == {"action": "T2"}


def test_plan_flags_structurally_broken_fail_safe(config_path):
    service = RouterService(config_path)
    # rules.lint never inspects fail_safe; blanking its model must still be
    # caught by the structural guard before it reaches the hot file.
    plan = service.plan({"fail_safe": {"model": ""}})
    assert plan["valid"] is False
    assert any("fail_safe.model" in e for e in plan["errors"])


def test_apply_commits_on_matching_hash_and_reverts(config_path):
    service = RouterService(config_path)
    plan = service.plan({"tiers": {"T4": {"model": "stronger", "provider": "strong-rail"}}})

    result = service.apply(plan["base_hash"], plan["policy"])
    assert result["ok"] is True

    # Hot-reload: a fresh read reflects the change, and key order is preserved.
    reloaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert reloaded["tiers"]["T4"]["model"] == "stronger"
    assert list(reloaded.keys())[0] == "enabled"

    backup = config_path.with_suffix(config_path.suffix + ".bak")
    assert backup.exists()

    revert = service.apply_revert()
    assert revert["ok"] is True
    assert yaml.safe_load(config_path.read_text())["tiers"]["T4"]["model"] == "strong"


def test_apply_returned_hash_matches_next_plan_no_false_409(config_path):
    """The base_hash apply returns must equal the next plan's base_hash.

    Guards against hashing a re-read (which can differ from the bytes written by
    a trailing newline/formatting nudge and turn every follow-up apply into a
    false 409 conflict).
    """
    service = RouterService(config_path)
    plan1 = service.plan({"default": {"action": "T1"}})
    applied = service.apply(plan1["base_hash"], plan1["policy"])
    assert applied["ok"] is True

    plan2 = service.plan({"default": {"action": "T2"}})
    assert plan2["base_hash"] == applied["base_hash"]
    # And a second apply against that hash commits cleanly (no false conflict).
    assert service.apply(plan2["base_hash"], plan2["policy"])["ok"] is True


def test_revert_restores_byte_identical_original(config_path):
    """apply → revert returns the file to the exact original bytes."""
    service = RouterService(config_path)
    original = config_path.read_bytes()
    plan = service.plan({"tiers": {"T4": {"model": "x", "provider": "y"}}})
    service.apply(plan["base_hash"], plan["policy"])
    assert config_path.read_bytes() != original
    revert = service.apply_revert()
    assert revert["ok"] is True
    assert config_path.read_bytes() == original
    assert revert["base_hash"] == RouterService._hash_bytes(original)


def test_apply_refuses_on_stale_hash_and_leaves_file_untouched(config_path):
    service = RouterService(config_path)
    before = config_path.read_bytes()

    result = service.apply("deadbeef" * 8, {"default": {"action": "T1"}})
    assert result["ok"] is False
    assert result["conflict"] is True
    assert result["base_hash"] == RouterService._hash_bytes(before)
    assert config_path.read_bytes() == before


def test_apply_refuses_lint_invalid_merge(config_path):
    service = RouterService(config_path)
    # A rule referencing a nonexistent tier fails lint; apply must refuse.
    plan = service.plan({"rules": [
        {"id": "bad", "when": {"verb_class": {"eq": "hard"}}, "then": {"model": "T9"}}
    ]})
    assert plan["valid"] is False
    before = config_path.read_bytes()
    result = service.apply(plan["base_hash"], plan["policy"])
    assert result["ok"] is False
    assert result["errors"]
    assert config_path.read_bytes() == before  # refused write leaves file intact


def test_apply_lists_replace_wholesale(config_path):
    """Sending a shorter rules list must DELETE rules, not union them."""
    service = RouterService(config_path)
    plan = service.plan({"rules": []})
    service.apply(plan["base_hash"], plan["policy"])
    assert yaml.safe_load(config_path.read_text())["rules"] == []


def test_apply_is_serialized_under_concurrency(config_path):
    """Two concurrent applies with the SAME base_hash: exactly one commits."""
    service = RouterService(config_path)
    base = RouterService._hash_bytes(config_path.read_bytes())
    results = []
    barrier = threading.Barrier(2)

    def worker(model_name):
        barrier.wait()
        results.append(service.apply(base, {"tiers": {"T4": {"model": model_name, "provider": "strong-rail"}}}))

    threads = [threading.Thread(target=worker, args=(n,)) for n in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    oks = [r for r in results if r.get("ok")]
    conflicts = [r for r in results if r.get("conflict")]
    # The lock forces serialization: the first wins, the second sees drift.
    assert len(oks) == 1
    assert len(conflicts) == 1


def test_apply_revert_without_snapshot_is_explicit(config_path):
    service = RouterService(config_path)
    assert service.apply_revert() == {"ok": False, "error": "no snapshot"}


def test_fail_safe_guard_covers_mapping_and_fallback_shape(config_path):
    service = RouterService(config_path)
    # fail_safe present but not a mapping.
    bad_type = service.plan({"fail_safe": []})
    assert any("must be a mapping" in e for e in bad_type["errors"])
    # fallback present but not a list.
    bad_fallback = service.plan({"fail_safe": {"fallback": "nope"}})
    assert any("fail_safe.fallback must be a list" in e for e in bad_fallback["errors"])


def test_plan_and_apply_reject_non_mapping_changes(config_path):
    service = RouterService(config_path)
    with pytest.raises(ValueError, match="changes must be a mapping"):
        service.plan(["not", "a", "dict"])
    with pytest.raises(ValueError, match="changes must be a mapping"):
        service.apply("hash", ["nope"])


def test_plan_reports_unreadable_config(tmp_path):
    missing = tmp_path / "gone.yaml"
    plan = RouterService(missing).plan({"default": {"action": "T1"}})
    assert plan["valid"] is False
    assert any("could not read router config" in e for e in plan["errors"])


def test_apply_reports_unreadable_config(tmp_path):
    missing = tmp_path / "gone.yaml"
    result = RouterService(missing).apply("hash", {"default": {"action": "T1"}})
    assert result["ok"] is False
    assert any("could not read router config" in e for e in result["errors"])


def test_apply_reports_malformed_yaml_after_hash_match(tmp_path):
    """A config that hashes fine but parses to a non-mapping is a parse error."""
    path = tmp_path / "router.yaml"
    path.write_text("just-a-scalar", encoding="utf-8")
    service = RouterService(path)
    base = RouterService._hash_bytes(path.read_bytes())
    result = service.apply(base, {"default": {"action": "T1"}})
    assert result["ok"] is False
    assert any("could not parse router config" in e for e in result["errors"])


def test_apply_write_failure_leaves_config_and_backup_consistent(config_path, monkeypatch):
    """If the config write fails after the .bak snapshot, os.replace atomicity
    guarantees the config stays at the OLD bytes — which is exactly what .bak
    holds — so a later revert restores a state that matches, never a mismatch.
    """
    import router.service as service_mod

    service = RouterService(config_path)
    original = config_path.read_bytes()
    plan = service.plan({"default": {"action": "T1"}})

    # Fail ONLY the main-config replace (the second _atomic_write_bytes call),
    # after the .bak snapshot has been written.
    calls = {"n": 0}
    real_replace = service_mod.os.replace

    def flaky_replace(src, dst):
        calls["n"] += 1
        if str(dst) == str(config_path):
            raise OSError("write interrupted")
        return real_replace(src, dst)

    monkeypatch.setattr(service_mod.os, "replace", flaky_replace)
    with pytest.raises(OSError, match="write interrupted"):
        service.apply(plan["base_hash"], plan["policy"])

    # Config untouched (atomic replace never happened) and the .bak equals it.
    assert config_path.read_bytes() == original
    backup = config_path.with_suffix(config_path.suffix + ".bak")
    assert backup.read_bytes() == original


def _seed_traces(tmp_path, monkeypatch, entries, backups=None):
    """Write route traces to a temp HERMES_HOME state dir; return the base path."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from router.durable_decision_log import routes_path
    base = routes_path()
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
    if backups:
        for suffix, backup_entries in backups.items():
            p = base.with_suffix(base.suffix + suffix)
            p.write_text("".join(json.dumps(e) + "\n" for e in backup_entries), encoding="utf-8")
    return base


def test_routes_lists_recent_first_with_projection(tmp_path, monkeypatch, config_path):
    _seed_traces(tmp_path, monkeypatch, [
        {"ts": 1.0, "cause": "hard_rule", "task": "a", "output": {"model": "m1"}},
        {"ts": 2.0, "cause": "classifier", "task": "b", "output": {"model": "m2"}},
    ])
    svc = RouterService(config_path)
    result = svc.routes()
    assert result["count"] == 2
    assert result["trace_path"].endswith("routes.jsonl")
    # Most recent first.
    assert result["routes"][0]["cause"] == "classifier"
    # These entries predate `attempted_model` (every historical trace does), so
    # the declared primary IS the honest answer for what ran — and both fields
    # say so rather than one of them going blank.
    assert result["routes"][0]["model"] == "m2"
    assert result["routes"][0]["declared_model"] == "m2"
    assert result["routes"][1]["task"] == "a"


def test_routes_honors_limit_and_bad_limit_falls_back(tmp_path, monkeypatch, config_path):
    _seed_traces(tmp_path, monkeypatch, [
        {"ts": float(i), "cause": "classifier", "task": f"t{i}", "output": {"model": f"m{i}"}}
        for i in range(5)
    ])
    svc = RouterService(config_path)
    assert len(svc.routes(limit=2)["routes"]) == 2
    # A non-numeric limit falls back to the default, not a crash.
    assert svc.routes(limit="oops")["count"] == 5


def test_routes_skips_corrupt_lines_and_missing_file(tmp_path, monkeypatch, config_path):
    # This test asserts the HERMES_HOME-derived path, so it opts out of the
    # blanket HERMES_ROUTE_TRACE_FILE guard in conftest.
    monkeypatch.delenv("HERMES_ROUTE_TRACE_FILE", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from router.durable_decision_log import routes_path
    base = routes_path()
    base.parent.mkdir(parents=True, exist_ok=True)
    base.write_text(
        json.dumps({"ts": 1.0, "cause": "hard_rule", "output": {"model": "ok"}}) + "\n"
        + "\n"  # blank line — skipped
        + "   \n"  # whitespace-only — skipped
        + "{ this is not json\n"
        + json.dumps("a-string-not-a-dict") + "\n",
        encoding="utf-8",
    )
    svc = RouterService(config_path)
    assert svc.routes()["count"] == 1  # only the valid dict line

    # Missing file → empty, never raises.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "empty"))
    assert svc.routes()["count"] == 0
    assert svc.routes()["routes"] == []


def test_route_by_id_returns_full_entry_with_steps(tmp_path, monkeypatch, config_path):
    _seed_traces(tmp_path, monkeypatch, [
        {"ts": 7.0, "cause": "classifier", "task": "x", "output": {"model": "m"},
         "steps": [{"stage": "blocklist"}, {"stage": "classifier"}]},
    ])
    svc = RouterService(config_path)
    listed = svc.routes()["routes"][0]
    full = svc.route(listed["id"])
    assert full is not None
    assert full["steps"][1]["stage"] == "classifier"
    assert svc.route("nonexistent-id") is None
    assert svc.route("") is None


def test_routes_backfills_from_rotated_backup(tmp_path, monkeypatch, config_path):
    # Current file has 1, backup .1 has 2 → limit 3 back-fills across rotation.
    _seed_traces(
        tmp_path, monkeypatch,
        [{"ts": 3.0, "cause": "classifier", "task": "new", "output": {}}],
        backups={".1": [
            {"ts": 1.0, "cause": "hard_rule", "task": "old1", "output": {}},
            {"ts": 2.0, "cause": "hard_rule", "task": "old2", "output": {}},
        ]},
    )
    svc = RouterService(config_path)
    result = svc.routes(limit=3)
    assert result["count"] == 3
    assert result["routes"][0]["task"] == "new"  # most recent first


def test_routes_skips_absent_backup_in_chain(tmp_path, monkeypatch, config_path):
    # A gap in the backup chain (.1 absent, .2 present) exercises the
    # missing-file continue without raising.
    base = _seed_traces(
        tmp_path, monkeypatch,
        [{"ts": float(i), "cause": "classifier", "task": f"c{i}", "output": {}} for i in range(3)],
        backups={".2": [{"ts": 99.0, "cause": "hard_rule", "task": "deep", "output": {}}]},
    )
    assert not base.with_suffix(base.suffix + ".1").exists()  # gap in the chain
    svc = RouterService(config_path)
    result = svc.routes(limit=2)
    assert len(result["routes"]) == 2  # limit caps the projection
    # The reader walks past the absent .1 (continue) into .2 without raising;
    # count reflects all readable entries across the chain.
    assert result["count"] == 4  # 3 current + 1 from .2
    assert any(r["task"] == "deep" for r in svc.routes(limit=100)["routes"])


# ---------------------------------------------------------------------------
# routes() must name the elo that RAN — the recurring defect, on the primary
# operator surface. Every test below drives a REAL routing decision through
# adapter.route (the running path) and reads it back through routes() (the
# displaying path), then asserts the two AGREE. Asserting either side alone is
# what let this diverge: the writer was fixed and the reader was not.
# ---------------------------------------------------------------------------

# The live policy — the same one production routes on. `_seed_live_router_config`
# (tests/conftest.py) guarantees it exists in a fresh checkout.
LIVE_POLICY = Path(__file__).resolve().parents[2] / "router.yaml"

# A vision turn. While the T2 primary could not see, the capability filter
# promoted a fallback hop and the DECLARED tier primary was not what ran — that
# gap is the whole subject of these tests.
# Since 2026-08-27 the shipped primary is natively multimodal, so on the live
# policy this turn has no gap left: nothing is filtered and a screenshot stays on
# the plan rail. Good for the bill, fatal for a test that borrowed the gap from
# the roster, so the gap is now built by `_vision_gap_policy` below.
VISION_TASK = "Look at this screenshot and tell me why the layout breaks"

#: The text-only plan model T2 declared until then. A real registry entry with
#: ``vision: False``, so the filter drops it for an image turn exactly as before.
VISION_BLIND_PRIMARY = "glm-5.3"

# 07:00 UTC Monday — inside the shipped peak windows, so the time layer is live
# for these turns rather than a no-op. Injected, never read from the wall clock.
PEAK_CLOCK = datetime(2026, 8, 17, 7, 0, tzinfo=timezone.utc)


def _executor_targets(routed):
    """The attempt order the delegate_profile executor really derives.

    The plugin's own ``_routed_targets``, loaded from ``__init__.py`` rather than
    re-implemented here: a mirror of "which target runs first" written inside the
    test is exactly the second opinion this whole family of defects is made of.
    """
    import importlib.util
    import sys

    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location(
        "delegate_profile_executor_view", root / "__init__.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._routed_targets(routed)


def _vision_gap_policy(tmp_path):
    """The live policy written back out with a T2 primary that cannot see.

    These tests are about the gap between the model a surface DISPLAYS and the
    model the executor RUNS, and a vision turn against a text-only primary is the
    shape that guarantees one. The shipped roster stopped supplying it — which is
    an improvement, not a regression — so it is constructed here rather than
    borrowed. Written to a file because RouterService reads a path, and the reader
    under test has to load the same policy the router routed on.
    """
    config = yaml.safe_load(LIVE_POLICY.read_text(encoding="utf-8"))
    config["tiers"]["T2"]["model"] = VISION_BLIND_PRIMARY
    path = tmp_path / "router-vision-gap.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _route_and_list(task, *, policy=None, **kwargs):
    """Route ``task`` for real, then read the decision back off routes().

    Returns ``(result, entry, listed)`` — the executor's decision, the persisted
    trace entry, and the projection routes() serves for that same decision. The
    autouse ``_isolate_route_trace`` fixture points the trace file at tmp_path, so
    the writer and this reader converge on one temp file.

    ``policy`` defaults to the live file; both sides always read the SAME path, so
    a caller cannot accidentally route on one policy and list from another.
    """
    from router.durable_decision_log import DurableDecisionLog
    from router.adapter import route as adapter_route

    policy = policy or LIVE_POLICY
    config = yaml.safe_load(policy.read_text(encoding="utf-8"))
    dlog = DurableDecisionLog()
    result = adapter_route(task, config, decision_log=dlog, now=PEAK_CLOCK, **kwargs)
    listed = RouterService(policy).routes()["routes"][0]
    return result, dlog.tail(1)[0], listed


def test_routes_names_the_elo_that_ran_and_keeps_the_tier_identity(tmp_path):
    """The agreement itself: routes()'s model IS the executor's first target.

    routes() labelled every decision with ``output.model``, the DECLARED tier
    primary, which after a capability filter, a time cap, a shuffle or a blocklist
    veto is not the elo the executor dispatches. Measured on the then-shipped policy
    at 07:00Z: the Decisions tab said the turn below ran ``glm-5.3`` — a model that
    cannot see, and was never attempted — while ``gpt-5.6-luna`` served it. The
    writer side (``output.attempted_model``) was fixed on this branch; this reader
    was not, so the branch is what made the two disagree.
    """
    from router.decision_log import attempted_head_of

    result, entry, listed = _route_and_list(
        VISION_TASK, policy=_vision_gap_policy(tmp_path)
    )

    executor_first = _executor_targets(result)[0]
    # The precondition, asserted so this test can never pass vacuously: if the
    # policy stops promoting a fallback hop here, declared == attempted and there
    # is no disagreement left to catch.
    assert executor_first[0] != result["model"], (
        "this task no longer separates the declared primary from the attempted "
        "head, so the agreement below proves nothing"
    )

    # The two sides, on the same decision: what the operator is SHOWN, and what
    # the executor RUNS.
    assert (listed["model"], listed["provider"]) == executor_first
    # ...and the surface agrees with the single accessor the writer persists
    # through, rather than having its own reading of "the head".
    assert (listed["model"], listed["provider"]) == attempted_head_of(entry)
    # The tier identity is kept as secondary, not dropped: it is what ties the
    # decision back to the tier the rule chose.
    assert listed["declared_model"] == result["model"]
    assert listed["rule_id"] == entry["rule_id"]


@pytest.mark.parametrize("task", [
    VISION_TASK,
    "Rename getCwd in src/utils.py",
    "Debug a race condition in the user cache",
    "an entirely ambiguous request",
])
def test_every_listed_decision_agrees_with_what_the_executor_would_run(task, tmp_path):
    """One property over four routing paths, rather than four literals.

    A filtered chain, an unfiltered one, a hard rule and a classifier answer all
    have to project the same way: the elo that ran, whether or not it happens to
    equal the tier primary. The un-filtered cases are the ones that make the
    difference invisible, which is why they are in here too.
    """
    result, _entry, listed = _route_and_list(
        task, classify_fn=lambda _t, _f: {"tier": "T3", "confidence": "high"},
    )
    assert (listed["model"], listed["provider"]) == _executor_targets(result)[0], (
        f"the Decisions tab and the executor disagree for {task!r}"
    )


def test_a_denied_decision_names_no_model_rather_than_the_one_it_refused(tmp_path):
    """A veto that attempts nothing must not be labelled with the elo it blocked.

    ``output.model`` survives on a denial (it is the decision that WOULD have run),
    so projecting it would have the tab reporting a model for a turn nothing was
    dispatched for. ``attempted_head_of`` answers "" for an empty chain and that
    answer has to reach the surface intact.
    """
    from router.blocklist import Blocklist
    from router.durable_decision_log import DurableDecisionLog
    from router.adapter import route as adapter_route

    config = yaml.safe_load(LIVE_POLICY.read_text(encoding="utf-8"))
    chosen = adapter_route(VISION_TASK, config, now=PEAK_CLOCK)["model"]
    banned = yaml.safe_load(LIVE_POLICY.read_text(encoding="utf-8"))
    chain = [chosen, "gpt-5.6-luna", "deepseek-v4-flash", "glm-5.3"]
    banned["blocklist"]["fallback_chain"] = chain
    banned["blocklist"]["manual_ban"] = [
        {"model": model, "provider": "", "reason": "test-ban"} for model in chain
    ]

    dlog = DurableDecisionLog()
    result = adapter_route(VISION_TASK, banned, blocklist=Blocklist(banned),
                           decision_log=dlog, now=PEAK_CLOCK)
    assert result["deny"] is True
    assert _executor_targets(result) == [], "a denial attempts nothing"

    listed = RouterService(LIVE_POLICY).routes()["routes"][0]
    assert listed["cause"] == "blocklist_veto"
    assert (listed["model"], listed["provider"]) == ("", "")


def test_routes_survives_an_undecodable_byte_and_still_serves_the_readable_lines(
    tmp_path, monkeypatch, config_path,
):
    """One invalid byte must cost that LINE, not the whole read path.

    ``read_text`` raises ``UnicodeDecodeError`` — a ``ValueError``, so the reader's
    ``except OSError`` never saw it — and the trace is appended to by another
    process, so a torn multi-byte write is a thing that happens. Unhandled it
    escaped both routes() and route(), which the sidecar answers with an aborted
    connection and a Decisions tab that stays dead until someone finds the file.
    """
    monkeypatch.setenv("HERMES_ROUTE_TRACE_FILE", str(tmp_path / "routes.jsonl"))
    from router.durable_decision_log import routes_path
    path = routes_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    good_first = json.dumps({"ts": 1.0, "cause": "hard_rule", "task": "first",
                             "output": {"model": "m1"}}).encode("utf-8")
    torn = b'{"ts": 2.0, "cause": "classifier", "task": "\xff\xfe torn"}'
    good_last = json.dumps({"ts": 3.0, "cause": "classifier", "task": "last",
                            "output": {"model": "m3"}}).encode("utf-8")
    path.write_bytes(good_first + b"\n" + torn + b"\n" + good_last + b"\n")

    svc = RouterService(config_path)
    result = svc.routes()  # must not raise

    assert [r["task"] for r in result["routes"]] == ["last", "first"]
    assert result["count"] == 2
    # ...and route() reads the same file through the same reader, so the entry
    # AFTER the corruption is still fetchable by id.
    assert svc.route(result["routes"][0]["id"])["task"] == "last"


def test_validate_fail_safe_is_noop_when_absent():
    """No fail_safe key -> nothing to validate."""
    assert RouterService._validate_fail_safe({"default": {}}) == []


def test_atomic_write_cleans_up_temp_on_failure(config_path, monkeypatch):
    """If os.replace fails, the temp file is unlinked and the error propagates."""
    import router.service as service_mod

    monkeypatch.setattr(
        service_mod.os, "replace",
        lambda *_a: (_ for _ in ()).throw(OSError("disk full")),
    )
    unlinked = {"n": 0}
    real_unlink = service_mod.os.unlink
    monkeypatch.setattr(
        service_mod.os, "unlink",
        lambda p: (unlinked.__setitem__("n", unlinked["n"] + 1), real_unlink(p))[1],
    )
    with pytest.raises(OSError, match="disk full"):
        RouterService._atomic_write_bytes(config_path, b"data")
    assert unlinked["n"] == 1  # temp file was cleaned up


def test_atomic_write_swallows_unlink_error_during_cleanup(config_path, monkeypatch):
    """If cleanup unlink ALSO fails, the original error still propagates."""
    import router.service as service_mod

    monkeypatch.setattr(
        service_mod.os, "replace",
        lambda *_a: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(
        service_mod.os, "unlink",
        lambda *_a: (_ for _ in ()).throw(OSError("cleanup failed")),
    )
    with pytest.raises(OSError, match="disk full"):
        RouterService._atomic_write_bytes(config_path, b"data")


def test_policy_references_skips_malformed_declarations():
    """_policy_references ignores non-dict entries, blank/typed model/provider,
    non-dict tiers, and maps dict-form fallback-chain entries."""
    config = {
        "classifier": "not-a-dict",                       # add() early-return (non-dict)
        "tiers": {
            "T1": {"model": "", "provider": "x"},          # blank model -> skipped
            "T2": {"model": "m", "provider": 5},           # non-str provider -> skipped
            "T3": {"model": "good", "provider": "rail",
                   "fallback": [{"model": "fb", "provider": "fbrail"}, "loose"]},
            "T4": "not-a-dict-tier",                        # non-dict tier -> add() skips, no fallback recurse
        },
        "fail_safe": {"model": "fs", "provider": "fsrail",
                      "fallback": [{"model": "fsfb", "provider": "fsfbrail"}]},
    }
    # chain has: a known-model string (continue branch), an UNKNOWN-model string
    # (falls through, added to neither), and a dict form (added).
    refs = RouterService._policy_references(
        config, ["good", "totally-unknown", {"model": "chain", "provider": "chainrail"}]
    )
    pairs = set(refs)
    assert ("good", "rail") in pairs
    assert ("fb", "fbrail") in pairs
    assert ("fs", "fsrail") in pairs
    assert ("chain", "chainrail") in pairs      # dict-form chain entry added
    assert ("", "x") not in pairs               # blank model dropped
    assert ("m", 5) not in pairs                # typed provider dropped


def test_policy_references_fail_safe_dict_without_fallback():
    """fail_safe is a dict but has no fallback list -> the fallback loop is skipped."""
    refs = RouterService._policy_references(
        {"fail_safe": {"model": "fs", "provider": "fsrail"}}, []
    )
    assert ("fs", "fsrail") in set(refs)


def test_policy_references_handles_non_dict_tiers_block():
    """A tiers value that is not a mapping is tolerated (no crash, no refs)."""
    assert RouterService._policy_references({"tiers": "nope"}, []) == []


def test_policy_references_empty_config_skips_all_loops():
    """No classifier/tiers/fail_safe: every add() and both fallback loops are
    no-ops (covers the fail_safe-is-{} -> skip-fallback-loop branch)."""
    assert RouterService._policy_references({}, []) == []


def test_policy_references_non_dict_fail_safe_skips_block():
    """A fail_safe that is not a mapping skips the whole fail_safe.fallback
    block (the isinstance(fail_safe, dict) False branch, 222->229)."""
    refs = RouterService._policy_references(
        {"fail_safe": "not-a-dict", "classifier": {"model": "c", "provider": "r"}}, []
    )
    assert ("c", "r") in set(refs)


def test_is_manually_banned_skips_non_dict_and_matches_blank_provider():
    # A non-dict ban entry is skipped; a ban with no provider matches any.
    bans = ["not-a-dict", {"model": "x", "provider": ""}]
    assert RouterService._is_manually_banned(bans, "x", "any-rail") is True
    assert RouterService._is_manually_banned(bans, "other", "rail") is False


def test_is_manually_banned_specific_provider_must_match():
    """A ban scoped to a specific provider does NOT fire for another provider
    (exercises the same-model/different-provider fall-through)."""
    bans = [{"model": "x", "provider": "rail-a"}]
    assert RouterService._is_manually_banned(bans, "x", "rail-b") is False
    assert RouterService._is_manually_banned(bans, "x", "rail-a") is True


def test_liveness_reports_validation_errors_and_survives_internal_error(tmp_path, monkeypatch):
    # (a) invalid config -> liveness still returns, carrying validation_errors.
    invalid = tmp_path / "bad.yaml"
    invalid.write_text("enabled: true\n", encoding="utf-8")  # missing default/tiers
    result = RouterService(invalid).liveness()
    assert result.get("validation_errors")

    # (b) an unexpected internal error is caught -> degraded envelope, no raise.
    service = RouterService(invalid)
    monkeypatch.setattr(
        "router.service.Blocklist.fallback_chain",
        lambda _self: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    degraded = service.liveness()
    assert degraded["worst"] == "degraded"
    assert "could not compose liveness" in degraded["error"]


def test_liveness_composes_states(config_path, monkeypatch):
    """Policy references are composed with breaker and manual-ban state."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["tiers"] = {
        "T1": {"model": "alive", "provider": "cheap"},
        "T2": {"model": "probing", "provider": "cheap"},
        "T3": {
            "model": "quota", "provider": "primary",
            "fallback": [{"model": "backup", "provider": "backup-rail"}],
        },
        "T4": {"model": "manual", "provider": "blocked-rail"},
    }
    config["classifier"] = {"model": "judge", "provider": "judge-rail"}
    config["fail_safe"] = {
        "model": "safe", "provider": "safe-rail",
        "fallback": [{"model": "backup", "provider": "backup-rail"}],
    }
    config["blocklist"] = {
        "manual_ban": [{"model": "manual", "provider": "blocked-rail"}],
        "fallback_chain": ["quota", "backup"],
        "auto_breaker": {"enabled": True},
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    monkeypatch.setattr(
        "router.service.Blocklist.breaker_status",
        lambda _self: [
            {
                "model_key": "quota@primary",
                "state": "OPEN",
                "cooldown_remaining_s": 42.0,
                "last_failure_kind": "quota_exhausted",
            },
            {
                "model_key": "probing@cheap",
                "state": "HALF_OPEN",
                "cooldown_remaining_s": 0.0,
                "last_failure_kind": "hard_timeout",
            },
        ],
    )

    liveness = RouterService(config_path).liveness()

    states = {entry["model_key"]: entry["state"] for entry in liveness["models"]}
    assert states == {
        "alive@cheap": "alive",
        "backup@backup-rail": "alive",
        "judge@judge-rail": "alive",
        "manual@blocked-rail": "dead",
        "probing@cheap": "degraded",
        "quota@primary": "quota_exhausted",
        "safe@safe-rail": "alive",
    }
    assert liveness["worst"] == "dead"
    assert "429" not in repr(liveness)


# ---------------------------------------------------------------------------
# Capability routing: chain preview, tier knobs, warnings, registry coverage
# ---------------------------------------------------------------------------


@pytest.fixture
def capability_config_path(tmp_path):
    """A policy that exercises every new tier field with REAL registry models.

    T2 is the interesting one: its primary cannot see an image (glm-5.3 has
    vision False in the registry) and all three fallback hops can, so a vision
    task rejects the primary and leaves three eligible hops to order — enough
    permutations that a reshuffling preview would be obvious.
    """
    path = tmp_path / "router.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "enabled": True,
                "classifier": {
                    # Top-level pair is the compatibility mirror of chain[0].
                    "model": "glm-4.7",
                    "provider": "zai",
                    "chain": [
                        {"model": "glm-4.7", "provider": "zai",
                         "billing_mode": "plan"},
                        {"model": "deepseek-v4-flash", "provider": "deepseek",
                         "billing_mode": "metered"},
                    ],
                    "temperature": 0,
                    "timeout_seconds": 15,
                },
                "fail_safe": {
                    "profile": "coder", "model": "glm-4.7", "provider": "zai",
                },
                "blocklist": {
                    "manual_ban": [],
                    "fallback_chain": [],
                    "auto_breaker": {"enabled": False},
                },
                "rules": [
                    {
                        "id": "vision-required",
                        "status": "stable",
                        "when": {"needs_vision": {"eq": True}},
                        "then": {"profile": "coder", "model": "T2"},
                    }
                ],
                "default": {"action": "classify"},
                "tiers": {
                    "T1": {"model": "glm-4.7", "provider": "zai",
                           "billing_mode": "plan"},
                    "T2": {
                        "model": "glm-5.3",
                        "provider": "zai",
                        "billing_mode": "plan",
                        "fallback": [
                            {"model": "mimo-v2.5", "provider": "xiaomi",
                             "billing_mode": "metered"},
                            {"model": "kimi-k3", "provider": "moonshot",
                             "billing_mode": "metered"},
                            {"model": "dots-studio/dots-3-note-preview:free",
                             "provider": "openrouter", "billing_mode": "free"},
                        ],
                        "fallback_strategy": "random",
                        "pin_primary": False,
                    },
                    "T3": {
                        "model": "gpt-5.6-terra", "provider": "openai-codex",
                        "requirements": {"min_context": 200000},
                    },
                    "T4": {"model": "gpt-5.5", "provider": "openai-codex"},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


_VISION_TASK = "Look at this screenshot of the ui and fix the layout"


def test_explain_surfaces_chain_plan_with_requirements_and_rejected(
    capability_config_path,
):
    """explain() reaches the chain plan: requirements, survivors, rejections."""
    service = RouterService(capability_config_path)

    result = service.explain(_VISION_TASK)

    plan = result["chain_plan"]
    # Lifted to the top level AND still inside the decision trace.
    assert plan == result["decision"]["chain_plan"]

    # Derived from the signal vector: a screenshot task needs vision.
    assert plan["requirements"]["vision"] is True
    assert plan["requirements"]["min_context"] > 0

    # The primary is dropped with a reason from the closed reason set, and the
    # rejection names the elo so the console can render "skipped, because".
    assert [(r["model"], r["reject_reason"]) for r in plan["rejected"]] == [
        ("glm-5.3", "no_vision")
    ]
    # Only vision-capable hops survive; the rejected primary is gone.
    assert {hop["model"] for hop in plan["chain"]} == {
        "mimo-v2.5", "kimi-k3", "dots-studio/dots-3-note-preview:free",
    }
    assert plan["strategy"] == "random"
    assert plan["bypassed"] is False
    assert plan["independent_rails"] == 3


def test_explain_preview_is_stable_across_calls(capability_config_path):
    """Two explains of the same task preview the SAME order, shuffle included.

    A console polls /explain. Under fallback_strategy: random an unseeded rng
    would reorder the chain on every poll, so the operator could not tell a
    policy change from shuffle noise.
    """
    service = RouterService(capability_config_path)

    first = service.explain(_VISION_TASK)
    second = service.explain(_VISION_TASK)

    assert first["chain_plan"]["strategy"] == "random"  # the churn-prone path
    assert [hop["model"] for hop in first["chain_plan"]["chain"]] == [
        hop["model"] for hop in second["chain_plan"]["chain"]
    ]
    assert first == second  # whole response, not just the chain


def test_explain_injects_a_deterministic_rng(capability_config_path, monkeypatch):
    """The stability above comes from an rng service OWNS, not from luck."""
    import router.service as service_mod

    seen = []
    real_explain = service_mod.rules_explain

    def spy(*args, **kwargs):
        seen.append(kwargs.get("rng"))
        return real_explain(*args, **kwargs)

    monkeypatch.setattr(service_mod, "rules_explain", spy)
    service = RouterService(capability_config_path)
    service.explain(_VISION_TASK)
    service.explain(_VISION_TASK)

    assert len(seen) == 2
    assert all(rng is not None for rng in seen)
    # Same seed => identical stream, which is what makes the preview stable.
    assert seen[0].random() == seen[1].random()


def test_explain_chain_plan_shape_survives_a_planless_rules(
    capability_config_path, monkeypatch
):
    """A rules.py that returns no chain_plan still yields the plan SHAPE.

    And the shape is the PLANNER's degraded shape, phase-2 keys included. The
    console branches on these keys and cannot see which module produced the plan
    it was handed, so a narrower shape here would make "no clock" indistinguishable
    from "unreported" — and the console's planWhen falls back to the BROWSER's hour
    on that ambiguity, pricing a plan that never saw a clock.
    """
    import router.rules as rules_mod
    import router.service as service_mod

    monkeypatch.setattr(
        service_mod, "rules_explain",
        lambda *_a, **_kw: {"matched_rule_id": None, "output": {"action": "classify"},
                            "matched_clauses": {}, "cause": "default_fallthrough"},
    )
    result = RouterService(capability_config_path).explain(_VISION_TASK)
    assert result["chain_plan"] == {
        "chain": [], "requirements": {}, "rejected": [], "unknown": [],
        "bypassed": False, "unsatisfiable": [],
        "strategy": "sequential", "strategy_declared": "sequential",
        "strategy_degraded": False, "strategy_degraded_reason": "",
        "pin_primary": True, "independent_rails": 0,
        "time_agnostic": True, "time_cap_bypassed": False,
        "capped": [], "demoted": [], "promoted": [], "peak_priced": [],
        "multipliers": {},
    }
    # time_agnostic is the key the console needs: it says "no clock", which is a
    # state planWhen renders, instead of leaving it to guess.
    assert result["chain_plan"]["time_agnostic"] is True
    # utc_hour/utc_weekday/time_cap stay ABSENT rather than null, for the reason
    # rules.plan_chain documents: a JSON consumer reads a null hour as midnight,
    # which is a specific wrong answer where absence is a correct one.
    for absent in ("utc_hour", "utc_weekday", "time_cap"):
        assert absent not in result["chain_plan"]
    # One shape, one definition: service's mirror must not drift from the
    # planner's own degraded plan, or two surfaces disagree about what an empty
    # plan looks like.
    assert service_mod._empty_chain_plan() == rules_mod._empty_chain_plan()
    # No plan was made, so no hour can be claimed for one: the clock is still
    # reported (it is what was asked about) but flagged as never having landed,
    # and nothing is described as hour-relative.
    assert result["evaluated_at"]["time_aware"] is False
    assert result["preview"]["time_relative"] is False
    assert result["preview"]["time_relative_reasons"] == []


def test_policy_exposes_the_new_tier_fields(capability_config_path):
    """All four knobs reach the console, and absent ones are NOT invented."""
    policy = RouterService(capability_config_path).policy()

    t2 = policy["tiers"]["T2"]
    assert t2["fallback_strategy"] == "random"
    assert t2["pin_primary"] is False
    assert t2["billing_mode"] == "plan"
    assert policy["tiers"]["T3"]["requirements"] == {"min_context": 200000}

    # T1 declares none of the ordering knobs. They must NOT be materialised
    # with their defaults: the console posts this policy back through
    # plan()/apply(), which would then write knobs nobody chose.
    assert "fallback_strategy" not in policy["tiers"]["T1"]
    assert "pin_primary" not in policy["tiers"]["T1"]

    # The tier mapping is copied whole, so the fallback hops keep their own
    # per-elo fields too.
    assert t2["fallback"][0] == {
        "model": "mimo-v2.5", "provider": "xiaomi", "billing_mode": "metered",
    }


def test_policy_tiers_are_a_copy_and_degrade_on_bad_yaml(capability_config_path):
    """Mutating the returned policy cannot corrupt anything; junk yields {}."""
    service = RouterService(capability_config_path)
    policy = service.policy()
    policy["tiers"]["T2"]["model"] = "mutated"
    assert service.policy()["tiers"]["T2"]["model"] == "glm-5.3"

    # A read path degrades instead of handing a scalar to the console.
    assert RouterService._policy_tiers("nope") == {}
    assert RouterService._policy_tiers(None) == {}
    assert RouterService._policy_tiers({"T1": "not-a-mapping"}) == {
        "T1": "not-a-mapping"
    }


@pytest.fixture
def warned_config_path(tmp_path):
    """A VALID policy that nonetheless has two advisory findings.

    T1's two hops sit behind one upstream (nous is a white-label reseller in
    front of openrouter), and T3's second hop is unknown to the registry.
    """
    path = tmp_path / "router.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "enabled": True,
                "classifier": {"model": "glm-4.7", "provider": "zai"},
                "fail_safe": {"profile": "coder", "model": "glm-4.7",
                              "provider": "zai"},
                "blocklist": {"manual_ban": [], "fallback_chain": [],
                              "auto_breaker": {"enabled": False}},
                "rules": [],
                "default": {"action": "classify"},
                "tiers": {
                    "T1": {
                        "model": "meituan/longcat-2.0:free", "provider": "nous",
                        "fallback": [{"model": "openrouter/free",
                                      "provider": "openrouter"}],
                    },
                    "T2": {"model": "glm-5.3", "provider": "zai"},
                    "T3": {
                        # Unknown to the registry but DESCRIBED in yaml, so it
                        # is verifiable; its fallback hop describes nothing.
                        "model": "house-model", "provider": "local-rail",
                        "context_window": 500000, "vision": True,
                        "fallback": [{"model": "ghost-model",
                                      "provider": "other-rail"}],
                    },
                    "T4": {"model": "gpt-5.5", "provider": "openai-codex"},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_status_reports_warnings_without_flipping_valid(warned_config_path):
    """Warnings inform; errors block. A warning must never do an error's job."""
    service = RouterService(warned_config_path)
    status = service.status()

    assert status["valid"] is True
    assert status["validation_errors"] == []
    assert any("share upstream 'openrouter'" in w for w in status["warnings"])
    assert any("unknown to the capability registry" in w
               for w in status["warnings"])
    # Strictly separate channels: no warning text leaks into the blocking list.
    assert not set(status["warnings"]) & set(status["validation_errors"])

    # And the write gate is untouched by them: a warned config still applies.
    plan = service.plan({"default": {"action": "classify"}})
    assert plan["valid"] is True
    assert plan["errors"] == []
    assert service.apply(plan["base_hash"], plan["policy"])["ok"] is True

    # lint() stays the pure error gate — warnings are status()'s job alone.
    assert service.lint() == {"valid": True, "errors": [], "error_targets": []}


def test_status_warnings_degrade_when_rules_cannot_warn(config_path, monkeypatch):
    """No lint_warnings, or a raising one, yields [] — never a broken status."""
    import router.service as service_mod

    monkeypatch.setattr(service_mod, "rules_lint_warnings", None)
    assert RouterService(config_path).status()["warnings"] == []

    monkeypatch.setattr(
        service_mod, "rules_lint_warnings",
        lambda _c: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert RouterService(config_path).status()["warnings"] == []

    # A non-list return is ignored rather than passed through.
    monkeypatch.setattr(service_mod, "rules_lint_warnings", lambda _c: "nope")
    assert RouterService(config_path).status()["warnings"] == []


def _shadowed_config(tmp_path):
    """A router.yaml whose later rule can never fire: the console's jump target."""
    path = tmp_path / "shadowed.yaml"
    path.write_text(
        "enabled: true\n"
        "rules:\n"
        "  - id: broad\n"
        "    when:\n"
        "      has_code: {eq: true}\n"
        "    then: {model: T2}\n"
        "  - id: narrow\n"
        "    when:\n"
        "      has_code: {eq: true}\n"
        "    then: {model: T1}\n"
        "default: {action: classify}\n"
        "tiers:\n"
        "  T1: {model: m1, provider: p1}\n"
        "  T2: {model: m2, provider: p2}\n"
        "  T3: {model: m3, provider: p3}\n"
        "  T4: {model: m4, provider: p4}\n",
        encoding="utf-8",
    )
    return path


def test_error_targets_align_with_validation_errors(tmp_path):
    """A shadowed config reports the structured jump target beside each error.

    The console's "Ver regra N" button is built from ``error_targets[0]`` — the
    target of the error it actually shows. Both /status (validation_errors) and
    /lint (errors) must carry the same pairing under their own field names.
    """
    service = RouterService(_shadowed_config(tmp_path))
    status = service.status()

    assert status["valid"] is False
    assert status["validation_errors"] == [
        "rule 'narrow' is shadowed by earlier rule 'broad'"
    ]
    assert status["error_targets"] == [{
        "code": "shadowed",
        "later_index": 1,
        "later_id": "narrow",
        "earlier_index": 0,
        "earlier_id": "broad",
        "message": "rule 'narrow' is shadowed by earlier rule 'broad'",
    }]

    lint = service.lint()
    assert lint["errors"] == status["validation_errors"]
    assert lint["error_targets"] == status["error_targets"]


def test_error_targets_are_none_for_errors_that_name_no_rule(tmp_path):
    """A missing default blocks the write but has no row to jump to.

    Every error here is about the config as a whole, so every target is None —
    and the two lists stay the same length, which is the alignment contract.
    """
    config = tmp_path / "no-default.yaml"
    config.write_text("enabled: true\n", encoding="utf-8")
    service = RouterService(config)
    status = service.status()
    assert status["valid"] is False
    assert any("missing mandatory 'default'" in e for e in status["validation_errors"])
    assert len(status["error_targets"]) == len(status["validation_errors"])
    assert all(target is None for target in status["error_targets"])


def test_error_targets_degrade_when_rules_cannot_find(tmp_path, monkeypatch):
    """No lint_findings, a raising one, or a non-list return — never a broken read.

    Mirror of the lint_warnings degrade: an older rules.py must cost the
    operator the jump button, not the whole status surface.
    """
    import router.service as service_mod

    service = RouterService(_shadowed_config(tmp_path))
    assert service.status()["error_targets"] == [{
        "code": "shadowed", "later_index": 1, "later_id": "narrow",
        "earlier_index": 0, "earlier_id": "broad",
        "message": "rule 'narrow' is shadowed by earlier rule 'broad'",
    }], "the degrade below is only meaningful against a real finding"

    monkeypatch.setattr(service_mod, "rules_lint_findings", None)
    assert service.status()["error_targets"] == [None]

    monkeypatch.setattr(
        service_mod, "rules_lint_findings",
        lambda _c: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    assert service.status()["error_targets"] == [None]

    monkeypatch.setattr(service_mod, "rules_lint_findings", lambda _c: "nope")
    assert service.status()["error_targets"] == [None]

    # A finding that cannot pair with an error string is ignored, not a crash.
    monkeypatch.setattr(
        service_mod, "rules_lint_findings",
        lambda _c: [{"code": "shadowed", "later_index": 1}, "nope"],
    )
    assert service.status()["error_targets"] == [None]


def test_hot_apply_round_trips_fallback_strategy_and_reordered_fallback(
    capability_config_path,
):
    """A tier carrying the new knobs survives _merge_hot/_deep_merge_value.

    No new _HOT_KEYS member was needed: these fields live inside `tiers`, which
    is already hot. The reordered `fallback` also proves lists still REPLACE
    wholesale — that is what makes reordering and deleting an elo expressible.
    """
    service = RouterService(capability_config_path)
    reordered = [
        {"model": "kimi-k3", "provider": "moonshot", "billing_mode": "metered"},
        {"model": "mimo-v2.5", "provider": "xiaomi", "billing_mode": "metered"},
    ]

    plan = service.plan({"tiers": {"T2": {
        "fallback_strategy": "sequential",
        "pin_primary": True,
        "requirements": {"min_context": 128000},
        "fallback": reordered,
    }}})
    assert plan["valid"] is True
    assert service.apply(plan["base_hash"], plan["policy"])["ok"] is True

    written = yaml.safe_load(capability_config_path.read_text(encoding="utf-8"))
    t2 = written["tiers"]["T2"]
    assert t2["fallback_strategy"] == "sequential"
    assert t2["pin_primary"] is True
    assert t2["requirements"] == {"min_context": 128000}
    # Reordered AND shortened: the list replaced, it did not union (3 -> 2).
    assert t2["fallback"] == reordered
    # Untouched sibling keys survive the deep merge.
    assert t2["model"] == "glm-5.3"
    assert t2["billing_mode"] == "plan"
    # Sibling tiers are untouched.
    assert written["tiers"]["T3"]["requirements"] == {"min_context": 200000}

    # The edit is live on the next read, with no restart.
    assert service.policy()["tiers"]["T2"]["fallback_strategy"] == "sequential"
    assert service.explain(_VISION_TASK)["chain_plan"]["strategy"] == "sequential"


def test_apply_rejects_an_invalid_fallback_strategy(capability_config_path):
    """The write gate still fails closed on the new fields."""
    service = RouterService(capability_config_path)
    before = capability_config_path.read_bytes()

    plan = service.plan({"tiers": {"T2": {"fallback_strategy": "round-robin"}}})
    assert plan["valid"] is False
    assert any("fallback_strategy" in e for e in plan["errors"])

    result = service.apply(plan["base_hash"], plan["policy"])
    assert result["ok"] is False
    assert capability_config_path.read_bytes() == before


def test_liveness_covers_every_classifier_chain_hop_without_duplicates(
    capability_config_path,
):
    """Both classifier hops are watched, and the mirror is not double-counted."""
    liveness = RouterService(capability_config_path).liveness()
    keys = [entry["model_key"] for entry in liveness["models"]]

    # chain[1] is only reachable by walking the chain.
    assert "deepseek-v4-flash@deepseek" in keys
    # chain[0] and the top-level compatibility mirror are the same elo.
    assert keys.count("glm-4.7@zai") == 1
    assert len(keys) == len(set(keys))


def test_liveness_flags_a_registry_unknown_elo(warned_config_path):
    """capabilities_known is an extra FIELD, not a fifth state."""
    liveness = RouterService(warned_config_path).liveness()
    known = {e["model_key"]: e["capabilities_known"] for e in liveness["models"]}

    # Declares nothing and the registry has never heard of it -> flagged.
    assert known["ghost-model@other-rail"] is False
    # Also unknown to the registry, but described in yaml: `declared` wins over
    # the registry, so it is verifiable and must NOT be flagged.
    assert known["house-model@local-rail"] is True
    assert known["glm-5.3@zai"] is True

    # The state set stays closed, and coverage does not degrade health.
    assert {e["state"] for e in liveness["models"]} == {"alive"}
    assert liveness["worst"] == "alive"


def test_liveness_capability_coverage_fails_open_without_a_registry(monkeypatch):
    """No registry (or a raising one) means nothing is PROVABLY unknown."""
    import router.service as service_mod

    monkeypatch.setattr(service_mod, "_caps", None)
    assert RouterService._capabilities_known("who-knows") is True

    class Exploding:
        @staticmethod
        def capabilities_for(*_a, **_kw):
            raise TypeError("stale registry")

    monkeypatch.setattr(service_mod, "_caps", Exploding)
    assert RouterService._capabilities_known("who-knows") is True


def test_declared_capability_index_ignores_identity_and_junk_keys():
    """Only capability-ish keys count as a declaration.

    `provider` is identity and IS a registry field, so counting it would make
    every elo in policy look known; tuning keys are dropped by the registry.
    """
    index = RouterService._declared_capability_index({
        "classifier": {"model": "c", "provider": "rail", "temperature": 0,
                       "chain": [{"model": "c2", "provider": "rail",
                                  "context_window": 1000}]},
        "fail_safe": {"model": "fs", "provider": "rail",
                      "fallback": [{"model": "fsfb", "provider": "rail"}]},
        "tiers": {
            "T1": {"model": "t1", "provider": "rail", "fallback_strategy": "random",
                   "pin_primary": True, "requirements": {"min_context": 1}},
            "T2": "not-a-mapping",
            "T3": {"model": "t3", "provider": "rail", "vision": True,
                   "fallback": "not-a-list"},
            "T4": {"provider": "rail"},  # no model -> nothing to index
        },
    })
    # Identity + routing keys only -> no declaration recorded at all.
    assert "t1" not in index
    assert "fsfb" not in index
    # Capability keys are recorded; a stray tuning key is recorded but the
    # registry filters it, so it cannot fake knowledge.
    assert index["c2"] == {"context_window": 1000}
    assert index["t3"] == {"vision": True}
    assert RouterService._capabilities_known("c", index.get("c")) is False
    assert RouterService._capabilities_known("t3", index.get("t3")) is True


def test_config_without_any_new_keys_returns_the_prior_responses(config_path):
    """The pre-capability policy still produces exactly the old answers."""
    service = RouterService(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # policy(): tiers pass through unchanged, with no defaults invented.
    assert service.policy()["tiers"] == raw["tiers"]

    # status(): every prior field keeps its prior value; only `warnings` is new.
    status = service.status()
    assert status["valid"] is True
    assert status["validation_errors"] == []
    assert status["classifier"] == {"model": "judge", "provider": "judge-rail"}

    # explain(): the prior decision, plus an additive chain_plan describing the
    # declared chain in declared order (no strategy, nothing rejected).
    result = service.explain("Debug a race condition")
    assert result["mode"] == "deterministic_dry_run"
    assert result["requires_classifier"] is False
    assert result["decision"]["cause"] == "hard_rule"
    plan = result["chain_plan"]
    assert [hop["model"] for hop in plan["chain"]] == ["strong", "backup"]
    assert plan["strategy"] == "sequential"
    assert plan["rejected"] == []
    assert plan["bypassed"] is False
    # These elos are unknown to the registry, which never rejects — it reports.
    assert set(plan["unknown"]) == {"strong", "backup"}

    # liveness(): the same states as before, plus the coverage field.
    liveness = service.liveness()
    assert {e["state"] for e in liveness["models"]} == {"alive"}
    assert all(e["capabilities_known"] is False for e in liveness["models"])


@pytest.mark.parametrize(
    "accepts_rng, accepts_when, expected_kwargs",
    [
        (False, False, set()),          # a rules.py predating both injections
        (True, False, {"rng"}),         # fallback strategies but no time layer
        (False, True, {"when"}),        # time layer but no rng (not shipped, still safe)
    ],
)
def test_explain_omits_injected_parameters_an_older_rules_lacks(
    capability_config_path, monkeypatch, accepts_rng, accepts_when, expected_kwargs
):
    """Each injected parameter is passed only if rules.explain declares it.

    service.py is deployed by file copy, so it can land beside a rules.py that
    predates fallback strategies, the time layer, or both. The two flags are
    INDEPENDENT: losing one injection must not cost the other, and neither may
    become a TypeError inside a read path. Such a rules.py orders sequentially and
    plans time-agnostically, so what matters is that explain() still answers — and
    says the plan was not time-aware instead of claiming an hour it never used.
    """
    import router.service as service_mod

    seen = []
    real_explain = service_mod.rules_explain

    def older_explain(*args, **kwargs):
        seen.append(set(kwargs))
        return real_explain(*args, **kwargs)

    monkeypatch.setattr(service_mod, "_EXPLAIN_ACCEPTS_RNG", accepts_rng)
    monkeypatch.setattr(service_mod, "_EXPLAIN_ACCEPTS_WHEN", accepts_when)
    monkeypatch.setattr(service_mod, "rules_explain", older_explain)

    result = RouterService(capability_config_path).explain(_VISION_TASK)

    assert seen == [expected_kwargs]
    assert result["mode"] == "deterministic_dry_run"
    assert isinstance(result["chain_plan"], dict)
    # The clock is reported either way; `time_aware` says whether it landed.
    assert result["evaluated_at"]["utc_hour"] in range(24)
    assert result["evaluated_at"]["time_aware"] is accepts_when


# ---------------------------------------------------------------------------
# Time-windowed pricing: the injected clock on the read paths
# ---------------------------------------------------------------------------

# Fixed clocks, all Monday 2026-08-17 unless the weekday is the point.
# deepseek is 2.0x at 01:00-04:00 and 06:00-10:00 UTC every day; zai's plan
# models are 2.0x at 06:00-10:00 on WEEKDAYS; glm-4.6 and MiniMax-M3 are flat.
_PEAK_0200 = datetime(2026, 8, 17, 2, tzinfo=timezone.utc)      # deepseek peak
_PEAK_0300 = datetime(2026, 8, 17, 3, tzinfo=timezone.utc)      # last peak hour
_OFFPEAK_0400 = datetime(2026, 8, 17, 4, tzinfo=timezone.utc)   # [1, 4) is half-open
_PEAK_0700 = datetime(2026, 8, 17, 7, tzinfo=timezone.utc)      # both rails peak
_OFFPEAK_1500 = datetime(2026, 8, 17, 15, tzinfo=timezone.utc)

_TRIVIAL_TASK = "Rename a variable"          # verb_class trivial -> T1
_HARD_TASK = "Refactor the parser module"    # verb_class hard    -> T4
_PLAIN_TASK = "Summarize this note"          # verb_class unknown -> T3 at peak, else T2


@pytest.fixture
def time_config_path(tmp_path):
    """A policy whose plan legitimately depends on the hour, with REAL prices.

    One tier per time-layer stage, so a failure names the stage:

    * T1 — ``time_cap`` over two deepseek elos, so the cap EMPTIES the chain in a
      peak window and has to bypass itself (a cost control must not cause an
      outage) while keeping its per-elo diagnostics.
    * T2 — ``cheapest_now`` over one windowed elo and two flat ones. Off-peak
      output prices are MiniMax-M3 1.2 < deepseek-v4-pro 1.98 < glm-4.6 2.2; in a
      deepseek peak the primary doubles to 3.96 and lands last, so the ORDER is
      hour-relative without anything being dropped.
    * T3 — ``time_cap`` 1.5 over a windowed and a flat elo, so the peak makes
      deepseek ineligible and glm-4.6 still routes.
    * T4 — ``avoid_peak`` demotes deepseek while it is peak-priced and never
      removes it.

    The ``peak-hours`` rule is keyed on the INJECTED ``utc_hour`` feature, which is
    the 4am-cron question: at 06:00-10:00 UTC the plain task routes to the capped
    tier, and at any other hour it falls through to the cheapest_now default.
    """
    path = tmp_path / "router.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "enabled": True,
                # A plan-covered, windowed elo: liveness below asserts its 2.0x.
                "classifier": {"model": "glm-5.3-flash", "provider": "zai"},
                "fail_safe": {"profile": "coder", "model": "glm-4.6",
                              "provider": "zai"},
                "blocklist": {"manual_ban": [], "fallback_chain": [],
                              "auto_breaker": {"enabled": False}},
                "rules": [
                    {"id": "trivial-verbs", "status": "stable",
                     "when": {"verb_class": {"eq": "trivial"}},
                     "then": {"profile": "coder", "model": "T1"}},
                    {"id": "hard-verbs", "status": "stable",
                     "when": {"verb_class": {"eq": "hard"}},
                     "then": {"profile": "coder", "model": "T4"}},
                    {"id": "peak-hours", "status": "stable",
                     "when": {"utc_hour": {"gte": 6, "lt": 10}},
                     "then": {"profile": "coder", "model": "T3"}},
                ],
                "default": {"profile": "coder", "model": "T2"},
                "tiers": {
                    "T1": {
                        "model": "deepseek-v4-flash", "provider": "deepseek",
                        "billing_mode": "metered",
                        "fallback": [{"model": "deepseek-v4-pro",
                                      "provider": "deepseek",
                                      "billing_mode": "metered"}],
                        "time_cap": {"max_multiplier": 1.5},
                    },
                    "T2": {
                        "model": "deepseek-v4-pro", "provider": "deepseek",
                        "billing_mode": "metered",
                        "fallback": [
                            {"model": "glm-4.6", "provider": "zai",
                             "billing_mode": "metered"},
                            {"model": "MiniMax-M3", "provider": "minimax",
                             "billing_mode": "metered"},
                        ],
                        "fallback_strategy": "cheapest_now",
                        "pin_primary": False,
                    },
                    "T3": {
                        "model": "deepseek-v4-pro", "provider": "deepseek",
                        "billing_mode": "metered",
                        "fallback": [{"model": "glm-4.6", "provider": "zai",
                                      "billing_mode": "metered"}],
                        "time_cap": {"max_multiplier": 1.5},
                    },
                    "T4": {
                        "model": "deepseek-v4-pro", "provider": "deepseek",
                        "billing_mode": "metered",
                        "fallback": [{"model": "glm-4.6", "provider": "zai",
                                      "billing_mode": "metered"}],
                        "time_policy": {"avoid_peak": ["deepseek"]},
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _chain(result):
    return [hop["model"] for hop in result["chain_plan"]["chain"]]


def test_explain_reports_the_evaluation_time_it_defaulted_to(
    time_config_path, monkeypatch
):
    """With no explicit time, explain evaluates at the real current UTC hour.

    A console rendering multipliers and a cheapest_now order has to know which
    hour they belong to; an unlabelled time-relative answer is indistinguishable
    from a wrong one.
    """
    import router.service as service_mod

    monkeypatch.setattr(
        service_mod, "_utc_now",
        lambda: datetime(2026, 8, 17, 15, 42, 17, 123456, tzinfo=timezone.utc),
    )

    result = RouterService(time_config_path).explain(_PLAIN_TASK)

    assert result["evaluated_at"] == {
        # Truncated to the top of the hour: price windows are declared in whole
        # UTC hours, so minutes could only churn the payload, never the answer.
        "at": "2026-08-17T15:00:00+00:00",
        "at_source": "now",
        "utc_hour": 15,
        "utc_weekday": 0,
        "time_aware": True,
    }
    # The hour explain reports and the hour the PLAN was made at are one reading.
    assert result["chain_plan"]["utc_hour"] == 15
    assert result["chain_plan"]["utc_weekday"] == 0
    assert result["chain_plan"]["time_agnostic"] is False


def test_explain_at_an_explicit_hour_changes_the_plan(time_config_path):
    """The 4am-cron question: ask for 07:00 UTC and get 07:00 UTC's plan.

    At 07:00 the utc_hour rule fires (the injected clock feature reaches match())
    and the capped tier drops the peak-priced primary; at 15:00 the same task
    falls through to the cheapest_now default with nothing capped.
    """
    service = RouterService(time_config_path)

    peak = service.explain(_PLAIN_TASK, _PEAK_0700)
    offpeak = service.explain(_PLAIN_TASK, _OFFPEAK_1500)

    assert peak["evaluated_at"]["at_source"] == "explicit"
    assert peak["evaluated_at"]["utc_hour"] == 7
    assert peak["decision"]["matched_rule_id"] == "peak-hours"
    # The rule matched ON the injected feature, so the trace can show it.
    assert peak["decision"]["matched_clauses"] == {"utc_hour": {"gte": 6, "lt": 10}}
    assert _chain(peak) == ["glm-4.6"]
    assert [(c["model"], c["multiplier"]) for c in peak["chain_plan"]["capped"]] == [
        ("deepseek-v4-pro", 2.0)
    ]
    assert peak["chain_plan"]["time_cap"] == {"max_multiplier": 1.5}
    assert peak["chain_plan"]["multipliers"] == {
        "deepseek-v4-pro": 2.0, "glm-4.6": 1.0,
    }

    assert offpeak["decision"]["matched_rule_id"] is None
    assert offpeak["chain_plan"]["capped"] == []
    assert _chain(offpeak) == ["MiniMax-M3", "deepseek-v4-pro", "glm-4.6"]
    # Same task, same policy, different hour -> a different plan, on purpose.
    assert _chain(peak) != _chain(offpeak)


def test_explain_cheapest_now_order_is_hour_relative_and_labelled(time_config_path):
    """The order flips across a window boundary, and the response says why.

    01:00-04:00 UTC is half-open, so 03:00 is inside deepseek's peak and 04:00 is
    outside it. Both hours fall through to the cheapest_now tier, so the only
    thing that changed is the price — which is exactly what an operator must be
    able to conclude instead of suspecting a shuffle.
    """
    service = RouterService(time_config_path)

    inside = service.explain(_PLAIN_TASK, _PEAK_0300)
    outside = service.explain(_PLAIN_TASK, _OFFPEAK_0400)

    assert inside["chain_plan"]["strategy"] == "cheapest_now"
    # deepseek-v4-pro output 1.98 -> 3.96 at 2.0x, so it sinks behind glm-4.6 2.2.
    assert _chain(inside) == ["MiniMax-M3", "glm-4.6", "deepseek-v4-pro"]
    assert _chain(outside) == ["MiniMax-M3", "deepseek-v4-pro", "glm-4.6"]

    assert inside["preview"] == {
        "seed": 0,
        "reproducible_within": "utc_hour",
        "time_relative": True,
        "time_relative_reasons": ["cheapest_now", "price_window"],
        # No prompt_text was passed, so the preview measured the goal line and
        # says so — the alternative is a response that looks identical whether it
        # reproduced the real turn or six tokens of goal.
        "sized_from": "task",
        "prompt_chars": len(_PLAIN_TASK),
    }
    # Outside every window the ORDER is still price-derived, but no elo is priced
    # off its base rate, so `price_window` drops out.
    assert outside["preview"]["time_relative_reasons"] == ["cheapest_now"]


def test_explain_is_identical_within_an_hour_and_free_to_differ_across_one(
    time_config_path, monkeypatch
):
    """Two polls seconds apart are byte-identical; the next hour may differ.

    Those two properties are the whole contract: a polled preview that churned
    every second would be unreadable, and a preview frozen against the clock
    would show an order production is not using.
    """
    import router.service as service_mod

    service = RouterService(time_config_path)
    clock = {"now": datetime(2026, 8, 17, 3, 0, 1, tzinfo=timezone.utc)}
    monkeypatch.setattr(service_mod, "_utc_now", lambda: clock["now"])

    first = service.explain(_PLAIN_TASK)
    clock["now"] += timedelta(seconds=59, microseconds=999999)
    second = service.explain(_PLAIN_TASK)

    assert first == second
    assert json.dumps(first, default=str) == json.dumps(second, default=str)

    # One hour later the same task legitimately orders differently, and the
    # response already carries the reason: the hour moved out of deepseek's peak.
    clock["now"] = datetime(2026, 8, 17, 4, 0, 0, tzinfo=timezone.utc)
    next_hour = service.explain(_PLAIN_TASK)
    assert _chain(next_hour) != _chain(first)
    assert first["preview"]["time_relative"] is True
    assert first["evaluated_at"]["utc_hour"] == 3
    assert next_hour["evaluated_at"]["utc_hour"] == 4


def test_explain_accepts_an_iso_string_and_refuses_anything_else(time_config_path):
    """A query parameter arrives as text, so ISO-8601 is accepted here.

    Every caller parsing the string itself is how two surfaces end up disagreeing
    about what "07:00" meant. An unusable value fails CLOSED (ValueError, which the
    sidecar renders as a 400) rather than silently answering about "now" — an audit
    surface that answers a different question is worse than one that refuses.
    """
    service = RouterService(time_config_path)

    zulu = service.explain(_PLAIN_TASK, "2026-08-17T07:30:00Z")
    offset = service.explain(_PLAIN_TASK, "2026-08-17T04:30:00-03:00")
    naive = service.explain(_PLAIN_TASK, "2026-08-17T07:30:00")

    # All three name the same instant; a naive string is read as UTC.
    assert zulu["evaluated_at"]["at"] == "2026-08-17T07:00:00+00:00"
    assert offset["evaluated_at"] == zulu["evaluated_at"]
    assert naive["evaluated_at"] == zulu["evaluated_at"]
    assert zulu["decision"]["matched_rule_id"] == "peak-hours"

    # A naive datetime is read as UTC too, and an aware one is converted.
    assert service.explain(
        _PLAIN_TASK, datetime(2026, 8, 17, 7, 30)
    )["evaluated_at"] == zulu["evaluated_at"]

    with pytest.raises(ValueError, match="ISO-8601"):
        service.explain(_PLAIN_TASK, "half past seven")
    with pytest.raises(ValueError, match="ISO-8601"):
        service.explain(_PLAIN_TASK, "")
    with pytest.raises(ValueError, match="must be a datetime"):
        service.explain(_PLAIN_TASK, 7)
    with pytest.raises(ValueError, match="must be a datetime"):
        service.explain(_PLAIN_TASK, ["2026-08-17T07:00:00Z"])


def test_explain_shows_a_bypassing_time_cap_with_its_diagnostics(time_config_path):
    """A cap that would empty the chain gives way, and says which elos it hit.

    T1's only elos are two deepseek models, both 2.0x in the peak window, so the
    1.5x cap cannot be satisfied. The chain must survive intact — a cost control
    must never cause an outage — and the per-elo multipliers must survive with it,
    because the flag alone cannot tell an operator whether the cap or the tier is
    wrong.
    """
    service = RouterService(time_config_path)

    peak = service.explain(_TRIVIAL_TASK, _PEAK_0200)
    plan = peak["chain_plan"]

    assert plan["time_cap_bypassed"] is True
    assert _chain(peak) == ["deepseek-v4-flash", "deepseek-v4-pro"]
    assert [(c["model"], c["multiplier"]) for c in plan["capped"]] == [
        ("deepseek-v4-flash", 2.0), ("deepseek-v4-pro", 2.0),
    ]
    assert peak["preview"]["time_relative_reasons"] == ["time_cap", "price_window"]

    # Off-peak the same tier needs no bypass at all.
    calm = service.explain(_TRIVIAL_TASK, _OFFPEAK_1500)
    assert calm["chain_plan"]["time_cap_bypassed"] is False
    assert calm["chain_plan"]["capped"] == []
    assert _chain(calm) == ["deepseek-v4-flash", "deepseek-v4-pro"]


def test_explain_shows_time_policy_demotion_without_dropping_an_elo(time_config_path):
    """avoid_peak moves a peak-priced rail to the back; it never removes it."""
    service = RouterService(time_config_path)

    peak = service.explain(_HARD_TASK, _PEAK_0700)
    offpeak = service.explain(_HARD_TASK, _OFFPEAK_1500)

    assert peak["decision"]["output"]["time_policy"] == {"avoid_peak": ["deepseek"]}
    assert peak["chain_plan"]["demoted"] == ["deepseek-v4-pro"]
    assert _chain(peak) == ["glm-4.6", "deepseek-v4-pro"]
    # A permutation, not a filter: the demoted rail is still attemptable.
    assert sorted(_chain(peak)) == sorted(_chain(offpeak))
    assert _chain(offpeak) == ["deepseek-v4-pro", "glm-4.6"]
    assert offpeak["chain_plan"]["demoted"] == []
    assert peak["preview"]["time_relative_reasons"] == ["time_policy", "price_window"]
    # The declared knob is what makes it hour-relative, so the off-peak plan —
    # where the policy moved nothing — is still labelled time-relative.
    assert offpeak["preview"]["time_relative"] is True
    assert "time_policy" in offpeak["preview"]["time_relative_reasons"]


def test_explain_weekend_is_off_peak_for_the_plan_rails(tmp_path):
    """zai's peak is weekday-only, so Saturday 07:00 prices at the base rate.

    Asserted through the service because the clock the console injects is the one
    that has to carry the weekday: an hour with no date could not answer this.
    """
    path = tmp_path / "router.yaml"
    path.write_text(
        yaml.safe_dump({
            "enabled": True,
            "classifier": {"model": "glm-4.7", "provider": "zai"},
            "fail_safe": {"profile": "coder", "model": "glm-4.6", "provider": "zai"},
            "blocklist": {"manual_ban": [], "fallback_chain": [],
                          "auto_breaker": {"enabled": False}},
            "rules": [],
            "default": {"profile": "coder", "model": "T1"},
            "tiers": {
                "T1": {"model": "glm-5.3-flash", "provider": "zai",
                       "billing_mode": "plan",
                       "fallback": [{"model": "glm-4.6", "provider": "zai",
                                     "billing_mode": "metered"}]},
                "T2": {"model": "glm-4.6", "provider": "zai"},
                "T3": {"model": "glm-4.6", "provider": "zai"},
                "T4": {"model": "glm-4.6", "provider": "zai"},
            },
        }, sort_keys=False),
        encoding="utf-8",
    )
    service = RouterService(path)

    monday = service.explain(_PLAIN_TASK, datetime(2026, 8, 17, 7, tzinfo=timezone.utc))
    saturday = service.explain(_PLAIN_TASK, datetime(2026, 8, 22, 7, tzinfo=timezone.utc))

    assert monday["chain_plan"]["multipliers"]["glm-5.3-flash"] == 2.0
    assert saturday["chain_plan"]["multipliers"]["glm-5.3-flash"] == 1.0
    assert saturday["evaluated_at"]["utc_weekday"] == 5
    assert saturday["preview"]["time_relative"] is False


def test_policy_exposes_time_policy_and_time_cap(time_config_path):
    """Both time knobs reach the console verbatim, nested values included."""
    service = RouterService(time_config_path)
    policy = service.policy()

    assert policy["tiers"]["T3"]["time_cap"] == {"max_multiplier": 1.5}
    assert policy["tiers"]["T4"]["time_policy"] == {"avoid_peak": ["deepseek"]}
    assert policy["tiers"]["T2"]["fallback_strategy"] == "cheapest_now"
    assert policy["tiers"]["T2"]["pin_primary"] is False
    # Absent knobs are still not invented: T1 declares no time_policy.
    assert "time_policy" not in policy["tiers"]["T1"]

    # The copy is deep, so a console normalising the nested knobs in place cannot
    # edit the policy it is only displaying.
    policy["tiers"]["T4"]["time_policy"]["avoid_peak"].append("zai")
    policy["tiers"]["T3"]["time_cap"]["max_multiplier"] = 99
    fresh = service.policy()
    assert fresh["tiers"]["T4"]["time_policy"] == {"avoid_peak": ["deepseek"]}
    assert fresh["tiers"]["T3"]["time_cap"] == {"max_multiplier": 1.5}


def test_status_carries_time_warnings_without_flipping_valid(time_config_path):
    """A cap that will bypass is advisory: reported, never write-blocking."""
    service = RouterService(time_config_path)
    status = service.status()

    assert status["valid"] is True
    assert status["validation_errors"] == []
    assert any("time_cap will bypass" in w for w in status["warnings"])
    assert any("share upstream 'deepseek'" in w for w in status["warnings"])
    assert not set(status["warnings"]) & set(status["validation_errors"])

    # The write gate is untouched by them: the warned config still applies.
    plan = service.plan({"default": {"profile": "coder", "model": "T2"}})
    assert plan["valid"] is True
    assert service.apply(plan["base_hash"], plan["policy"])["ok"] is True
    # lint() stays the pure error gate.
    assert service.lint() == {"valid": True, "errors": [], "error_targets": []}


def test_status_passes_registry_diagnostics_through_verbatim(
    config_path, monkeypatch
):
    """Whatever lint_warnings reports reaches the operator unfiltered.

    rules.lint_warnings folds capabilities.registry_diagnostics() in — a check
    that had no caller at all before phase 2 — and status() is the only surface an
    operator reads warnings from. A projection here would make a registry defect
    (an overlapping price window, say) unobservable, which is the same as not
    having written the check.
    """
    import router.service as service_mod

    diagnostics = [
        "model 'glm-5.3': price_windows entries overlap",
        "model 'mimo-v2.5': price_windows[0].hours_utc must be [start, end)",
    ]
    monkeypatch.setattr(service_mod, "rules_lint_warnings", lambda _c: diagnostics)

    status = RouterService(config_path).status()

    assert status["warnings"] == diagnostics
    assert status["valid"] is True
    assert status["validation_errors"] == []


def test_liveness_reports_price_window_state_as_extra_fields(
    time_config_path, monkeypatch
):
    """Window state rides as FIELDS; the four-value state enum is untouched."""
    import router.service as service_mod

    monkeypatch.setattr(
        service_mod, "_utc_now",
        lambda: datetime(2026, 8, 17, 7, 12, tzinfo=timezone.utc),
    )

    liveness = RouterService(time_config_path).liveness()
    by_key = {entry["model_key"]: entry for entry in liveness["models"]}

    # Both primary rails are peak-priced at 07:00 UTC on a weekday.
    deepseek = by_key["deepseek-v4-pro@deepseek"]
    assert deepseek["in_expensive_window"] is True
    assert deepseek["price_multiplier"] == 2.0
    # The registry's own MAPPING, carried through whole rather than reduced to its
    # hour: a weekday-gated window makes a bare hour ambiguous by up to two days,
    # so a console rendering "peak ends in 2h" needs hours_ahead, not hour.
    # The peak ends at 10:00 UTC, which is `hour` here.
    assert deepseek["next_window_change"] == {
        "hour": 10, "weekday": 0, "hours_ahead": 3, "multiplier": 1.0,
    }
    assert by_key["glm-5.3-flash@zai"]["price_multiplier"] == 2.0

    # A flat-priced elo is never "in a window", and nothing ever changes for it.
    flat = by_key["glm-4.6@zai"]
    assert flat["in_expensive_window"] is False
    assert flat["price_multiplier"] == 1.0
    assert flat["next_window_change"] is None

    # The state enum stays a closed set of four, and a peak price is not a health
    # problem: an expensive rail is still perfectly alive.
    assert {entry["state"] for entry in liveness["models"]} <= {
        "alive", "degraded", "quota_exhausted", "dead",
    }
    assert liveness["worst"] == "alive"
    assert liveness["evaluated_at"] == {
        "at": "2026-08-17T07:00:00+00:00",
        "at_source": "now",
        "utc_hour": 7,
        "utc_weekday": 0,
    }


def test_liveness_price_window_state_degrades_per_elo(monkeypatch):
    """No registry, a raising registry, or a nonsense multiplier -> the neutral.

    The neutral is the registry's OWN answer for "no window matched or nothing is
    known", and `capabilities_known` beside it is what distinguishes the two.
    Degrading per elo means one unpriceable model cannot blank every other rail.
    """
    import router.service as service_mod

    neutral = {
        "in_expensive_window": False,
        "price_multiplier": 1.0,
        "next_window_change": None,
        "price_windows_origin": None,
    }
    when = _PEAK_0700

    monkeypatch.setattr(service_mod, "_caps", None)
    assert RouterService._time_state("deepseek-v4-pro", None, when) == neutral

    class Exploding:
        @staticmethod
        def price_multiplier(*_a, **_kw):
            raise TypeError("stale registry")

    monkeypatch.setattr(service_mod, "_caps", Exploding)
    assert RouterService._time_state("deepseek-v4-pro", None, when) == neutral

    class Nonsense:
        @staticmethod
        def price_multiplier(*_a, **_kw):
            return "2x"          # not a number -> nothing usable to report

        @staticmethod
        def in_expensive_window(*_a, **_kw):
            return True

        @staticmethod
        def next_window_change(*_a, **_kw):
            return True          # a bool is not an hour

    monkeypatch.setattr(service_mod, "_caps", Nonsense)
    assert RouterService._time_state("deepseek-v4-pro", None, when) == neutral


def test_liveness_honours_a_declared_price_window_override(tmp_path, monkeypatch):
    """A per-elo price_windows override in YAML wins over the registry.

    `declared` beats the registry everywhere else, so an operator who corrected a
    stale window in the file must see the corrected window here too — otherwise
    the console reports a peak the router is not charging for.
    """
    import router.service as service_mod

    path = tmp_path / "router.yaml"
    path.write_text(
        yaml.safe_dump({
            "enabled": True,
            "classifier": {"model": "glm-4.6", "provider": "zai"},
            "fail_safe": {"profile": "coder", "model": "glm-4.6", "provider": "zai"},
            "blocklist": {"manual_ban": [], "fallback_chain": [],
                          "auto_breaker": {"enabled": False}},
            "rules": [],
            "default": {"profile": "coder", "model": "T1"},
            "tiers": {
                # The registry says deepseek is 2.0x at 07:00; the file says the
                # peak moved to 20:00-22:00, and the file wins.
                "T1": {"model": "deepseek-v4-pro", "provider": "deepseek",
                       "billing_mode": "metered",
                       "price_windows": [{"hours_utc": [20, 22],
                                          "multiplier": 3.0}]},
                "T2": {"model": "glm-4.6", "provider": "zai"},
                "T3": {"model": "glm-4.6", "provider": "zai"},
                "T4": {"model": "glm-4.6", "provider": "zai"},
            },
        }, sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        service_mod, "_utc_now",
        lambda: datetime(2026, 8, 17, 21, tzinfo=timezone.utc),
    )

    liveness = RouterService(path).liveness()
    entry = {e["model_key"]: e for e in liveness["models"]}["deepseek-v4-pro@deepseek"]

    # The registry's peak (06:00-10:00) is over by now, so 3.0x can only come from
    # the declared window — and the declared window is what the change instant is
    # read off too: the override ends at 22:00 UTC, one hour ahead, back to base.
    assert entry["price_multiplier"] == 3.0
    assert entry["in_expensive_window"] is True
    assert entry["next_window_change"] == {
        "hour": 22, "weekday": 0, "hours_ahead": 1, "multiplier": 1.0,
    }
    # The four-value enum is untouched by an override, same as by a registry window.
    assert {e["state"] for e in liveness["models"]} <= {
        "alive", "degraded", "quota_exhausted", "dead",
    }


def test_hot_apply_round_trips_time_cap_and_a_reordered_fallback(time_config_path):
    """A tier carrying `time_cap` survives the write path with no new hot key.

    `_HOT_KEYS` still allows exactly `tiers`, because both time knobs live INSIDE a
    tier; `_deep_merge_value` recurses into the tier mapping, so sibling keys
    survive, while the `fallback` LIST replaces wholesale — which is what makes
    reordering hops and DELETING an elo expressible at all.
    """
    service = RouterService(time_config_path)
    # Step 1: add the cap and the policy, and REORDER the two declared hops (plus
    # one more, so step 2 has something to delete).
    reordered = [
        {"model": "MiniMax-M3", "provider": "minimax", "billing_mode": "metered"},
        {"model": "kimi-k3", "provider": "moonshot", "billing_mode": "metered"},
        {"model": "glm-4.6", "provider": "zai", "billing_mode": "metered"},
    ]

    plan = service.plan({"tiers": {
        "T2": {
            "time_cap": {"max_multiplier": 1.5},
            "time_policy": {"avoid_peak": ["deepseek"], "prefer": ["MiniMax-M3"]},
            "fallback": reordered,
        },
    }})
    assert plan["valid"] is True
    assert service.apply(plan["base_hash"], plan["policy"])["ok"] is True

    written = yaml.safe_load(time_config_path.read_text(encoding="utf-8"))
    t2 = written["tiers"]["T2"]
    assert t2["time_cap"] == {"max_multiplier": 1.5}
    assert t2["time_policy"] == {"avoid_peak": ["deepseek"], "prefer": ["MiniMax-M3"]}
    assert t2["fallback"] == reordered          # replaced in the order sent
    # Untouched sibling keys survive the deep merge.
    assert t2["model"] == "deepseek-v4-pro"
    assert t2["fallback_strategy"] == "cheapest_now"
    assert t2["pin_primary"] is False
    # Sibling tiers are untouched.
    assert written["tiers"]["T4"]["time_policy"] == {"avoid_peak": ["deepseek"]}

    # Step 2: DELETE an elo by sending the shorter list. A union/index merge would
    # make deletion inexpressible, so this is the guarantee, not a side effect.
    shortened = [hop for hop in reordered if hop["model"] != "kimi-k3"]
    plan2 = service.plan({"tiers": {"T2": {"fallback": shortened}}})
    assert service.apply(plan2["base_hash"], plan2["policy"])["ok"] is True
    t2_after = yaml.safe_load(time_config_path.read_text(encoding="utf-8"))["tiers"]["T2"]
    assert t2_after["fallback"] == shortened            # 3 -> 2, kimi-k3 is gone
    assert t2_after["time_cap"] == {"max_multiplier": 1.5}   # cap survived step 2

    # Live on the next read, no restart: the cap now fires for this tier and the
    # surviving hops are the ones the operator left behind.
    assert service.policy()["tiers"]["T2"]["time_cap"] == {"max_multiplier": 1.5}
    peak = service.explain(_PLAIN_TASK, _PEAK_0200)
    assert peak["chain_plan"]["time_cap"] == {"max_multiplier": 1.5}
    assert [c["model"] for c in peak["chain_plan"]["capped"]] == ["deepseek-v4-pro"]
    assert _chain(peak) == ["MiniMax-M3", "glm-4.6"]


def test_lifting_a_cost_control_is_expressible_and_a_no_op_edit_says_so(
    time_config_path,
):
    """A mapping knob can be REMOVED, and an edit that changes nothing admits it.

    The write path could only add or overwrite: a deep merge of ``{'time_cap': {}}``
    merges no sub-keys, so the cap stayed on disk while plan() answered
    ``valid: True`` with an empty diff and apply() answered ``ok: True``. Two of the
    four mapping knobs this branch added are COST controls, and "lift the cap now"
    is the request an operator makes in a hurry — the one case where being told the
    edit succeeded while nothing happened costs money.

    Both halves are asserted here: the null spelling REMOVES, and every spelling
    that changes nothing (an empty mapping, and a removal plan's merged policy fed
    back in) is reported as ``no_op`` instead of as a committed edit.
    """
    service = RouterService(time_config_path)
    # Put T1 in the shape the finding was measured on: BOTH cost controls at once.
    setup = service.plan({"tiers": {
        "T1": {"time_policy": {"avoid_peak": ["deepseek"]}},
    }})
    assert setup["no_op"] is False
    assert service.apply(setup["base_hash"], setup["policy"])["ok"] is True
    t1 = yaml.safe_load(time_config_path.read_text(encoding="utf-8"))["tiers"]["T1"]
    assert t1["time_cap"] == {"max_multiplier": 1.5}
    assert t1["time_policy"] == {"avoid_peak": ["deepseek"]}

    # The spelling that cannot work: an empty mapping merges nothing. It is now
    # reported as the no-op it always was, on BOTH halves of the write path.
    empty = service.plan({"tiers": {"T1": {"time_cap": {}}}})
    assert (empty["valid"], empty["diff"], empty["no_op"]) == (True, "", True)
    before = time_config_path.read_bytes()
    assert service.apply(empty["base_hash"], empty["policy"]) == {
        "ok": True, "no_op": True,
        "base_hash": RouterService._hash_bytes(before),
    }
    assert time_config_path.read_bytes() == before  # and it did not write

    # The spelling that works: an explicit null removes the knob, and only that one.
    removal = {"tiers": {"T1": {"time_cap": None}}}
    planned = service.plan(removal)
    assert planned["valid"] is True
    assert planned["no_op"] is False
    assert "time_cap" not in planned["policy"]["tiers"]["T1"]
    applied = service.apply(planned["base_hash"], removal)
    assert applied["ok"] is True and applied["no_op"] is False

    lifted = yaml.safe_load(time_config_path.read_text(encoding="utf-8"))["tiers"]["T1"]
    assert "time_cap" not in lifted                      # the cap is GONE, not null
    assert lifted["time_policy"] == {"avoid_peak": ["deepseek"]}  # sibling survived
    assert lifted["fallback"][0]["model"] == "deepseek-v4-pro"
    # Live on the next read: the peak hour no longer caps this tier's elos, which
    # is the operator-visible effect they were asking for.
    assert "time_cap" not in service.policy()["tiers"]["T1"]
    peak = service.explain(_TRIVIAL_TASK, _PEAK_0200)
    assert peak["chain_plan"]["capped"] == []
    assert _chain(peak) == ["deepseek-v4-flash", "deepseek-v4-pro"]

    # A removal cannot be round-tripped through plan()['policy'] — a merge cannot
    # see absence — and that path is reported as the no-op it is, never as a
    # deletion that happened.
    restore = service.plan({"tiers": {"T1": {"time_cap": {"max_multiplier": 1.5}}}})
    assert service.apply(restore["base_hash"], restore["policy"])["ok"] is True
    assert "time_cap" in service.policy()["tiers"]["T1"]  # cap back on disk
    round_trip = service.plan(removal)
    assert round_trip["no_op"] is False                   # the PLAN removes it
    replayed = service.apply(round_trip["base_hash"], round_trip["policy"])
    assert replayed["no_op"] is True                      # the merged policy cannot
    assert service.policy()["tiers"]["T1"]["time_cap"] == {"max_multiplier": 1.5}


def test_a_null_removal_that_would_misroute_is_refused_by_the_write_gate(
    time_config_path,
):
    """Removal rides the same fail-closed gate as every other edit.

    ``null`` is a delete, not an escape hatch: dropping a tier a rule targets is a
    misroute, so lint refuses the merged result before anything is written.
    """
    service = RouterService(time_config_path)
    before = time_config_path.read_bytes()

    # T4 is the tier `hard-verbs` routes to. Deleting it breaks that rule.
    plan = service.plan({"tiers": {"T4": None}})
    assert plan["valid"] is False
    assert plan["no_op"] is False
    assert any("T4" in error for error in plan["errors"])
    assert service.apply(plan["base_hash"], {"tiers": {"T4": None}})["ok"] is False
    assert time_config_path.read_bytes() == before


def test_apply_rejects_an_invalid_time_cap_and_time_policy(time_config_path):
    """The write gate still fails closed on the time knobs."""
    service = RouterService(time_config_path)
    before = time_config_path.read_bytes()

    # A cap below the base rate could only ever empty the chain and bypass itself.
    low = service.plan({"tiers": {"T3": {"time_cap": {"max_multiplier": 0.5}}}})
    assert low["valid"] is False
    assert any("time_cap.max_multiplier" in e for e in low["errors"])

    bad_policy = service.plan({"tiers": {"T4": {"time_policy": {"avoid_peak": "zai"}}}})
    assert bad_policy["valid"] is False
    assert any("time_policy.avoid_peak" in e for e in bad_policy["errors"])

    assert service.apply(low["base_hash"], low["policy"])["ok"] is False
    assert time_config_path.read_bytes() == before


def test_config_without_time_keys_is_unaffected_by_the_hour(config_path, monkeypatch):
    """A policy with none of the new keys answers the same at every hour.

    The clock is now always injected, so this is the guard that injecting it
    changed nothing for a config that does not use it: same decision, same chain,
    no elo capped, demoted or promoted, and a preview explicitly labelled NOT
    time-relative so the console does not warn about an hour that cannot matter.
    """
    import router.service as service_mod

    service = RouterService(config_path)

    peak = service.explain("Debug a race condition", _PEAK_0700)
    offpeak = service.explain("Debug a race condition", _OFFPEAK_1500)

    for result in (peak, offpeak):
        plan = result["chain_plan"]
        assert result["mode"] == "deterministic_dry_run"
        assert result["decision"]["cause"] == "hard_rule"
        assert [hop["model"] for hop in plan["chain"]] == ["strong", "backup"]
        assert plan["strategy"] == "sequential"
        assert plan["capped"] == [] and plan["demoted"] == [] and plan["promoted"] == []
        assert plan["time_cap_bypassed"] is False
        assert "time_cap" not in plan
        assert result["preview"]["time_relative"] is False
        assert result["preview"]["time_relative_reasons"] == []

    # The plan's CONTENT is identical across the two hours. Only the hour it
    # reports differs, which is the plan being honest about when it was made and
    # not a routing difference: nothing here reads a clock.
    clock_keys = {"utc_hour", "utc_weekday"}
    assert {k: v for k, v in peak["chain_plan"].items() if k not in clock_keys} == {
        k: v for k, v in offpeak["chain_plan"].items() if k not in clock_keys
    }
    assert peak["decision"]["output"] == offpeak["decision"]["output"]
    assert peak["decision"]["matched_clauses"] == offpeak["decision"]["matched_clauses"]
    assert peak["preview"] == offpeak["preview"]
    assert peak["requires_classifier"] == offpeak["requires_classifier"]

    # liveness(): the prior states, plus neutral window fields for elos the
    # registry has never heard of — a 1.0 multiplier is "no window known", which
    # `capabilities_known: False` beside it makes unambiguous.
    monkeypatch.setattr(service_mod, "_utc_now", lambda: _PEAK_0700)
    liveness = service.liveness()
    assert {e["state"] for e in liveness["models"]} == {"alive"}
    assert all(e["capabilities_known"] is False for e in liveness["models"])
    assert all(e["price_multiplier"] == 1.0 for e in liveness["models"])
    assert all(e["in_expensive_window"] is False for e in liveness["models"])
    assert all(e["next_window_change"] is None for e in liveness["models"])


def test_read_paths_survive_a_loadable_but_wrong_config(tmp_path):
    """status()/policy() must not raise on a config that parses but is nonsense.

    ``rules: 5`` and ``tiers: nope`` load fine and lint rejects both, and status is
    the endpoint an operator opens BECAUSE the config is broken. Reporting the
    breakage in ``validation_errors`` while raising a TypeError out of the same call
    would kill the one surface that could explain it.
    """
    path = tmp_path / "router.yaml"
    path.write_text(
        "enabled: true\nrules: 5\ntiers: nope\ndefault: {}\n"
        "classifier: [not, a, mapping]\nblocklist: {auto_breaker: not-a-mapping}\n",
        encoding="utf-8",
    )
    service = RouterService(path)

    status = service.status()
    assert status["valid"] is False
    assert any("'rules' must be a list" in e for e in status["validation_errors"])
    # Degraded, never invented: no rules, no tiers, no classifier, no breaker.
    assert status["rules_count"] == 0
    assert status["tiers"] == []
    assert status["classifier"] == {"model": "", "provider": ""}
    assert status["breaker_enabled"] is False

    assert service.policy()["rules"] == []
    assert service.policy()["tiers"] == {}
    # The other read paths already held this line; assert it so a regression in
    # any one of them is caught here.
    assert service.lint()["valid"] is False
    assert service.blocklist()["manual_bans"] == []
    assert service.liveness()["models"] == []


# ---------------------------------------------------------------------------
# The capability catalogue: the read path behind the console's price audit
# ---------------------------------------------------------------------------


def _multiplier_from_served_windows(entry, hour, weekday):
    """Price one hour from the ``price_windows`` the catalogue SERVED.

    A deliberate SECOND implementation of the reading a consumer has to make —
    half-open ``[start, end)`` hours, absent ``weekdays`` meaning every day, 1.0
    when nothing matches — so the assertions below compare the audit surface with
    the running planner through a third computation instead of trusting either
    one's word. It is the console's own rule, written out in the test, because a
    helper shared with the code under test would agree with it by construction.
    """
    multiplier = 1.0
    for window in entry.get("price_windows") or []:
        start, end = window["hours_utc"]
        if not start <= hour < end:
            continue
        weekdays = window.get("weekdays")
        if weekdays is not None and weekday not in weekdays:
            continue
        multiplier = window["multiplier"]
    return multiplier


def test_capabilities_serves_the_facts_the_price_audit_needs(capability_config_path):
    """Per model: the capability facts, the billing mode, the prices, the windows."""
    catalogue = RouterService(capability_config_path).capabilities()

    assert catalogue["registry_available"] is True
    # Base rates with no hour applied — `liveness` is the surface that prices NOW.
    assert catalogue["time_agnostic"] is True
    assert catalogue["unknown_models"] == []
    assert catalogue["warnings"] == []

    flash = catalogue["models"]["deepseek-v4-flash"]
    # The finding's own case: with no /capabilities route the console rendered
    # this rail as publishing no per-token price. It publishes two.
    assert (flash["price_in"], flash["price_out"]) == (0.22, 0.66)
    assert flash["price_published"] is True
    # The capability facts the filter decides on, so a rejection is checkable.
    assert flash["context_window"] == 1_048_576
    assert flash["vision"] is False
    assert flash["tool_calling"] is True
    assert flash["structured_output"] is True
    assert flash["billing_mode"] == "metered"
    assert flash["provider"] == "deepseek"
    assert flash["in_registry"] is True
    # Windows verbatim, so a consumer can price ANY hour instead of one.
    assert flash["price_windows"] == [
        {"hours_utc": [1, 4], "weekdays": [0, 1, 2, 3, 4], "multiplier": 2.0},
        {"hours_utc": [6, 10], "weekdays": [0, 1, 2, 3, 4], "multiplier": 2.0},
    ]
    # A catalogue, not a credential store — and no free-text `notes` either,
    # which is the field a pasted key would land in.
    assert "notes" not in flash
    body = json.dumps(catalogue).lower()
    assert "api_key" not in body
    assert "secret" not in body


def test_capabilities_tells_an_unpublished_price_apart_from_a_price_of_zero(
    capability_config_path,
):
    """The distinction the panel exists for: None is not 0.0.

    An unpublished price rendered as $0 would look like the cheapest thing on the
    screen when it is merely the least KNOWN — the opposite of the truth. It used
    to be glm-5.3 that carried the case, because a plan rail published no dollars
    at all; the vendor launched its metered rate on 2026-08-27, so the unpriced
    case and the plan rail are now two different rows, and both are asserted.
    """
    models = RouterService(capability_config_path).capabilities()["models"]

    unpriced = models["glm-4.5-flash"]
    assert unpriced["price_in"] is None
    assert unpriced["price_out"] is None
    assert unpriced["price_published"] is False

    # The plan rail: priced now, and the dollars are still not what the operator
    # pays at the margin — that is what `billing_mode` says, and why `cheapest_now`
    # buckets on it rather than on this number.
    plan_rail = models["glm-5.3"]
    assert plan_rail["billing_mode"] == "plan"
    assert (plan_rail["price_in"], plan_rail["price_out"]) == (1.40, 4.40)
    assert plan_rail["price_published"] is True
    # It also declares a window, in CREDITS, so a plan rail's cost varies by hour
    # in a unit the dollar columns beside it cannot express.
    assert plan_rail["price_windows"]

    # A genuinely free rail publishes 0.0, which IS a price and survives as one.
    free_rail = models["glm-4.7-flash"]
    assert free_rail["billing_mode"] == "free"
    assert (free_rail["price_in"], free_rail["price_out"]) == (0.0, 0.0)
    assert free_rail["price_published"] is True


def test_capabilities_price_publication_agrees_with_the_running_cost_path(
    capability_config_path,
):
    """The audit surface and the path that RANKS on cost may not disagree.

    ``cheapest_now`` buckets an elo on whether ``capabilities.effective_price``
    answers at all, so a catalogue calling a rail priced where the ordering calls
    it unpriced would explain a decision the router never made. Asserted as an
    AGREEMENT between the two surfaces plus the fields actually served, never as a
    claim about one of them.
    """
    from router import capabilities as caps

    service = RouterService(capability_config_path)
    catalogue = service.capabilities()
    declared = RouterService._declared_capability_index(
        yaml.safe_load(capability_config_path.read_text(encoding="utf-8"))
    )

    assert catalogue["models"]
    for model, entry in catalogue["models"].items():
        overrides = declared.get(model) or None
        priced = caps.effective_price(model, None, overrides)

        # The flag agrees with the cost path...
        assert entry["price_published"] is (priced is not None), model
        # ...and with the two fields a console reads instead of the flag.
        both_real = all(
            isinstance(entry.get(key), (int, float))
            and not isinstance(entry.get(key), bool)
            for key in ("price_in", "price_out")
        )
        assert entry["price_published"] is both_real, model
        # A served price is the BASE rate: no hour has been folded into it.
        if priced is not None:
            assert (entry["price_in"], entry["price_out"]) == priced, model

        # And the capability half agrees with the filter's own verdict.
        assert RouterService._capabilities_known(model, overrides) is True, model


def test_capabilities_windows_reprice_the_hour_the_plan_was_made_at(time_config_path):
    """The served windows must reproduce the multipliers the planner used.

    The plan is the path that RUNS; the catalogue is the surface that DISPLAYS.
    If pricing the plan's own hour from the served windows gave a different
    number, the panel would be explaining a cost the router never applied.
    """
    service = RouterService(time_config_path)
    plan = service.explain(_HARD_TASK, at=_PEAK_0700)["chain_plan"]
    models = service.capabilities()["models"]

    # Both a windowed and a flat elo, so the agreement is not vacuous.
    assert plan["multipliers"] == {"deepseek-v4-pro": 2.0, "glm-4.6": 1.0}
    for model, multiplier in plan["multipliers"].items():
        assert model in models, model
        assert _multiplier_from_served_windows(
            models[model], plan["utc_hour"], plan["utc_weekday"]
        ) == multiplier, model

    # Every elo the plan could not describe is exactly one the catalogue omits.
    assert set(plan["unknown"]) & set(models) == set()


def test_capabilities_reads_no_clock(capability_config_path, monkeypatch):
    """Same catalogue at 07:00 and at 15:00 — this surface is time-agnostic.

    The prices are base rates and the windows are declared data; folding an hour
    in here would give the console a second, competing answer to a question
    ``liveness`` already answers at an hour it names.
    """
    import router.service as service_mod

    service = RouterService(capability_config_path)
    monkeypatch.setattr(service_mod, "_utc_now", lambda: _PEAK_0700)
    peak = service.capabilities()
    monkeypatch.setattr(service_mod, "_utc_now", lambda: _OFFPEAK_1500)

    assert service.capabilities() == peak


def test_capabilities_declared_overrides_win_and_are_named(tmp_path):
    """An operator's corrected number is served, and attributed to them.

    ``declared`` WINS over the registry on the filter's path, so it has to win
    here too; ``declared_overrides`` names which fields came from router.yaml so
    an operator can tell their own correction from the registry's data.
    """
    path = tmp_path / "router.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "enabled": True,
                "classifier": {"model": "glm-4.7", "provider": "zai"},
                "fail_safe": {"profile": "coder", "model": "glm-4.7",
                              "provider": "zai"},
                "blocklist": {"manual_ban": [], "fallback_chain": [],
                              "auto_breaker": {"enabled": False}},
                "rules": [],
                "default": {"action": "classify"},
                "tiers": {
                    "T1": {
                        # A stale registry rate and window, corrected in YAML.
                        "model": "deepseek-v4-flash", "provider": "deepseek",
                        "billing_mode": "metered",
                        "price_in": 0.11, "price_out": 0.33,
                        "price_windows": [
                            {"hours_utc": [6, 10], "multiplier": 3.0},
                        ],
                    },
                    "T2": {"model": "glm-4.7", "provider": "zai",
                           "billing_mode": "plan"},
                    "T3": {"model": "glm-4.6", "provider": "zai"},
                    "T4": {"model": "gpt-5.5", "provider": "openai-codex"},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    corrected = RouterService(path).capabilities()["models"]["deepseek-v4-flash"]

    assert (corrected["price_in"], corrected["price_out"]) == (0.11, 0.33)
    assert corrected["price_published"] is True
    assert corrected["price_windows"] == [{"hours_utc": [6, 10], "multiplier": 3.0}]
    assert corrected["declared_overrides"] == [
        "billing_mode", "price_in", "price_out", "price_windows",
    ]
    # Still the registry's model, so the untouched facts are still the registry's.
    assert corrected["in_registry"] is True
    assert corrected["context_window"] == 1_048_576
    # A copy, never a live view: a consumer mutating this cannot edit the registry.
    from router import capabilities as caps

    assert caps.MODEL_CAPABILITIES["deepseek-v4-flash"]["price_in"] == 0.22


def test_capabilities_flags_an_unknown_elo_instead_of_vouching_for_it(
    warned_config_path,
):
    """An unknown model fails OPEN with a loud flag — and stays OUT of `models`.

    A console reads the PRESENCE of an entry as "these capabilities are verified",
    so a hollow entry would silence the unverified badge on the one elo that
    routes unchecked. Dropping such an elo from a chain could empty it, so it
    still routes; it just routes visibly unverified.
    """
    service = RouterService(warned_config_path)
    catalogue = service.capabilities()

    # Unknown to the registry AND describing nothing: absent, listed, loud.
    assert "ghost-model" not in catalogue["models"]
    assert "ghost-model" in catalogue["unknown_models"]
    assert any(
        "ghost-model" in warning and "UNCHECKED" in warning
        for warning in catalogue["warnings"]
    )

    # Unknown to the registry but DESCRIBED in yaml: declared wins, so it is
    # auditable — with the operator's numbers and the provider policy names it by.
    house = catalogue["models"]["house-model"]
    assert house["context_window"] == 500_000
    assert house["vision"] is True
    assert house["provider"] == "local-rail"
    assert house["in_registry"] is False
    assert house["declared_overrides"] == ["context_window", "vision"]
    # It publishes no price, which is not a price of zero here either.
    assert house["price_published"] is False
    assert "price_in" not in house

    # The catalogue and the running filter agree, elo by elo, about which are
    # verifiable: presence here means exactly `capabilities_known` there.
    for entry in service.liveness()["models"]:
        assert (entry["model"] in catalogue["models"]) is entry[
            "capabilities_known"
        ], entry["model"]


def test_capabilities_serves_an_allowlist_not_whatever_the_registry_grows(
    capability_config_path, monkeypatch
):
    """A field the registry gains later is NOT published by default.

    This read path hands registry material to a browser. The day an entry grows a
    field carrying a credential, a passthrough would publish it with no edit and
    no review — ``capabilities_for`` merges an entry WHOLE, unknown fields
    included, so the allowlist here is the only thing standing between the two.
    """
    from router import capabilities as caps

    entry = dict(caps.MODEL_CAPABILITIES["glm-4.7"])
    entry["api_key"] = "sk-must-never-be-served"
    entry["notes"] = "an operator pasted sk-live-0000 in here"
    monkeypatch.setitem(caps.MODEL_CAPABILITIES, "glm-4.7", entry)

    catalogue = RouterService(capability_config_path).capabilities()
    served = catalogue["models"]["glm-4.7"]

    assert "api_key" not in served
    assert "notes" not in served
    assert "sk-" not in json.dumps(catalogue)
    # The facts it does serve still arrive intact.
    assert served["context_window"] == 200_000
    assert served["price_published"] is True


def test_capabilities_never_raises_and_degrades_out_loud(tmp_path, monkeypatch):
    """Read-path contract: a broken config or registry degrades, never 500s."""
    import router.service as service_mod

    # A config that parses but is nonsense: the registry does not depend on it,
    # so the catalogue is still served in full.
    path = tmp_path / "router.yaml"
    path.write_text("enabled: true\nrules: 5\ntiers: nope\n", encoding="utf-8")
    catalogue = RouterService(path).capabilities()
    assert catalogue["models"]
    assert "error" not in catalogue

    # No registry importable at all: an empty catalogue, said out loud. A console
    # renders that as "unverified", which is the honest answer for "unreadable".
    monkeypatch.setattr(service_mod, "_caps", None)
    assert RouterService(path).capabilities() == {
        "models": {},
        "unknown_models": [],
        "warnings": [],
        "registry_available": False,
        "time_agnostic": True,
    }

    # A registry that raises is the same answer, not a traceback — and the model
    # it could not describe is flagged rather than served hollow.
    class Exploding:
        MODEL_CAPABILITIES = {"boom": {"context_window": 1}}

        @staticmethod
        def capabilities_for(*_a, **_kw):
            raise TypeError("stale registry")

    monkeypatch.setattr(service_mod, "_caps", Exploding)
    exploding = RouterService(path).capabilities()
    assert exploding["models"] == {}
    assert exploding["unknown_models"] == ["boom"]
    assert exploding["warnings"]


def test_capabilities_flags_every_named_elo_when_there_is_no_registry_at_all(
    capability_config_path, monkeypatch,
):
    """No registry -> nothing is vouched for, and every elo is named as unchecked.

    The previous test's no-registry case runs against a policy that names no
    model, so it can only show the empty answer. With a real policy the branch
    that matters is reachable: ``_merged_capabilities`` has to answer None for
    every elo, which the caller must report as ``unknown_models`` + a warning
    rather than as an entry. A console reads the PRESENCE of an entry as "this
    elo's capabilities are verified", so a hollow entry would silence the
    unknown-model flag on exactly the elos routing unchecked.
    """
    import router.service as service_mod

    named = sorted(RouterService._policy_provider_index(
        yaml.safe_load(capability_config_path.read_text(encoding="utf-8"))
    ))
    assert named, "a policy naming no elo could not show this"

    monkeypatch.setattr(service_mod, "_caps", None)
    catalogue = RouterService(capability_config_path).capabilities()

    assert catalogue["registry_available"] is False
    assert catalogue["models"] == {}, "an unverifiable elo is never served as a fact"
    assert catalogue["unknown_models"] == named
    assert len(catalogue["warnings"]) == len(named)
    assert all("routes UNCHECKED" in warning for warning in catalogue["warnings"])

    # The other surface answers the OPPOSITE, and deliberately: liveness's
    # ``capabilities_known`` fails OPEN, because with no registry importable
    # nothing is provably unknown and painting every rail unverified would nag
    # about a condition ``registry_available: False`` already states once. The two
    # are pinned together here so the difference stays a decision rather than
    # becoming a surprise: the catalogue refuses to VOUCH for an elo it cannot
    # describe, liveness refuses to ACCUSE one.
    assert all(RouterService._capabilities_known(model, None) is True
               for model in named)
    assert all(entry["capabilities_known"] is True
               for entry in RouterService(capability_config_path).liveness()["models"])


def test_capabilities_reports_a_registry_it_cannot_even_enumerate(tmp_path, monkeypatch):
    """A registry keyed by something unsortable is an error, not a traceback.

    Provoked with a real corrupt registry rather than a patched raise: this file
    is deployed by copy, so it can land beside a capabilities.py whose dict picked
    up a non-string model id, and then ``sorted(candidates)`` — the call that makes
    two polls of this endpoint byte-identical — is what fails.
    """
    import router.service as service_mod

    path = tmp_path / "router.yaml"
    path.write_text("enabled: true\n", encoding="utf-8")

    class Unsortable:
        MODEL_CAPABILITIES = {2.5: {"context_window": 1}, "glm-4.7": {"vision": True}}

        @staticmethod
        def capabilities_for(model, declared=None):  # pragma: no cover - never reached
            return {"context_window": 1}

    monkeypatch.setattr(service_mod, "_caps", Unsortable)
    catalogue = RouterService(path).capabilities()

    assert "could not compose capabilities" in catalogue["error"]
    # The documented shape survives the failure, so a console renders "unverified"
    # instead of branching on keys that are suddenly absent.
    assert catalogue["models"] == {}
    assert catalogue["unknown_models"] == []
    assert catalogue["registry_available"] is True  # importable, just unreadable
    assert catalogue["time_agnostic"] is True


def test_a_malformed_tier_degrades_the_read_paths_and_is_refused_by_the_gate(
    capability_config_path,
):
    """``fallback: 5`` — a scalar where a list belongs — must not 500 anything.

    router.yaml is hot and hand-editable, so this typo reaches the read paths
    before any gate sees it. The three answers are deliberately different and all
    three are asserted: the catalogue degrades PARTIALLY (it loses the providers
    policy declares and keeps every registry fact), liveness degrades WHOLESALE
    because the malformed walk is the source of its model list, and ``lint`` — the
    fail-closed write gate — refuses the file outright.
    """
    config = yaml.safe_load(capability_config_path.read_text(encoding="utf-8"))
    config["tiers"]["T4"]["fallback"] = 5
    capability_config_path.write_text(yaml.safe_dump(config, sort_keys=False),
                                      encoding="utf-8")
    service = RouterService(capability_config_path)

    # The provider index is the guarded half: it degrades to empty rather than
    # letting the malformed walk out of a read path.
    assert RouterService._policy_provider_index(config) == {}
    with pytest.raises(TypeError):
        RouterService._policy_references(config, [])

    catalogue = service.capabilities()
    assert "error" not in catalogue
    assert catalogue["models"]["glm-5.3"]["context_window"] == 1_000_000

    liveness = service.liveness()
    assert liveness["models"] == []
    assert "could not compose liveness" in liveness["error"]

    assert any("fallback" in error for error in service.lint()["errors"])


def test_the_local_price_test_agrees_with_the_registry_it_stands_in_for(
    capability_config_path, monkeypatch,
):
    """The fallback price test must not be able to answer differently.

    ``_price_published`` asks ``capabilities.effective_price`` — the function
    ``cheapest_now`` ranks on — and only falls back to a local ``price_in``/
    ``price_out`` test when the installed registry cannot be asked (this file is
    deployed by copy, so it can land beside a capabilities.py that predates
    ``effective_price``). A fallback that graded prices differently would make the
    audit panel disagree with the ordering it exists to audit, on exactly the
    installs nobody is watching.

    Both ways of losing the registry are exercised, because they take different
    branches: absent entirely, and present but too old to answer.
    """
    import router.service as service_mod
    from router import capabilities as caps

    service = RouterService(capability_config_path)
    declared = RouterService._declared_capability_index(
        yaml.safe_load(capability_config_path.read_text(encoding="utf-8"))
    )
    # The reference answers, taken from the running cost path while it is live.
    catalogue = service.capabilities()["models"]
    expected = {
        model: entry["price_published"] for model, entry in catalogue.items()
    }
    assert set(expected.values()) == {True, False}, (
        "both verdicts must be represented or the agreement proves nothing"
    )

    class TooOldToAnswer:
        """A registry that can describe an elo but cannot price one."""

        MODEL_CAPABILITIES = caps.MODEL_CAPABILITIES
        capabilities_for = staticmethod(caps.capabilities_for)
        # No effective_price at all -> AttributeError, the real shape of "older".

    for stand_in in (None, TooOldToAnswer):
        monkeypatch.setattr(service_mod, "_caps", stand_in)
        for model, published in expected.items():
            merged = caps.capabilities_for(model, declared.get(model) or None)
            assert RouterService._price_published(
                model, declared.get(model), merged
            ) is published, (model, stand_in)

    # And the local test grades a BOOL as unpriced: True is an int in Python, so
    # a price of True would otherwise read as a published rate of 1.0.
    monkeypatch.setattr(service_mod, "_caps", None)
    assert RouterService._price_published(
        "glm-4.7", None, {"price_in": True, "price_out": True}
    ) is False
    assert RouterService._price_published(
        "glm-4.7", None, {"price_in": 0.0, "price_out": 0.0}
    ) is True, "a genuinely free rail publishes 0.0, which IS a price"


def test_an_older_registrys_bare_hour_keeps_the_hour_and_invents_no_weekday(
    capability_config_path, monkeypatch,
):
    """``next_window_change`` as a bare int is the pre-weekday spelling.

    A weekday-gated window makes a bare hour ambiguous by up to two days, so the
    countdown fields report None rather than being guessed as "today" — guessing
    is the 45-hours-wrong error the richer shape was introduced to remove. The
    hour is still carried, because an older registry can genuinely answer that
    much and dropping it would lose a fact nobody disputes.
    """
    import router.service as service_mod
    from router import capabilities as caps

    when = _PEAK_0700

    class PreWeekday:
        MODEL_CAPABILITIES = caps.MODEL_CAPABILITIES
        capabilities_for = staticmethod(caps.capabilities_for)
        price_multiplier = staticmethod(caps.price_multiplier)
        in_expensive_window = staticmethod(caps.in_expensive_window)

        @staticmethod
        def next_window_change(model, at, declared=None):
            richer = caps.next_window_change(model, at, declared)
            return None if richer is None else richer["hour"]

    monkeypatch.setattr(service_mod, "_caps", PreWeekday)
    state = RouterService._time_state("deepseek-v4-pro", None, when)

    assert state["next_window_change"] == {
        "hour": caps.next_window_change("deepseek-v4-pro", when)["hour"],
        "weekday": None,
        "hours_ahead": None,
        "multiplier": None,
    }
    # The multiplier and the window flag are unaffected — one key, one shape.
    assert state["price_multiplier"] == caps.price_multiplier("deepseek-v4-pro", when)
    assert state["in_expensive_window"] is True

    # The richer registry's answer is carried through whole, and as a COPY, so a
    # caller editing the liveness payload cannot edit the registry's answer.
    monkeypatch.setattr(service_mod, "_caps", caps)
    live = RouterService._time_state("deepseek-v4-pro", None, when)
    registry_answer = caps.next_window_change("deepseek-v4-pro", when)
    assert live["next_window_change"] == registry_answer
    live["next_window_change"]["hour"] = 99
    assert caps.next_window_change("deepseek-v4-pro", when) == registry_answer


def test_a_top_level_block_is_removable_by_null_and_still_lint_gated(config_path):
    """``{'blocklist': None}`` drops the whole block — same rule, one level up.

    A deep merge can only add or overwrite, so removal is spelled as an explicit
    null at EVERY level; the top level is where a whole subsystem is lifted (the
    blocklist is the manual bans plus the auto-breaker), which is the kind of thing
    an operator does in a hurry. The gate is unchanged: dropping ``tiers`` is a
    misroute and lint refuses it before anything is written.
    """
    service = RouterService(config_path)
    assert "blocklist" in yaml.safe_load(config_path.read_text(encoding="utf-8"))

    planned = service.plan({"blocklist": None})
    assert planned["valid"] is True
    assert planned["no_op"] is False
    assert "blocklist" not in planned["policy"]

    assert service.apply(planned["base_hash"], {"blocklist": None})["ok"] is True
    on_disk = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "blocklist" not in on_disk, "the block is GONE, not present-and-null"
    assert on_disk["tiers"]["T4"]["model"] == "strong", "siblings untouched"
    # Idempotent: removing what is already absent changes nothing and says so.
    assert service.plan({"blocklist": None})["no_op"] is True

    # And the fail-closed gate still owns which removals may happen.
    before = config_path.read_bytes()
    refused = service.plan({"tiers": None})
    assert refused["valid"] is False
    assert any("tiers" in error for error in refused["errors"])
    assert service.apply(refused["base_hash"], {"tiers": None})["ok"] is False
    assert config_path.read_bytes() == before


def test_the_attempted_head_degrade_matches_the_accessor_it_stands_in_for(
    tmp_path, monkeypatch, config_path,
):
    """An older decision_log has no accessor, and the degrade must not lie.

    ``_attempted_head`` is a pass-through to ``decision_log.attempted_head_of``
    because there must be exactly one definition of "the head". The local branch
    exists only for a decision_log that predates the accessor — and every entry
    such a build wrote has no ``attempted_model``, so for those entries the two
    answers have to be identical. Asserting the pair is the whole point: a degrade
    that answered differently would put a second definition back.
    """
    import router.service as service_mod
    from router.decision_log import attempted_head_of

    pre_feature = [
        {"ts": 1.0, "cause": "hard_rule", "task": "a",
         "output": {"model": "glm-5.3", "provider": "zai"}},
        {"ts": 2.0, "cause": "blocklist_veto", "task": "b", "output": {}},
        {"ts": 3.0, "cause": "classifier", "task": "c",
         "output": {"model": "gpt-5.5"}},
    ]
    assert service_mod._attempted_head_of is not None, "the accessor must be live"
    for entry in pre_feature:
        assert RouterService._attempted_head(entry) == attempted_head_of(entry)

    _seed_traces(tmp_path, monkeypatch, pre_feature)
    service = RouterService(config_path)
    with_accessor = service.routes()

    monkeypatch.setattr(service_mod, "_attempted_head_of", None)
    for entry in pre_feature:
        assert RouterService._attempted_head(entry) == attempted_head_of(entry), entry
    assert service.routes() == with_accessor, (
        "the Decisions list must read the same on a build without the accessor"
    )
    # Malformed entries answer ("", "") on the degraded branch too, rather than
    # raising out of the reader that serves the tab.
    for junk in (None, "", [], {}, {"output": None}, {"output": "x"}):
        assert RouterService._attempted_head(junk) == ("", "")

    # The one case the degrade CANNOT reproduce, asserted so it is deliberate: an
    # entry written by the CURRENT writer names the head it attempted, and a build
    # with no accessor can only report the declared tier primary.
    filtered = {"ts": 4.0, "cause": "keyword_match", "task": "d", "output": {
        "model": "glm-5.3", "provider": "zai",
        "attempted_model": "gpt-5.6-luna", "attempted_provider": "openai-codex",
    }}
    assert RouterService._attempted_head(filtered) == ("glm-5.3", "zai")
    monkeypatch.setattr(service_mod, "_attempted_head_of", attempted_head_of)
    assert RouterService._attempted_head(filtered) == ("gpt-5.6-luna", "openai-codex")


# ---------------------------------------------------------------------------
# Loading shape: the DECISIONS surface must be live under the PACKAGE name
# ---------------------------------------------------------------------------
#
# Every other test in this file imports ``router.service`` — the FLAT shape, where
# ``router`` is a top-level package on sys.path. Production is not that shape:
# Hermes loads this plugin as ``hermes_plugins.<slug>.router.service``, where
# ``router`` is not a top-level package at all. ``_trace_files`` and ``routes``
# resolved ``durable_decision_log`` by the ABSOLUTE name only, from inside their
# function bodies, so under the production shape BOTH raised
# ``ModuleNotFoundError: No module named 'router'``: /routes and /routes?id= 500'd
# on every call and the Decisions tab was dead in production while this file was
# green, because this file only ever loaded the other shape. (The module-scope
# imports in service.py were swept; these two were missed for being indented.)
#
# The probe runs in a CHILD interpreter whose sys.path cannot reach this repo, so
# the absolute name is genuinely unresolvable — the production condition itself
# rather than a simulation. In-process it would prove NOTHING: ``router.*`` is
# already in this interpreter's sys.modules from the import at the top of this
# file, so the absolute fallback would find the cached top-level copies and every
# assertion would pass with the defect fully restored. The child asserts that
# unreachability itself, and the parent asserts it was asserted.
#
# What it checks is the AGREEMENT: the rows the package shape serves must EQUAL the
# rows this file's flat-shape service serves for the SAME trace file. Asserting
# only that the child returned something would pass on an empty list, and asserting
# only the flat shape is exactly the mistake that hid this.

_PLUGIN_SLUG = "delegate_profile"

#: The seeded trace, defined once as bytes because the corruption is byte-level.
#: It carries, deliberately: an entry from BEFORE ``attempted_model`` existed; an
#: entry whose attempted head DIFFERS from the declared tier primary (so a
#: projection that dropped the distinction cannot pass); a corrupt line; a line
#: with a torn multi-byte sequence; and one older entry in the rotated backup, so
#: the cascade ``_trace_files`` walks — the function whose import was broken — is
#: exercised rather than just the current file.
_SHAPE_TRACE = b"\n".join([
    json.dumps({"ts": 2.0, "cause": "hard_rule", "task": "pre-feature",
                "output": {"model": "gpt-5.5", "provider": "openai-codex"}}).encode(),
    b"{ this is not json",
    b'{"ts": 2.5, "cause": "classifier", "task": "\xff\xfe torn"}',
    json.dumps({"ts": 3.0, "cause": "keyword_match", "task": "filtered",
                "output": {"model": "glm-5.3", "provider": "zai",
                           "attempted_model": "gpt-5.6-luna",
                           "attempted_provider": "openai-codex"},
                "steps": [{"stage": "rules"}]}).encode(),
]) + b"\n"

_SHAPE_TRACE_BACKUP = json.dumps(
    {"ts": 1.0, "cause": "size_rule", "task": "rotated-out",
     "output": {"model": "glm-4.7", "provider": "zai"}}
).encode() + b"\n"

_SHAPE_TOKEN = "decisions-shape-probe-token"

#: Runs in the child. Reads the trace through the SERVICE and through the real
#: sidecar dispatcher, because the dispatcher is what production calls and the
#: dispatcher is where the ModuleNotFoundError surfaced as a dropped connection.
_SHAPE_PROBE = '''
import json
import pathlib
import sys

out = {}

# Non-vacuity, checked from inside: if the top-level name were reachable here the
# absolute import could satisfy every assertion in the parent with the defect
# restored, and this probe would be testing nothing.
try:
    import router  # noqa: F401
except ImportError:
    out["absolute_name_reachable"] = False
else:
    out["absolute_name_reachable"] = True

from hermes_plugins.__SLUG__.router.service import RouterService
from hermes_plugins.__SLUG__.router.one_sidecar import SidecarApp

out["module"] = RouterService.__module__
token = pathlib.Path("token")
token.write_text("__TOKEN__")
service = RouterService("router.yaml")
app = SidecarApp(service, token_path=lambda: token)
auth = {"X-Hermes-Sidecar-Token": "__TOKEN__"}

out["trace_files"] = [str(path) for path in service._trace_files()]
out["routes"] = service.routes()
rid = out["routes"]["routes"][0]["id"]
out["route"] = service.route(rid)
out["dispatch"] = app.dispatch("GET", "/routes", auth)
out["dispatch_one"] = app.dispatch("GET", "/routes", auth, {"id": [rid]})

json.dump(out, sys.stdout)
'''


def _build_plugin_package(root):
    """Lay out ``hermes_plugins.<slug>.router`` under ``root``.

    COPIED, never symlinked: the child must not be able to reach this repo by any
    path other than the one under test.
    """
    import shutil

    import router.service as service_mod

    pkg = root / "hermes_plugins" / _PLUGIN_SLUG
    pkg.mkdir(parents=True)
    shutil.copytree(
        Path(service_mod.__file__).resolve().parent,
        pkg / "router",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    (root / "hermes_plugins" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    return pkg


def _probe_decisions_under_package_shape(root, trace):
    """Read the trace in a child interpreter that can only see the package shape."""
    import os
    import subprocess
    import sys

    _build_plugin_package(root)
    # routes() never reads the policy, but RouterService is constructed with one
    # and the child must not have to reach outside its own root for anything.
    (root / "router.yaml").write_text("enabled: true\n", encoding="utf-8")
    script = root / "decisions_probe.py"
    script.write_text(
        _SHAPE_PROBE.replace("__SLUG__", _PLUGIN_SLUG).replace("__TOKEN__", _SHAPE_TOKEN)
    )

    # cwd AND PYTHONPATH are the package root and nothing else. Python prepends
    # cwd to sys.path, so leaving it at the repo root would put `router` back
    # within reach and hand the absolute import a way to succeed.
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        cwd=str(root),
        env={**os.environ,
             "PYTHONPATH": str(root),
             "HERMES_ROUTE_TRACE_FILE": str(trace)},
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


class TestTheDecisionsSurfaceIsLiveUnderHermesPluginPackageShape:
    """The shape production loads must serve the traces the flat shape serves."""

    @pytest.fixture(scope="class")
    def probe(self, tmp_path_factory):
        """One child interpreter, plus the flat-shape answer for the same file."""
        root = tmp_path_factory.mktemp("decisions_shape")
        trace = root / "state" / "routes.jsonl"
        trace.parent.mkdir(parents=True, exist_ok=True)
        trace.write_bytes(_SHAPE_TRACE)
        trace.with_suffix(trace.suffix + ".1").write_bytes(_SHAPE_TRACE_BACKUP)

        # The flat-shape reading of the SAME file, taken through this file's
        # ordinary import. pytest.MonkeyPatch rather than the fixture because this
        # fixture is class-scoped: one child launch answers every case below.
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("HERMES_ROUTE_TRACE_FILE", str(trace))
            service = RouterService(root / "router.yaml")
            flat = service.routes()
            flat_one = service.route(flat["routes"][0]["id"])
            flat_files = [str(path) for path in service._trace_files()]

        child = _probe_decisions_under_package_shape(root, trace)
        return {"child": child, "flat": flat, "flat_one": flat_one,
                "flat_files": flat_files, "trace": trace}

    def test_the_child_really_could_not_reach_the_absolute_name(self, probe):
        """Without this the whole probe is vacuous — see the section comment."""
        assert probe["child"]["absolute_name_reachable"] is False
        assert probe["child"]["module"] == (
            f"hermes_plugins.{_PLUGIN_SLUG}.router.service"
        )

    def test_routes_answers_at_all_under_the_package_shape(self, probe):
        """The regression itself: this call raised ModuleNotFoundError.

        Both function-scope imports are on this path — ``routes`` resolves
        ``routes_path`` for ``trace_path`` and ``_trace_files`` resolves it plus
        ``_TRACE_BACKUPS`` for the cascade — so a payload with a resolved path and
        a back-filled row is what proves both.
        """
        child = probe["child"]
        assert child["routes"]["trace_path"] == str(probe["trace"])
        assert child["trace_files"] == probe["flat_files"]
        assert len(child["trace_files"]) == 4  # current + .1 .. .3
        assert [row["task"] for row in child["routes"]["routes"]] == [
            "filtered", "pre-feature", "rotated-out",
        ]

    def test_the_two_shapes_serve_the_same_rows(self, probe):
        """The agreement, on every field of every row."""
        assert probe["child"]["routes"] == probe["flat"]
        assert probe["child"]["route"] == probe["flat_one"]

    def test_the_package_shape_projection_actually_did_the_work(self, probe):
        """Agreement would also hold if BOTH shapes served nothing.

        So this pins the side that has to be non-trivial: the corrupt and torn
        lines are skipped, the rotated backup is back-filled, and the row whose
        chain was filtered names the elo that RAN with the tier kept beside it.
        """
        routes = probe["child"]["routes"]
        assert routes["count"] == 3, "two junk lines skipped, one backup back-filled"

        filtered = routes["routes"][0]
        assert (filtered["model"], filtered["provider"]) == (
            "gpt-5.6-luna", "openai-codex"
        )
        assert filtered["declared_model"] == "glm-5.3"
        # A pre-feature entry has no attempted head, so the declared pair IS the
        # honest answer — both fields say so rather than one going blank.
        pre_feature = routes["routes"][1]
        assert (pre_feature["model"], pre_feature["declared_model"]) == (
            "gpt-5.5", "gpt-5.5"
        )

    def test_the_sidecar_dispatcher_serves_both_routes_under_the_package_shape(
        self, probe,
    ):
        """The surface production actually calls, not just the method behind it.

        The sidecar answers an exception with a dropped connection, so this is the
        form the operator met: a Decisions tab that stays empty with nothing in the
        UI to say why.
        """
        child = probe["child"]
        status, body = child["dispatch"]
        assert status == 200
        assert body == probe["flat"]

        status_one, entry = child["dispatch_one"]
        assert status_one == 200
        assert entry == probe["flat_one"]
        assert entry["steps"] == [{"stage": "rules"}], "the full trace, not the row"


# ---------------------------------------------------------------------------
# The preview's size bound vs the size production actually routes on
# ---------------------------------------------------------------------------
#
# ``_DEFAULT_MAX_PROMPT_CHARS`` (1 MiB) is ~291k estimated tokens, and the shipped
# ``huge-context-read`` rule fires above 400k — so the one rule whose entire
# subject is a giant read cannot be previewed at the default bound: explain()
# answers the operator asking "why did this turn go to T3?" with a ValueError, for
# the very input that triggered it. The bound STAYS (it is the only thing between
# one HTTP request and unbounded CPU on a linear scan), so the two tests below pin
# the two halves of that decision: the refusal has to NAME the limitation and the
# escape hatches, and lifting the bound has to make the preview agree with the
# path that ran — otherwise the escape hatch leads somewhere else.


def _chars_per_token():
    """The router's chars-per-token ratio, MEASURED rather than copied.

    ``signals`` owns the constant. A literal 3.6 here would be a second copy that
    drifts, and this test would then compute a threshold the router does not use —
    which is the same class of defect it is about.
    """
    from router.signals import extract

    probe = 36_000
    return probe / extract("x" * probe)["est_input_tokens"]


def _huge_context_rule():
    """The shipped rule, read off the live policy rather than restated."""
    config = yaml.safe_load(LIVE_POLICY.read_text(encoding="utf-8"))
    rule = next(r for r in config["rules"] if r.get("id") == "huge-context-read")
    return config, rule


def _prompt_that_fires_the_rule():
    """The smallest composed prompt whose SIZE ALONE clears the rule's threshold.

    Size alone, deliberately: ``est_input_tokens`` also charges for referenced
    files and whole-repo reads, so a prompt mentioning 28 files could clear 400k
    inside the bound. The turn this rule exists for is a big READ, and that is the
    case the bound cannot reach.
    """
    import math

    _config, rule = _huge_context_rule()
    threshold = rule["when"]["est_input_tokens"]["gt"]
    needed = math.ceil(threshold * _chars_per_token()) + 1
    filler = "WARN retry scheduled for the nightly job\n"
    return ("Context: " + filler * (needed // len(filler) + 1))[:needed], threshold


def test_the_default_preview_bound_cannot_reach_the_shipped_huge_context_rule(
    config_path,
):
    """The limitation, pinned as a fact, and stated where the operator meets it.

    Asserted rather than fixed by raising the bound: the scans ``signals.extract``
    runs are linear (~0.35 ms per 1000 chars), this is an HTTP read path, and the
    sidecar puts no limit on a request body — so an unbounded preview is a request
    that can cost arbitrary CPU. What was wrong was not the number but the silence:
    a bare "exceeds N characters" reads as "the router cannot route a turn this
    big", which is false.
    """
    import router.service as service_mod

    prompt, threshold = _prompt_that_fires_the_rule()
    service = RouterService(config_path)

    # The finding itself: the rule's own threshold is out of reach of the default.
    assert len(prompt) > service_mod._DEFAULT_MAX_PROMPT_CHARS
    assert service._max_prompt_chars == service_mod._DEFAULT_MAX_PROMPT_CHARS
    with pytest.raises(ValueError) as raised:
        service.explain("summarise this log", prompt_text=prompt)
    message = str(raised.value)

    # The refusal names the bound, that it is THIS surface's and not the router's,
    # and both ways past it — because the error is the operator's only encounter
    # with the limitation.
    assert str(service._max_prompt_chars) in message
    assert "not a routing limit" in message
    assert "max_prompt_chars" in message
    assert "--prompt-text" in message

    # And the escape hatch the message names is real, on both surfaces it names.
    import argparse

    from router import cli

    assert cli.resolve_prompt(argparse.Namespace(
        task="summarise this log", prompt_text=prompt,
    )) == (prompt, "prompt_text")
    lifted = RouterService(config_path, max_prompt_chars=len(prompt))
    assert lifted._resolve_prompt("summarise this log", prompt) == (
        prompt, "prompt_text"
    )
    # Nothing about the refusal depends on the rule's exact threshold, but a
    # policy that lowered it under the bound would make this test vacuous.
    assert threshold > 0


def test_the_refusal_reaches_the_operator_on_the_surface_that_displays_it(
    config_path, tmp_path,
):
    """A limitation stated only in an exception is stated nowhere.

    The bound is argued in this module's comments and named in the ValueError, but
    an operator meets it in the CONSOLE — so the thing that has to carry the
    sentence is the HTTP surface, and that is asserted here through the real
    dispatcher instead of assumed from the raise site. A 400 whose body read "bad
    request" would leave precisely the impression the message exists to prevent:
    that the ROUTER cannot route a turn this big. Same reason the escape hatch is
    exercised on the same surface — advice an operator cannot follow where they
    are standing is worse than the refusal alone.
    """
    from router.one_sidecar import SidecarApp

    token = tmp_path / "sidecar-token"
    token.write_text("shape-token", encoding="utf-8")
    auth = {"X-Hermes-Sidecar-Token": "shape-token"}
    prompt, _threshold = _prompt_that_fires_the_rule()
    body = {"task": "summarise this log", "prompt_text": prompt}

    status, payload = SidecarApp(
        RouterService(config_path), token_path=lambda: token
    ).dispatch("POST", "/explain", auth, None, body)

    assert status == 400
    message = payload["error"]
    assert str(RouterService(config_path)._max_prompt_chars) in message
    assert "not a routing limit" in message
    assert "max_prompt_chars" in message
    assert "--prompt-text" in message

    # The hatch the message names, on the surface that named it: the SAME request
    # against a service whose bound was lifted answers the question rather than a
    # different one, and says which text it measured.
    lifted_status, lifted = SidecarApp(
        RouterService(config_path, max_prompt_chars=len(prompt)),
        token_path=lambda: token,
    ).dispatch("POST", "/explain", auth, None, body)

    assert lifted_status == 200
    assert lifted["preview"]["sized_from"] == "prompt_text"
    assert lifted["preview"]["prompt_chars"] == len(prompt)
    assert lifted["features"]["est_input_tokens"] > 400_000


def test_a_lifted_bound_previews_the_decision_the_running_path_really_made(tmp_path):
    """Past the bound, the preview must reproduce production — not merely answer.

    The turn is routed for real through ``adapter.route`` (the path that RUNS) and
    previewed through ``RouterService.explain`` (the surface that DISPLAYS), at one
    injected clock, off the shipped policy, with the same composed prompt. The
    assertion is that the two agree: rule, cause, tier and the elo the executor
    would dispatch first. An operator who follows the refusal's advice has to land
    on the decision that happened, or the advice is worse than the refusal.
    """
    from router.decision_log import DecisionLog
    from router.adapter import route as adapter_route

    prompt, _threshold = _prompt_that_fires_the_rule()
    goal = "summarise this log"
    config, _rule = _huge_context_rule()

    log = DecisionLog()
    ran = adapter_route(goal, config, decision_log=log, prompt_text=prompt,
                        now=_OFFPEAK_1500)
    entry = log.tail(1)[0]

    shown = RouterService(
        LIVE_POLICY, max_prompt_chars=len(prompt)
    ).explain(goal, at=_OFFPEAK_1500, prompt_text=prompt)
    decision = shown["decision"]

    # Non-vacuity: this is the rule the bound cannot reach, and the goal line on
    # its own does not come near it.
    assert entry["rule_id"] == "huge-context-read"
    assert shown["features"]["est_input_tokens"] > 400_000
    goal_only = RouterService(LIVE_POLICY).explain(goal, at=_OFFPEAK_1500)
    assert goal_only["decision"]["matched_rule_id"] != "huge-context-read"

    # The agreement, on the fields a decision IS.
    assert decision["matched_rule_id"] == entry["rule_id"]
    assert decision["cause"] == entry["cause"]
    assert decision["output"]["model"] == ran["model"]
    assert decision["output"]["provider"] == ran["provider"]
    assert decision["output"]["profile"] == ran["profile"]
    # ...and on the elo the executor would actually dispatch first, which is the
    # head of the planned chain rather than the declared tier primary.
    assert _executor_targets(ran)[0] == (
        shown["chain_plan"]["chain"][0]["model"],
        shown["chain_plan"]["chain"][0]["provider"],
    )
    # The preview says which text it measured, so this answer cannot be mistaken
    # for the goal-sized one above.
    assert shown["preview"]["sized_from"] == "prompt_text"
    assert shown["preview"]["prompt_chars"] == len(prompt)


def test_explain_refuses_a_prompt_text_that_is_not_a_string(config_path):
    """A non-string is the CALLER's bug: a clear 400, not an AttributeError.

    ``_resolve_prompt`` runs before anything is measured, so a JSON number or list
    arriving from a POST body must be refused there rather than reaching
    ``signals.extract`` and failing somewhere an operator cannot read.
    """
    service = RouterService(config_path)
    for junk in (17, 3.5, [], {}, None, object()):
        with pytest.raises(ValueError, match="prompt_text must be a string"):
            service.explain(_TRIVIAL_TASK, prompt_text=junk)
    # ``None`` is refused here too, and that is not a gap: turning an OMITTED JSON
    # field into "" is the HTTP edge's job (``one_sidecar._omitted_as_text``), so
    # a None arriving this far is a caller that meant something by it. The empty
    # string is the spelling that means "measure the task".
    assert service.explain(_TRIVIAL_TASK, prompt_text="")["preview"]["sized_from"] == (
        "task"
    )


# ---------------------------------------------------------------------------
# Compaction block (spec t_6f911222): the declarative model + fallback choice.
# ---------------------------------------------------------------------------

_TIERS = {
    "T1": {
        "model": "glm-4.7",
        "provider": "zai",
        "fallback": [
            {"model": "gpt-5.6-luna", "provider": "openai-codex"},
            {"model": "mimo-v2.5", "provider": "xiaomi"},
        ],
    },
}


def _block(**overrides):
    base = {"provider": "zai", "model": "glm-4.5-flash", "fallback_mode": "tier:T1"}
    base.update(overrides)
    return base


def test_compaction_absent_is_noop():
    aux, errors = resolve_compaction_auxiliary(None, _TIERS)
    assert aux == {}
    assert errors == []


def test_compaction_tier_reference_resolves_the_full_queue():
    aux, errors = resolve_compaction_auxiliary(_block(), _TIERS)
    assert errors == []
    assert aux == {
        "provider": "zai",
        "model": "glm-4.5-flash",
        "fallback_chain": [
            {"provider": "zai", "model": "glm-4.7"},
            {"provider": "openai-codex", "model": "gpt-5.6-luna"},
            {"provider": "xiaomi", "model": "mimo-v2.5"},
        ],
    }


def test_compaction_standalone_resolves_the_declared_chain():
    aux, errors = resolve_compaction_auxiliary(
        _block(
            fallback_mode="standalone",
            fallback_chain=[
                {"provider": "deepseek", "model": "deepseek-v4-pro"},
            ],
        ),
        _TIERS,
    )
    assert errors == []
    assert aux["fallback_chain"] == [
        {"provider": "deepseek", "model": "deepseek-v4-pro"}
    ]


def test_compaction_base_url_is_carried_when_set():
    aux, errors = resolve_compaction_auxiliary(
        _block(base_url="https://example.test/v1"), _TIERS
    )
    assert errors == []
    assert aux["base_url"] == "https://example.test/v1"


def test_compaction_references_a_missing_group_by_name():
    aux, errors = resolve_compaction_auxiliary(_block(fallback_mode="tier:T9"), _TIERS)
    assert aux == {}
    assert any("T9" in e for e in errors)
    assert any("does not exist" in e for e in errors)


def test_compaction_refuses_an_unknown_model_by_name():
    aux, errors = resolve_compaction_auxiliary(
        _block(model="no-such-model-anywhere"), _TIERS
    )
    assert aux == {}
    assert any("no-such-model-anywhere" in e for e in errors)
    assert any("not in the capability registry" in e for e in errors)


def test_compaction_refuses_a_model_below_the_required_window():
    # glm-4.5-flash advertises 131_072 in the registry; ask for far more.
    aux, errors = resolve_compaction_auxiliary(
        _block(required_provider_context_window=999_999), _TIERS
    )
    assert aux == {}
    assert any("below the required" in e for e in errors)


def test_compaction_accepts_a_model_that_clears_the_required_window():
    aux, errors = resolve_compaction_auxiliary(
        _block(required_provider_context_window=100_000), _TIERS
    )
    assert errors == []


def test_compaction_requires_provider_and_model():
    aux, errors = resolve_compaction_auxiliary({}, _TIERS)
    assert aux == {}
    assert any("compaction.provider" in e for e in errors)
    assert any("compaction.model" in e for e in errors)


def test_compaction_must_be_a_mapping():
    aux, errors = resolve_compaction_auxiliary("not-a-mapping", _TIERS)
    assert aux == {}
    assert errors == ["compaction must be a mapping"]


def test_compaction_refuses_an_unknown_fallback_mode():
    aux, errors = resolve_compaction_auxiliary(_block(fallback_mode="bogus"), _TIERS)
    assert aux == {}
    assert any("fallback_mode must be" in e for e in errors)


def test_compaction_refuses_a_non_string_or_blank_fallback_mode():
    for bad in (17, "", "   "):
        aux, errors = resolve_compaction_auxiliary(_block(fallback_mode=bad), _TIERS)
        assert aux == {}, f"fallback_mode={bad!r} should be refused"
        assert any("fallback_mode must be" in e for e in errors)


def test_compaction_standalone_requires_a_non_empty_fallback_chain():
    for chain in (None, [], [{"provider": "deepseek"}], ["loose"]):
        aux, errors = resolve_compaction_auxiliary(
            _block(fallback_mode="standalone", fallback_chain=chain), _TIERS
        )
        assert aux == {}, f"chain={chain!r} should be refused"
        assert errors, f"chain={chain!r} should report an error"


def test_compaction_refuses_a_non_string_base_url_and_a_bad_window():
    aux, errors = resolve_compaction_auxiliary(
        _block(base_url=42, required_provider_context_window=True), _TIERS
    )
    assert aux == {}
    assert any("base_url" in e for e in errors)
    assert any("required_provider_context_window" in e for e in errors)


def test_compaction_tier_reference_to_a_non_mapping_tier_is_refused():
    aux, errors = resolve_compaction_auxiliary(
        _block(fallback_mode="tier:T2"), {"T2": "not-a-mapping"}
    )
    assert aux == {}
    assert any("not a mapping" in e for e in errors)


def test_compaction_tier_reference_to_an_empty_queue_is_refused():
    aux, errors = resolve_compaction_auxiliary(
        _block(fallback_mode="tier:T3"), {"T3": {"model": "", "provider": ""}}
    )
    assert aux == {}
    assert any("queue is empty" in e for e in errors)


def test_compaction_registry_check_degrades_when_registry_is_absent(monkeypatch):
    import router.service as service_mod

    monkeypatch.setattr(service_mod, "_caps", None)
    aux, errors = resolve_compaction_auxiliary(
        _block(model="no-such-model-anywhere"), _TIERS
    )
    # With no registry to ask, nothing is provably unknown: the refusal degrades
    # rather than blocking the RESTART-class write on a broken sibling import.
    assert errors == []
    assert aux["model"] == "no-such-model-anywhere"


def test_compaction_registry_check_degrades_when_registry_raises(monkeypatch):
    import router.service as service_mod

    class _RaisingCaps:
        def capabilities_for(self, model):
            raise ValueError("registry exploded")

    monkeypatch.setattr(service_mod, "_caps", _RaisingCaps())
    aux, errors = resolve_compaction_auxiliary(_block(model="glm-4.5-flash"), _TIERS)
    assert errors == []
    assert aux["model"] == "glm-4.5-flash"


class TestCompactionWindows:
    """The threshold curve's inputs, derived rather than restated.

    The numbers these produce are written into Hermes' own ``config.yaml`` by the
    RESTART-class apply, so every degrade here has to answer rather than raise —
    and none of them may INVENT a window: ``threshold.p_base(0)`` is a math-domain
    error and a fabricated small window compacts far too early.
    """

    def test_a_window_comes_from_the_registry(self):
        from router.capabilities import MODEL_CAPABILITIES
        from router.service import _registry_window

        assert _registry_window("glm-4.5-flash") == (
            MODEL_CAPABILITIES["glm-4.5-flash"]["context_window"]
        )

    def test_an_unaskable_registry_yields_zero_not_a_guess(self, monkeypatch):
        import router.service as service_mod
        from router.service import _registry_window

        class _RaisingCaps:
            def capabilities_for(self, model):
                raise TypeError("registry exploded")

        assert _registry_window("") == 0, "no model is not a window"
        monkeypatch.setattr(service_mod, "_caps", None)
        assert _registry_window("glm-4.5-flash") == 0
        monkeypatch.setattr(service_mod, "_caps", _RaisingCaps())
        assert _registry_window("glm-4.5-flash") == 0

    def test_a_registry_that_answers_with_the_wrong_shape_yields_zero(
        self, monkeypatch,
    ):
        import router.service as service_mod
        from router.service import _registry_window

        class _OddCaps:
            def __init__(self, answer):
                self._answer = answer

            def capabilities_for(self, model):
                return self._answer

        for answer in (None, "not-a-mapping", {}, {"context_window": "200000"},
                       {"context_window": 0}, {"context_window": -1}):
            monkeypatch.setattr(service_mod, "_caps", _OddCaps(answer))
            assert _registry_window("whatever") == 0, answer

    def test_policy_models_reads_primaries_hops_and_the_compaction_model(self):
        from router.service import _policy_models

        assert _policy_models({
            "tiers": {
                "T1": {"model": "a", "fallback": [{"model": "b"}]},
                "T2": {"model": "c"},
            },
            "compaction": {"model": "d"},
        }) == {"a", "b", "c", "d"}

    def test_policy_models_skips_every_half_edited_shape(self):
        """The config is HOT; a threshold map must not raise on a mid-edit file."""
        from router.service import _policy_models

        assert _policy_models({}) == set()
        assert _policy_models({"tiers": "still-a-scalar"}) == set()
        assert _policy_models({"tiers": {"T1": "still-a-scalar"}}) == set()
        # A tier with no model, a scalar `fallback:`, a non-mapping hop, and a hop
        # whose model is not a string or is empty.
        assert _policy_models({"tiers": {
            "T1": {"provider": "zai"},
            "T2": {"model": "", "fallback": "deepseek-v4-flash"},
            "T3": {"model": 7, "fallback": ["bare-string", {"model": 7},
                                            {"model": ""}, {"provider": "zai"}]},
        }}) == set()
        # A compaction block that is not a mapping, names no model, or names a
        # non-string / empty one.
        for compaction in ("scalar", None, {}, {"model": 7}, {"model": ""}):
            assert _policy_models({"compaction": compaction}) == set(), compaction

    def test_the_summarizer_falls_back_to_the_widest_routable_window(
        self, tmp_path,
    ):
        """No describable compaction model still has to answer.

        The block itself is REFUSED by resolve_compaction in that case, so this only
        keeps the READ path serving — the widest routable window is the least-wrong
        stand-in, and it is a real published number rather than an invented one.
        """
        from router.capabilities import MODEL_CAPABILITIES

        path = tmp_path / "router.yaml"
        path.write_text(yaml.safe_dump({
            "enabled": True,
            "rules": [],
            "default": {"action": "classify"},
            "tiers": {
                "T1": {"model": "glm-4.5-flash", "provider": "zai"},
                "T2": {"model": "gpt-5.6-terra", "provider": "openai-codex"},
            },
            "compaction": {"provider": "zai", "model": "no-such-model-anywhere"},
        }), encoding="utf-8")

        summarizer, windows = RouterService(path).compaction_windows()
        assert summarizer == MODEL_CAPABILITIES["gpt-5.6-terra"]["context_window"]
        assert summarizer == max(windows.values())

    def test_no_registry_at_all_serves_no_windows_and_the_base_reference(
        self, tmp_path, monkeypatch,
    ):
        """128,000 is ``p_base``'s own reference point, not a guessed window."""
        import router.service as service_mod

        path = tmp_path / "router.yaml"
        path.write_text(yaml.safe_dump({
            "enabled": True, "rules": [], "default": {"action": "classify"},
            "tiers": {"T1": {"model": "glm-4.5-flash", "provider": "zai"}},
        }), encoding="utf-8")

        monkeypatch.setattr(service_mod, "_caps", None)
        summarizer, windows = RouterService(path).compaction_windows()
        assert windows == {}
        assert summarizer == 128_000


def test_compaction_without_fallback_mode_writes_no_chain():
    aux, errors = resolve_compaction_auxiliary(
        {"provider": "zai", "model": "glm-4.5-flash"}, _TIERS
    )
    assert errors == []
    assert aux == {"provider": "zai", "model": "glm-4.5-flash"}
    assert "fallback_chain" not in aux


def test_compaction_tier_fallback_skips_non_mapping_hops():
    tiers = {
        "T1": {
            "model": "glm-4.7",
            "provider": "zai",
            "fallback": [
                "loose",
                {"model": "mimo-v2.5", "provider": "xiaomi"},
            ],
        }
    }
    aux, errors = resolve_compaction_auxiliary(_block(), tiers)
    assert errors == []
    assert aux["fallback_chain"] == [
        {"provider": "zai", "model": "glm-4.7"},
        {"provider": "xiaomi", "model": "mimo-v2.5"},
    ]


def test_validate_compaction_is_noop_when_absent():
    assert RouterService._validate_compaction({"default": {}}) == []


def test_validate_compaction_reports_the_same_errors_as_the_resolver():
    errors = RouterService._validate_compaction(
        {"compaction": _block(model="ghost-model"), "tiers": _TIERS}
    )
    assert any("ghost-model" in e for e in errors)


def test_plan_accepts_a_valid_compaction_block_and_apply_round_trips(config_path):
    service = RouterService(config_path)
    # T1 in the fixture is {"model": "tiny", "provider": "cheap"} — a real tier,
    # so tier:T1 resolves; use a real registry model for the primary.
    plan = service.plan(
        {
            "compaction": {
                "provider": "zai",
                "model": "glm-4.5-flash",
                "fallback_mode": "tier:T1",
            }
        }
    )
    assert plan["valid"] is True, plan["errors"]
    result = service.apply(plan["base_hash"], plan["policy"])
    assert result["ok"] is True
    assert result.get("no_op") is False
    reloaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert reloaded["compaction"]["model"] == "glm-4.5-flash"


def test_a_disabled_rule_survives_the_round_trip_the_console_makes(config_path):
    """Disable a rule the way the console does, and read it back the same way.

    The console's per-row switch sets ``r.enabled = false`` on the draft and posts
    it through plan/apply; its ``off`` badge and its shadow suppression then test
    ``r.enabled === false`` on what ``policy()`` returns. That projection was
    dropping ``enabled``, so the write landed in router.yaml correctly and the
    screen read it back as ENABLED — the switch turned itself on again on reload.

    Asserted as agreement between the two sides of one round trip (what the write
    path persisted, and what the read path serves) rather than on either alone.
    """
    service = RouterService(config_path)

    before = service.policy()["rules"]
    assert "enabled" not in before[0], "absent means enabled — nothing to default"

    draft = service.policy()
    rules = [dict(rule) for rule in draft["rules"]]
    rules[0]["enabled"] = False
    plan = service.plan({"rules": rules})
    assert plan["valid"] is True, plan["errors"]
    assert service.apply(plan["base_hash"], plan["policy"])["ok"] is True

    # The file has it...
    persisted = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert persisted["rules"][0]["enabled"] is False
    # ...and so does the surface the console reads.
    assert service.policy()["rules"][0]["enabled"] is False
    # And the engine agrees the row is off, so the flag is not merely cosmetic:
    # match() returns (output, matched_rule_id), and the disabled row is skipped.
    import router.rules as rules_mod

    _output, matched = rules_mod.match(
        {"verb_class": "hard"},
        False,
        persisted["rules"],
        persisted.get("default", {}),
        persisted.get("tiers", {}),
    )
    assert matched != persisted["rules"][0]["id"]


def test_the_rule_projection_invents_neither_status_nor_enabled(config_path):
    """``status`` used to be served as ``"stable"`` for a row that never said so.

    A projection that defaults a field asserts a fact the operator did not write.
    Both fields are operator-declared and both are omitted when absent — which is
    also what lets the console tell "disabled" from "not specified".
    """
    path = config_path.parent / "bare.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["rules"] = [{
        "id": "bare-row",
        "when": {"verb_class": {"eq": "hard"}},
        "then": {"profile": "coder", "model": "T4"},
    }]
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    row = RouterService(path).policy()["rules"][0]
    assert row == {
        "id": "bare-row",
        "when": {"verb_class": {"eq": "hard"}},
        "then": {"profile": "coder", "model": "T4"},
    }


def test_plan_refuses_an_unknown_compaction_model(config_path):
    service = RouterService(config_path)
    plan = service.plan(
        {
            "compaction": {
                "provider": "zai",
                "model": "definitely-not-registered",
                "fallback_mode": "tier:T1",
            }
        }
    )
    assert plan["valid"] is False
    assert any("definitely-not-registered" in e for e in plan["errors"])



class TestThePreviewNamesWhatProductionWouldRefuse:
    """The two producers were only ever compared where they cannot disagree.

    The flagship agreement test runs against the SHIPPED policy, whose `manual_ban`
    ships empty — so `rules.explain` (no blocklist, and it cannot have one: the
    layer is pure) and `adapter.route` (blocklist-vetted) always matched. With a real
    ban on the T3 primary they diverge completely: production returns
    `deepseek-v4-pro / blocklist_substituted` with `chain_plan.blocked` naming the
    primary, while the preview returned `gpt-5.6-terra / size_rule` and NO `blocked`
    key at all.
    """

    HARD_TASK = "Debug an unknown-cause race condition across the pool and the executor"

    @staticmethod
    def _policy_with_banned_t3(tmp_path):
        import yaml as _yaml
        shipped = _yaml.safe_load(
            (Path(__file__).resolve().parents[2] / "router.example.yaml")
            .read_text(encoding="utf-8")
        )
        primary = shipped["tiers"]["T3"]["model"]
        shipped["blocklist"]["manual_ban"] = [
            {"model": primary, "provider": "", "reason": "test-ban"}
        ]
        path = tmp_path / "router.yaml"
        path.write_text(_yaml.safe_dump(shipped), encoding="utf-8")
        return path, shipped, primary

    def test_the_preview_reports_the_hop_the_veto_would_remove(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        path, config, primary = self._policy_with_banned_t3(tmp_path)

        # NON-VACUITY FIRST, on the production side: this ban must really bite.
        from router.adapter import route
        from router.blocklist import Blocklist
        from router.decision_log import DecisionLog

        log = DecisionLog()
        produced = route(self.HARD_TASK, config, blocklist=Blocklist(config),
                         decision_log=log)
        assert produced["cause"] == "blocklist_substituted", produced
        assert primary in {
            hop["model"] for hop in log.tail(1)[0]["chain_plan"]["blocked"]
        }

        # And now the preview, which used to say nothing at all about it.
        preview = RouterService(path).explain(self.HARD_TASK)
        refused = {hop["model"] for hop in preview["chain_plan"].get("blocked", [])}
        assert primary in refused, (
            "the preview shows an attempt order production would refuse, and says "
            "nothing about it"
        )
        assert all(
            hop.get("reject_reason") == "blocked"
            for hop in preview["chain_plan"]["blocked"]
        ), "the adapter's own spelling, so ONE console renderer serves both"

    def test_the_label_says_it_is_a_label_and_not_a_vetted_plan(
        self, tmp_path, monkeypatch,
    ):
        """LABELLED, not vetted — and the response has to admit which.

        Reproducing `_veto_blocked`'s substitute/widen/deny here would put a SECOND
        copy of the policy in the read model, which PRODUCT.md:60 forbids. So the
        chain keeps the PLANNER's order and the refusal is annotated; a reader must
        be able to tell that apart from a plan that was actually reordered.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        path, _config, primary = self._policy_with_banned_t3(tmp_path)
        preview = RouterService(path).explain(self.HARD_TASK)
        assert preview["chain_plan"]["blocked_source"] == "preview_label"
        # The planner's order is untouched: the head is still what the planner chose.
        assert preview["chain_plan"]["chain"][0]["model"] == primary

    def test_a_clean_policy_gains_no_blocked_key(self, tmp_path, monkeypatch):
        """Additive: the shipped policy's preview must be byte-identical to before."""
        import yaml as _yaml

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        shipped = _yaml.safe_load(
            (Path(__file__).resolve().parents[2] / "router.example.yaml")
            .read_text(encoding="utf-8")
        )
        path = tmp_path / "router.yaml"
        path.write_text(_yaml.safe_dump(shipped), encoding="utf-8")
        plan = RouterService(path).explain(self.HARD_TASK)["chain_plan"]
        assert "blocked_source" not in plan
        assert not plan.get("blocked")

    def test_the_preview_consumes_no_breaker_probe(self, tmp_path, monkeypatch):
        """`/explain` is a pollable GET; `is_blocked` would burn the probe slot.

        The console polls this. Using the mutating form here would take an
        expired-OPEN rail out of rotation once per poll — which is the defect
        `would_block` exists for, arriving through a new door.
        """
        import json as _json
        import time as _time

        import yaml as _yaml

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        shipped = _yaml.safe_load(
            (Path(__file__).resolve().parents[2] / "router.example.yaml")
            .read_text(encoding="utf-8")
        )
        shipped["blocklist"]["auto_breaker"]["enabled"] = True
        path = tmp_path / "router.yaml"
        path.write_text(_yaml.safe_dump(shipped), encoding="utf-8")

        elo = shipped["tiers"]["T3"]["model"]
        rail = shipped["tiers"]["T3"]["provider"]
        state_dir = tmp_path / "hermes-smart-router" / "state"
        state_dir.mkdir(parents=True, exist_ok=True)
        past = _time.time() - 200_000
        state = state_dir / "breaker-state.json"
        state.write_text(_json.dumps({
            "version": 1,
            "entries": {f"{elo}@{rail}": {
                "state": "OPEN",
                "failure_events": [{"kind": "ttfb_stall", "ts": past, "weight": 3}] * 2,
                "cooldown_until": past + 60, "backoff_seconds": 60.0,
                "last_failure_kind": "ttfb_stall",
            }},
        }), encoding="utf-8")
        before = state.read_bytes()

        service = RouterService(path)
        for _ in range(5):
            service.explain(self.HARD_TASK)

        assert state.read_bytes() == before, "a polled preview spent the probe slot"
        from router.blocklist import Blocklist
        assert Blocklist(shipped).would_block(elo, rail) is False

    def test_the_label_tolerates_every_shape_it_can_be_handed(
        self, tmp_path, monkeypatch,
    ):
        """A read path must not raise, and the plan comes from a HOT config.

        Tested on the helper directly, as this file already does for
        `_registry_window` and `_policy_models`: these branches exist for shapes a
        deployed-by-copy planner can produce, and driving each one through the whole
        explain path would mean stubbing `plan_chain` three different ways to assert
        one `continue`.
        """
        import yaml as _yaml
        import router.service as service_mod

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        shipped = _yaml.safe_load(
            (Path(__file__).resolve().parents[2] / "router.example.yaml")
            .read_text(encoding="utf-8")
        )
        path = tmp_path / "router.yaml"
        path.write_text(_yaml.safe_dump(shipped), encoding="utf-8")
        service = RouterService(path)
        banned = shipped["tiers"]["T3"]["model"]
        config = dict(shipped)
        config["blocklist"] = {
            **shipped["blocklist"],
            "manual_ban": [{"model": banned, "provider": "", "reason": "t"}],
        }

        # No chain, or an empty one: nothing to annotate, returned unchanged.
        for plan in ({}, {"chain": None}, {"chain": []}, {"chain": "nope"}):
            assert service._label_refused_hops(plan, {}, config) is plan

        # An unattributable hop is skipped rather than crashing on `str(None)`.
        plan = {"chain": ["bare-string", {}, {"provider": "zai"},
                          {"model": banned, "provider": "openai-codex"}]}
        annotated = service._label_refused_hops(plan, {}, config)
        assert [hop["model"] for hop in annotated["blocked"]] == [banned]

        # A hop the PLANNER already rejected is not listed twice.
        plan = {
            "chain": [{"model": banned, "provider": "openai-codex"}],
            "blocked": [{"model": banned, "provider": "openai-codex",
                         "reject_reason": "blocked"}],
        }
        annotated = service._label_refused_hops(plan, {}, config)
        assert annotated is plan, "nothing new to add means the plan is untouched"

        # An unreadable blocklist degrades to an unannotated plan.
        class _Exploding:
            def __init__(self, _config):
                raise RuntimeError("state dir is a file")

        monkeypatch.setattr(service_mod, "Blocklist", _Exploding)
        plan = {"chain": [{"model": banned, "provider": "openai-codex"}]}
        assert service._label_refused_hops(plan, {}, config) is plan


# ── the models THIS INSTALL is configured with ─────────────────────────────────
# The console's pickers offered `Object.keys(state.capabilities)` — the capability
# REGISTRY, 43 ids across zai/deepseek/openai/anthropic/xiaomi/minimax. That is the
# catalogue of what is KNOWN, and it answers a question nobody asked: an operator
# choosing a model wants the ones their Hermes can actually call.
#
# Measured on the docker stack, 2026-09-02: its config is
# `model.default: us.anthropic.claude-opus-5` on `provider: bedrock` and the only
# provider credential in the containers is AWS — while the pickers happily offered
# glm, deepseek and gpt ids it has no key for, and did NOT offer the bedrock id it
# actually runs on (`us.anthropic.*` is deliberately unregistered, see
# capabilities.py's module docstring).
#
# So "configured" is read from the AGENT's own config.yaml, which is the only thing
# that knows what this install can reach. Through paths.hermes_root(), so a
# profile-scoped HERMES_HOME still finds the root config rather than a profile's.


def _write_agent_config(root, payload):
    """Put an agent config.yaml at the Hermes root the service will resolve."""
    import yaml

    (root / "config.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )


def test_status_reports_the_models_this_install_is_configured_with(
    config_path, tmp_path, monkeypatch
):
    """Every config key that names a callable model, in one list, each with its source.

    The source is carried because it is the operator's trace: "why is this offered?"
    is answered by the key that put it there, and a picker can say so.
    """
    root = tmp_path / "hermes-root"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    _write_agent_config(
        root,
        {
            "model": {"default": "us.anthropic.claude-opus-5", "provider": "bedrock"},
            "fallback_providers": [
                {"label": "zai", "provider": "zai", "model": "glm-5.3-flash"},
                {"provider": "deepseek", "model": "deepseek-v4-pro"},
            ],
            "auxiliary": {
                "vision": {
                    "provider": "zai",
                    "model": "glm-4.5v",
                    # TWO hops on purpose: a one-entry chain never exercises the
                    # loop's back-edge, and this list is offered in order.
                    "fallback_chain": [
                        {"provider": "deepseek", "model": "deepseek-v4-flash-vision-exp"},
                        {"provider": "zai", "model": "glm-4.6v"},
                    ],
                },
                "compression": {"provider": "bedrock", "model": "us.anthropic.claude-haiku-4-5"},
            },
        },
    )

    served = RouterService(config_path).status()["configured_models"]

    assert [(e["model"], e["provider"]) for e in served] == [
        ("us.anthropic.claude-opus-5", "bedrock"),
        ("glm-5.3-flash", "zai"),
        ("deepseek-v4-pro", "deepseek"),
        ("glm-4.5v", "zai"),
        ("deepseek-v4-flash-vision-exp", "deepseek"),
        ("glm-4.6v", "zai"),
        ("us.anthropic.claude-haiku-4-5", "bedrock"),
    ], "the default leads, then the file's own order"
    assert served[0]["source"] == "model.default"
    assert served[1]["source"] == "fallback_providers"
    assert served[3]["source"] == "auxiliary.vision"
    assert served[6]["source"] == "auxiliary.compression"


def test_configured_models_dedupe_on_model_and_provider_keeping_the_first_source(
    config_path, tmp_path, monkeypatch
):
    """The same rail named twice is one option, credited to where it was named first.

    Real shape: a fallback_providers list that repeats the default, or repeats one
    entry with a different base_url (the live stack has exactly that — two zai
    glm-5.3-flash entries, one free-tier). Two identical options in a select is a
    picker that looks broken.
    """
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    _write_agent_config(
        root,
        {
            "model": {"default": "glm-5.3-flash", "provider": "zai"},
            "fallback_providers": [
                {"provider": "zai", "model": "glm-5.3-flash"},
                {"provider": "zai", "model": "glm-5.3-flash", "base_url": "https://other"},
                # Same model id on a DIFFERENT rail is a different option, not a dupe.
                {"provider": "openrouter", "model": "glm-5.3-flash"},
            ],
        },
    )

    served = RouterService(config_path).status()["configured_models"]
    assert [(e["model"], e["provider"], e["source"]) for e in served] == [
        ("glm-5.3-flash", "zai", "model.default"),
        ("glm-5.3-flash", "openrouter", "fallback_providers"),
    ]


def test_status_finds_the_root_config_from_a_profile_scoped_home(
    config_path, tmp_path, monkeypatch
):
    """The plugin runs under ~/.hermes/profiles/<name>; the config lives at the root.

    Same peel paths.hermes_root() exists for, and the reason it exists: resolving
    install-level facts under a profile splits every reader from every writer.
    """
    root = tmp_path / "root"
    (root / "profiles" / "coder").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "coder"))
    _write_agent_config(root, {"model": {"default": "m", "provider": "p"}})

    served = RouterService(config_path).status()["configured_models"]
    assert [(e["model"], e["provider"]) for e in served] == [("m", "p")]


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({}, id="empty-config"),
        pytest.param({"model": "not-a-mapping"}, id="model-is-a-scalar"),
        pytest.param({"model": {"default": "", "provider": "p"}}, id="empty-model-id"),
        pytest.param({"model": {"default": "m"}}, id="model-without-provider"),
        pytest.param({"fallback_providers": "not-a-list"}, id="fallbacks-not-a-list"),
        pytest.param({"fallback_providers": ["a string", 7, None]}, id="fallback-rows-not-mappings"),
        pytest.param({"fallback_providers": [{"provider": "p"}]}, id="fallback-without-model"),
        pytest.param({"fallback_providers": [{"model": 7, "provider": "p"}]}, id="model-not-a-string"),
        pytest.param({"auxiliary": "not-a-mapping"}, id="auxiliary-is-a-scalar"),
        pytest.param({"auxiliary": {"vision": "not-a-mapping"}}, id="vision-is-a-scalar"),
        pytest.param({"auxiliary": {"vision": {"provider": "p", "model": "m",
                                               "fallback_chain": "not-a-list"}}},
                     id="vision-chain-not-a-list"),
        pytest.param({"auxiliary": {"compression": {"provider": "auto"}}},
                     id="compression-names-no-model"),
        # A chain whose FIRST hop is unusable and whose second is fine: the bad row is
        # skipped and the good one still reaches the operator. A chain that lost one
        # entry must not lose the rest of itself.
        pytest.param({"auxiliary": {"vision": {"provider": "p", "model": "m",
                                               "fallback_chain": [{"provider": "q"},
                                                                  {"provider": "q", "model": "n"}]}}},
                     id="vision-chain-row-without-a-model"),
    ],
)
def test_configured_models_degrades_on_every_shape_an_operator_can_type(
    config_path, tmp_path, monkeypatch, payload
):
    """A config.yaml is hand-edited, so every field is untrusted shape.

    None of these may raise into /status, and none may invent an option: a picker
    offering a model the install cannot call is the defect this list exists to fix,
    so a row it cannot read is a row it drops.
    """
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    _write_agent_config(root, payload)

    served = RouterService(config_path).status()["configured_models"]
    assert isinstance(served, list)
    assert all(
        isinstance(e, dict) and e.get("model") and e.get("provider") for e in served
    ), f"a row that cannot be read must be dropped, got {served}"


def test_configured_models_is_empty_when_there_is_no_agent_config_at_all(
    config_path, tmp_path, monkeypatch
):
    """An absent or unparseable config.yaml is an absence, not an error.

    /status is what the console reads to describe a broken install, so it must
    answer. The console's own job is to say "no list to choose from" — which it
    already does, and which is honest.
    """
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    assert RouterService(config_path).status()["configured_models"] == []

    (root / "config.yaml").write_text("model: {unclosed\n", encoding="utf-8")
    assert RouterService(config_path).status()["configured_models"] == []


# ── the agent's own queues, in the SAME shape as a tier ────────────────────────
# "Até você se perde nessa configuração de models e fallback - de tão complexa que
# ela é." Counted: FIVE spellings of one idea across the two files.
#
#   fallback_providers            [{label, provider, model, base_url?}]
#   auxiliary.vision              {provider, model, fallback_chain: [{provider, model}]}
#   tiers.Tn                      {model, provider, fallback: [{...}], fallback_strategy, …}
#   blocklist.fallback_chain      [bare strings]
#   compaction                    fallback_mode: standalone|tier:Tn + fallback_chain
#
# Four field orders, one bare-string list, one that can be a REFERENCE to another
# queue. There is nothing to learn because there is no single thing to learn.
#
# Every one of them is the same idea: a PRIMARY and its RESERVES, tried in order. So
# the server normalises the agent's side to exactly that, and the console draws all of
# them with the one renderer it already shares (chainList).


def test_agent_queues_normalise_the_config_to_primary_and_reserves(
    config_path, tmp_path, monkeypatch
):
    """`model.default` + `fallback_providers` IS a queue; so is vision with its chain.

    Naming them that way is what makes `fallback_providers` legible: it is not a list of
    providers, it is the RESERVES of the main model.
    """
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    _write_agent_config(
        root,
        {
            "model": {"default": "us.anthropic.claude-opus-5", "provider": "bedrock"},
            "fallback_providers": [
                {"label": "sonnet", "provider": "bedrock", "model": "us.anthropic.claude-sonnet-5"},
                {"provider": "bedrock", "model": "us.anthropic.claude-haiku-4-5"},
            ],
            "auxiliary": {
                "vision": {
                    "provider": "bedrock",
                    "model": "us.anthropic.claude-opus-5",
                    "fallback_chain": [{"provider": "zai", "model": "glm-4.5v"}],
                },
                "compression": {"provider": "bedrock", "model": "us.anthropic.claude-haiku-4-5"},
            },
        },
    )

    queues = RouterService(config_path).status()["agent_queues"]
    by_key = {q["key"]: q for q in queues}

    assert list(by_key) == ["model", "auxiliary.vision", "auxiliary.compression"], (
        "the main chain first, because it is what runs when nothing else is asked"
    )
    assert [(a["model"], a["provider"]) for a in by_key["model"]["attempts"]] == [
        ("us.anthropic.claude-opus-5", "bedrock"),
        ("us.anthropic.claude-sonnet-5", "bedrock"),
        ("us.anthropic.claude-haiku-4-5", "bedrock"),
    ], "the default is attempt 1 and fallback_providers are its reserves, in order"
    assert [(a["model"], a["provider"]) for a in by_key["auxiliary.vision"]["attempts"]] == [
        ("us.anthropic.claude-opus-5", "bedrock"),
        ("glm-4.5v", "zai"),
    ]
    # `where` is the operator's pointer: which file and key owns this queue. The router
    # does not write config.yaml, so saying where it lives is the whole of the answer to
    # "why can I not edit this here?".
    assert by_key["model"]["where"] == "config.yaml: model.default + fallback_providers"
    assert all(q["editable"] is False for q in queues), (
        "config.yaml is RESTART-class and the router only READS it; claiming otherwise "
        "would put two authorities on one fact"
    )


def test_configured_models_is_the_agent_queues_flattened(config_path, tmp_path, monkeypatch):
    """ONE walk over config.yaml, two facts derived from it.

    These were two separate traversals and that is the defect class this repo keeps
    paying for — two walks over one document drift, exactly as the console's own
    bannableModels drifted from its twin. The flattening is asserted rather than
    described so the two cannot disagree again.
    """
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    _write_agent_config(
        root,
        {
            "model": {"default": "m1", "provider": "p1"},
            "fallback_providers": [{"provider": "p2", "model": "m2"}],
            "auxiliary": {"vision": {"provider": "p1", "model": "m1"}},
        },
    )

    status = RouterService(config_path).status()
    flat = [(e["model"], e["provider"]) for e in status["configured_models"]]
    from_queues = []
    for queue in status["agent_queues"]:
        for attempt in queue["attempts"]:
            pair = (attempt["model"], attempt["provider"])
            if pair not in from_queues:
                from_queues.append(pair)
    assert flat == from_queues, "the model list IS the queues, deduplicated"
    # m1/p1 appears in both the main queue and vision; it is one option.
    assert flat == [("m1", "p1"), ("m2", "p2")]


def test_a_queue_the_config_does_not_name_is_absent_rather_than_empty(
    config_path, tmp_path, monkeypatch
):
    """`auxiliary.compression: {provider: auto}` names no model — that is not a queue.

    An empty queue drawn as a queue is the "no empty frame" rule broken: it would put a
    heading and a border around nothing. The live docker stack is in exactly this state.
    """
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    _write_agent_config(
        root,
        {
            "model": {"default": "m", "provider": "p"},
            "auxiliary": {"compression": {"provider": "auto"}, "vision": {"provider": "auto"}},
        },
    )
    queues = RouterService(config_path).status()["agent_queues"]
    assert [q["key"] for q in queues] == ["model"], (
        "auto names no model, so there is no queue to draw"
    )


# ── the classifier: a SIXTH queue, and the console could not see it ────────────
# "não posso mudar o classifier?"
#
# Half of it. The console has a door (the "decide na hora" cell) and an editor, and that
# editor offers Modelo + Provedor — three fields. The classifier block in router.yaml also
# carries a `chain` (its own fallback queue, the SIXTH spelling of one idea), plus
# temperature, max_tokens, timeout_seconds and on_total_failure. None of it reaches the
# screen.
#
# The root cause is here, not in the console: `policy()` projects `fail_safe` verbatim and
# does not project `classifier` at all, and `status()` sends only {model, provider}. A
# console cannot offer what the server never sends.
#
# Measured on the docker stack 2026-09-03: its classifier chain was
# glm-5.3-flash/zai -> deepseek-v4-flash/deepseek — two rails that install has no
# credential for — and there was no surface on which an operator could have seen that.


def test_policy_projects_the_classifier_like_it_projects_fail_safe(config_path):
    """The last-resort block is served whole; the classifier was not served at all.

    Both are single blocks naming a model, a rail and their own fallback behaviour, and
    both are things an operator changes. Serving one and withholding the other is what made
    the classifier half-editable: the editor could only offer the two fields `status`
    happened to carry.

    Verbatim, like `fail_safe` — the block's knobs (`chain`, `temperature`, `max_tokens`,
    `timeout_seconds`, `on_total_failure`) are exactly what a console needs and none of them
    is a secret.
    """
    served = RouterService(config_path).policy()
    assert "classifier" in served, "the classifier block reaches the console"
    assert served["classifier"] == {"model": "judge", "provider": "judge-rail"}, (
        "verbatim, the same treatment fail_safe gets"
    )
    assert "fail_safe" in served, "and the sibling it is modelled on is still there"


def test_the_classifier_chain_is_served_whole_so_it_can_be_read_and_edited(tmp_path):
    """Its `chain` is a fallback queue and must arrive as one.

    This is the SIXTH place these two files spell "if this fails, try that", after
    fallback_providers, auxiliary.vision, tiers.Tn.fallback, blocklist.fallback_chain and
    compaction. It is served verbatim so the console can draw it with the one queue
    renderer instead of inventing a seventh shape.
    """
    import yaml as _yaml

    path = tmp_path / "router.yaml"
    path.write_text(
        _yaml.safe_dump(
            {
                "enabled": True,
                "classifier": {
                    "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                    "provider": "bedrock",
                    "chain": [
                        {"model": "us.anthropic.claude-sonnet-5", "provider": "bedrock"},
                    ],
                    "on_total_failure": "heuristic",
                    "temperature": 0,
                    "max_tokens": 128,
                    "timeout_seconds": 15,
                },
                "rules": [],
                "default": {"action": "classify"},
                "tiers": {"T1": {"model": "m", "provider": "p"}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    served = RouterService(path).policy()["classifier"]
    assert [h["model"] for h in served["chain"]] == ["us.anthropic.claude-sonnet-5"], (
        "the reserves arrive, in order"
    )
    # And every knob the file carries, because a console that shows three of seven fields
    # is a console that silently drops four operator decisions.
    for knob in ("on_total_failure", "temperature", "max_tokens", "timeout_seconds"):
        assert knob in served, f"{knob} is in the file and must reach the screen"


def test_a_classifier_the_file_omits_is_an_empty_block_not_a_missing_key(tmp_path):
    """Same shape as fail_safe's: the key is always present, empty when unconfigured.

    A console branching on presence would otherwise have to tell "no classifier" from "an
    older sidecar that does not serve it", which are different states with different
    remedies.
    """
    import yaml as _yaml

    path = tmp_path / "router.yaml"
    path.write_text(
        _yaml.safe_dump({"enabled": True, "rules": [], "default": {}, "tiers": {}}),
        encoding="utf-8",
    )
    served = RouterService(path).policy()
    assert served["classifier"] == {}


# --- source field + obsolete-model filter (I7) -------------------------------

def _policy_model(config_path):
    import yaml
    from router.service import _policy_models
    return sorted(_policy_models(yaml.safe_load(config_path.read_text())))[0]


def test_routes_report_source_and_in_policy_without_dropping_history(
        tmp_path, monkeypatch, config_path):
    live = _policy_model(config_path)
    _seed_traces(tmp_path, monkeypatch, [
        {"ts": 1.0, "cause": "classifier", "task": "old", "output": {"model": "gone-model"}},
        {"ts": 2.0, "cause": "classifier", "task": "k", "source": "kanban",
         "output": {"model": live}},
        {"ts": 3.0, "cause": "classifier", "task": "c", "source": "chat",
         "output": {"model": live}},
        {"ts": 4.0, "cause": "classifier", "task": "x", "source": "bogus",
         "output": {"model": live}},
    ])
    svc = RouterService(config_path)
    everything = svc.routes()
    assert everything["count"] == 4 and everything["matched"] == 4
    by_task = {r["task"]: r for r in everything["routes"]}
    assert by_task["old"]["source"] == "unknown" and by_task["old"]["in_policy"] is False
    assert by_task["x"]["source"] == "unknown"
    assert by_task["k"]["source"] == "kanban" and by_task["k"]["in_policy"] is True

    only_kanban = svc.routes(source="kanban")
    assert [r["task"] for r in only_kanban["routes"]] == ["k"]
    assert only_kanban["count"] == 4 and only_kanban["matched"] == 1

    current = svc.routes(policy_only=True)
    assert sorted(r["task"] for r in current["routes"]) == ["c", "k", "x"]
    assert current["count"] == 4  # the obsolete entry is still on record


def test_decision_log_stamps_source_only_when_given():
    from router.decision_log import DecisionLog, source_of
    log = DecisionLog(source="chat")
    log.record("classifier", {"model": "m"})
    assert log.entries()[0]["source"] == "chat"
    bare = DecisionLog()
    bare.record("classifier", {"model": "m"})
    assert "source" not in bare.entries()[0] and source_of(bare.entries()[0]) == "unknown"
    assert DecisionLog(source="nope")._source is None
