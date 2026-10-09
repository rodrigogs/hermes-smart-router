"""Closed-vocabulary lint for the top-level ``alignment:`` block (spec §5, F3)."""

import copy
from pathlib import Path

import pytest
import yaml

from router.rules import lint
from router.service import RouterService

GOOD = yaml.safe_load(
    """
enabled: false
mode: shadow
scopes: [kanban]
trigger:
  pct: [60, 85]
  min_iterations: 12
  goal_turn_pct: 70
  fallback_max_iterations: 90
  cooldown_iterations: 10
  max_evaluations_per_session: 3
overrides:
  tiers:
    T4: {pct: [50, 80]}
    T1: {enabled: false}
  profiles:
    researcher: {pct: [70, 90]}
    coder: {pct: [60, 85], min_confidence: {adjust: 0.6, block: 0.85}}
judge:
  chain:
    - {model: claude-opus-5.5, provider: copilot, billing_mode: premium}
    - {model: gpt-6.1-sol, provider: openai-codex, billing_mode: subscription}
  require_distinct_provider: true
  timeout_seconds: 90
  max_input_tokens: 120000
  max_output_tokens: 800
  temperature: 0
  confirm_block_with_second_judge: true
thresholds:
  adjust_min_confidence: 0.6
  block_min_confidence: 0.85
  require_evidence: true
  adjust_limit: 2
  allow_direct_block: false
actions:
  adjust: {deliver: auto, comment: true}
  block: {kanban_block: true, comment: true}
alert:
  on: [block]
  channels: [telegram]
  include_reasons: true
  include_transcript_excerpt: false
  quiet_hours: null
privacy:
  redact: true
  max_transcript_chars_to_judge: 400000
  allow_providers: [copilot, openai-codex]
budget:
  max_judge_calls_per_day: 40
  breaker: {threshold: 3, cooldown_seconds: 1800}
log:
  path: alignment.jsonl
"""
)

# One rejection per key of §5: (dotted path, a value the closed vocabulary refuses).
BAD_VALUES = {
    "enabled": "yes",
    "mode": "loud",
    "scopes": ["kanban", "everything"],
    "trigger.pct": [85, 60],
    "trigger.min_iterations": 0,
    "trigger.goal_turn_pct": 101,
    "trigger.fallback_max_iterations": "ninety",
    "trigger.cooldown_iterations": -1,
    "trigger.max_evaluations_per_session": True,
    "overrides.tiers.T4.pct": [0, 50],
    "overrides.tiers.T1.enabled": "no",
    "overrides.tiers.T4.min_iterations": 0,
    "overrides.tiers.T4.goal_turn_pct": 0,
    "overrides.tiers.T4.fallback_max_iterations": 0,
    "overrides.tiers.T4.cooldown_iterations": -2,
    "overrides.tiers.T4.max_evaluations_per_session": 0,
    "overrides.tiers.T4.min_confidence.adjust": 1.5,
    "overrides.tiers.T4.min_confidence.block": -0.1,
    "overrides.profiles.coder.pct": "60",
    "overrides.profiles.coder.min_confidence.block": "high",
    "judge.chain": [],
    "judge.chain.0.model": "",
    "judge.chain.0.provider": 7,
    "judge.chain.0.billing_mode": "free-ish",
    "judge.require_distinct_provider": "true",
    "judge.timeout_seconds": 0,
    "judge.max_input_tokens": 0,
    "judge.max_output_tokens": -5,
    "judge.temperature": -1,
    "judge.confirm_block_with_second_judge": 1,
    "thresholds.adjust_min_confidence": 2,
    "thresholds.block_min_confidence": "0.9",
    "thresholds.require_evidence": "yes",
    "thresholds.adjust_limit": -1,
    "thresholds.allow_direct_block": "no",
    "actions.adjust.deliver": "telepathy",
    "actions.adjust.comment": "yes",
    "actions.block.kanban_block": "yes",
    "actions.block.comment": 0,
    "alert.on": ["block", "panic"],
    "alert.True": ["block", "panic"],
    "alert.channels": [],
    "alert.include_reasons": "yes",
    "alert.include_transcript_excerpt": "yes",
    "alert.quiet_hours": 22,
    "privacy.redact": "yes",
    "privacy.max_transcript_chars_to_judge": 0,
    "privacy.allow_providers": ["copilot", ""],
    "budget.max_judge_calls_per_day": -1,
    "budget.breaker.threshold": 0,
    "budget.breaker.cooldown_seconds": 0,
    "log.path": "",
}


def _config(alignment):
    return {
        "enabled": True,
        "default": {"action": "classify"},
        "tiers": {t: {"model": f"m{t}", "provider": "p"} for t in ("T1", "T2", "T3", "T4")},
        "rules": [],
        "alignment": alignment,
    }


def _alignment_errors(alignment):
    return [e for e in lint(_config(alignment)) if e.startswith("alignment")]


def _set(tree, dotted, value):
    node = tree
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = node[int(p)] if isinstance(node, list) else node.setdefault(p, {})
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def _leaf_paths(node, prefix=""):
    if isinstance(node, dict) and node:
        for k, v in node.items():
            yield from _leaf_paths(v, f"{prefix}{k}.")
    elif isinstance(node, list) and node and isinstance(node[0], dict):
        for i, v in enumerate(node):
            yield from _leaf_paths(v, f"{prefix}{i}.")
    else:
        yield prefix[:-1]


def test_the_spec_example_is_valid():
    assert _alignment_errors(GOOD) == []


def test_absent_block_is_valid():
    config = _config(GOOD)
    del config["alignment"]
    assert lint(config) == []


def test_every_key_of_the_spec_has_a_rejection_case():
    """Each §5 leaf key (and each enum/shape) has a bad value below: none can be dropped."""
    covered = set(BAD_VALUES)
    for path in _leaf_paths(GOOD):
        # Overrides are validated through ONE table, exercised per tier/profile name
        # that GOOD declares plus the full override vocabulary on T4.
        if path.startswith("overrides.") or path.endswith(".billing_mode"):
            continue
        if path.startswith("judge.chain."):
            continue
        assert path in covered, f"no rejection test for alignment.{path}"


@pytest.mark.parametrize("path,bad", sorted(BAD_VALUES.items()))
def test_each_key_rejects_a_value_outside_its_vocabulary(path, bad):
    alignment = copy.deepcopy(GOOD)
    _set(alignment, path, bad)
    errors = _alignment_errors(alignment)
    assert errors, f"alignment.{path}={bad!r} was accepted"


@pytest.mark.parametrize(
    "path",
    [
        "bogus",
        "trigger.bogus",
        "overrides.bogus",
        "overrides.tiers.T9",
        "overrides.tiers.T4.bogus",
        "overrides.profiles.coder.bogus",
        "overrides.profiles.coder.min_confidence.bogus",
        "judge.bogus",
        "thresholds.bogus",
        "actions.bogus",
        "actions.adjust.bogus",
        "actions.block.bogus",
        "alert.bogus",
        "privacy.bogus",
        "budget.bogus",
        "budget.breaker.bogus",
        "log.bogus",
        "judge.chain.0.bogus",
    ],
)
def test_an_unknown_key_anywhere_fails_the_lint(path):
    alignment = copy.deepcopy(GOOD)
    _set(alignment, path, 1)
    assert any("bogus" in e or "T9" in e for e in _alignment_errors(alignment))


@pytest.mark.parametrize(
    "alignment",
    ["nope", None, ["a"]],
)
def test_a_non_mapping_block_is_rejected(alignment):
    assert _alignment_errors(alignment) == ["alignment must be a mapping"]


@pytest.mark.parametrize(
    "path,bad",
    [
        ("trigger", "x"),
        ("overrides", []),
        ("overrides.tiers", []),
        ("overrides.profiles", "x"),
        ("overrides.profiles.coder", 3),
        ("judge", "x"),
        ("judge.chain.0", "x"),
        ("judge.chain", "x"),
        ("budget.breaker", 1),
    ],
)
def test_a_wrong_shaped_section_is_rejected(path, bad):
    alignment = copy.deepcopy(GOOD)
    _set(alignment, path, bad)
    assert _alignment_errors(alignment)


def test_a_chain_hop_must_name_model_and_provider():
    alignment = copy.deepcopy(GOOD)
    alignment["judge"]["chain"] = [{"model": "only-model"}]
    assert any("must declare" in e for e in _alignment_errors(alignment))


_EXAMPLE = Path(__file__).resolve().parents[2] / "router.example.yaml"


def test_enabled_false_is_the_documented_default():
    text = _EXAMPLE.read_text(encoding="utf-8")
    # The block ships COMMENTED OUT: absent from the parsed file, so the feature is off.
    assert "alignment" not in yaml.safe_load(text)
    assert "# alignment:" in text and "#   enabled: false" in text


def test_the_example_block_in_the_comment_lints_clean():
    """Uncommenting the shipped block must yield a valid config."""
    lines = _EXAMPLE.read_text(encoding="utf-8").splitlines()
    start = lines.index("# alignment:")
    block = ["alignment:"]
    for ln in lines[start + 1:]:
        if not ln.startswith("#   "):
            break
        block.append(ln[2:])
    parsed = yaml.safe_load("\n".join(block))
    assert _alignment_errors(parsed["alignment"]) == []
    assert parsed["alignment"]["enabled"] is False


def test_console_plan_and_apply_accept_the_alignment_block(tmp_path):
    path = tmp_path / "router.yaml"
    path.write_text(yaml.safe_dump(_config(GOOD) | {"alignment": {"enabled": False}}), encoding="utf-8")
    service = RouterService(path)

    plan = service.plan({"alignment": {"mode": "alert"}})
    assert plan["valid"] is True
    assert plan["policy"]["alignment"]["mode"] == "alert"

    bad = service.plan({"alignment": {"mode": "loud"}})
    assert bad["valid"] is False
    assert any(e.startswith("alignment.mode") for e in bad["errors"])

    result = service.apply(plan["base_hash"], {"alignment": {"mode": "alert"}})
    assert result.get("applied", result.get("ok", True)) is not False
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["alignment"] == {
        "enabled": False,
        "mode": "alert",
    }
