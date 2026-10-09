"""Unit tests for the model capability registry (router/capabilities.py)."""

import ast
import inspect
import random
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import router.capabilities as caps_module
from router.capabilities import (
    BILLING_MODES,
    CAPABILITY_ASSERTION_KEYS,
    FALLBACK_STRATEGIES,
    MAX_REGISTERED_CONTEXT,
    MODEL_CAPABILITIES,
    REQUIREMENT_KEYS,
    apply_time_cap,
    apply_time_policy,
    capabilities_for,
    derive_requirements,
    effective_price,
    filter_chain,
    in_expensive_window,
    independent_rails,
    next_window_change,
    order_chain,
    price_multiplier,
    price_window_diagnostics,
    registry_diagnostics,
    satisfies,
    upstream_group,
)

# The clock is INJECTED, never read: every time-dependent assertion below names
# the instant it is asserting about. 2026-08-17 is a Monday, so the weekday of
# each date is unambiguous and asserted once in
# test_the_reference_clocks_are_the_weekdays_they_claim.
UTC = timezone.utc


def _at(day: int, hour: int, minute: int = 0) -> datetime:
    """A UTC datetime in the reference week: day 17 = Monday .. 23 = Sunday."""
    return datetime(2026, 8, day, hour, minute, tzinfo=UTC)


MON = 17
WED = 19
FRI = 21
SAT = 22
SUN = 23


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------

def test_registry_has_no_diagnostics():
    assert registry_diagnostics() == []


def test_every_billing_mode_is_in_the_closed_set():
    modes = {entry["billing_mode"] for entry in MODEL_CAPABILITIES.values()}
    assert modes <= BILLING_MODES


def test_requirement_keys_is_the_documented_closed_set():
    assert REQUIREMENT_KEYS == frozenset(
        {"min_context", "vision", "tool_calling", "structured_output"}
    )


# ---------------------------------------------------------------------------
# capabilities_for
# ---------------------------------------------------------------------------

def test_capabilities_for_known_model_returns_registry_entry():
    caps = capabilities_for("glm-5.3-flash")
    assert caps["provider"] == "zai"
    assert caps["context_window"] == 1_000_000
    assert caps["billing_mode"] == "plan"


def test_capabilities_for_unknown_model_without_declared_is_none():
    assert capabilities_for("no-such-model-v9") is None


def test_capabilities_for_unknown_model_with_declared_returns_declared():
    caps = capabilities_for(
        "no-such-model-v9", {"context_window": 32_000, "vision": True}
    )
    assert caps == {"context_window": 32_000, "vision": True}


def test_capabilities_for_never_mutates_the_registry():
    caps = capabilities_for("glm-4.7", {"context_window": 1})
    assert caps["context_window"] == 1
    assert MODEL_CAPABILITIES["glm-4.7"]["context_window"] == 200_000


def test_declared_overrides_beat_the_registry_entry():
    caps = capabilities_for("glm-4.7", {"vision": True, "context_window": 999})
    assert caps["vision"] is True
    assert caps["context_window"] == 999
    # untouched fields still come from the registry
    assert caps["provider"] == "zai"


def test_capabilities_for_ignores_non_capability_keys_in_declared():
    caps = capabilities_for("glm-4.7", {"model": "glm-4.7", "weight": 3})
    assert "model" not in caps
    assert "weight" not in caps


# ---------------------------------------------------------------------------
# satisfies — the four requirement kinds
# ---------------------------------------------------------------------------

def test_satisfies_min_context_passes_for_a_big_window():
    assert satisfies("glm-5.3", {"min_context": 500_000}) == (True, "")


def test_satisfies_min_context_rejects_context_too_small():
    assert satisfies("glm-4.5v", {"min_context": 500_000}) == (
        False,
        "context_too_small",
    )


def test_satisfies_vision_passes_for_a_vision_model():
    assert satisfies("glm-4.6v", {"vision": True}) == (True, "")


def test_satisfies_vision_rejects_no_vision():
    assert satisfies("glm-4.7", {"vision": True}) == (False, "no_vision")


def test_satisfies_tool_calling_passes_for_a_tool_model():
    assert satisfies("glm-4.7", {"tool_calling": True}) == (True, "")


def test_satisfies_tool_calling_rejects_no_tool_calling():
    assert satisfies("z-ai/glm-5.2:free", {"tool_calling": True}) == (
        False,
        "no_tool_calling",
    )


def test_satisfies_structured_output_passes_for_a_structured_model():
    assert satisfies("glm-4.7", {"structured_output": True}) == (True, "")


def test_satisfies_structured_output_rejects_no_structured_output():
    assert satisfies("MiniMax-M3", {"structured_output": True}) == (
        False,
        "no_structured_output",
    )


def test_satisfies_with_no_requirements_passes():
    assert satisfies("glm-4.7", {}) == (True, "")


def test_satisfies_false_requirement_does_not_constrain():
    # asking for vision=False must not reject a text-only model
    assert satisfies("glm-4.7", {"vision": False}) == (True, "")


def test_satisfies_ignores_keys_outside_the_requirement_set():
    assert satisfies("glm-4.7", {"needs_telepathy": True}) == (True, "")


def test_satisfies_reports_the_contradiction_over_the_unknown():
    # context is known-too-small; vision is unknown for this fake elo
    ok, reason = satisfies(
        "mystery-elo",
        {"min_context": 100_000, "vision": True},
        {"context_window": 8_000},
    )
    assert (ok, reason) == (False, "context_too_small")


# ---------------------------------------------------------------------------
# satisfies — context boundary
# ---------------------------------------------------------------------------

def test_min_context_exactly_equal_to_context_window_passes():
    assert satisfies("glm-4.7", {"min_context": 200_000}) == (True, "")


def test_min_context_one_token_over_context_window_fails():
    assert satisfies("glm-4.7", {"min_context": 200_001}) == (
        False,
        "context_too_small",
    )


def test_min_context_is_measured_against_max_input_tokens_when_published():
    """``min_context`` is an INPUT figure, so the INPUT bound is the ceiling.

    ``max_input_tokens`` used to be populated for five entries, served to the
    console catalogue, and read by nothing: `satisfies` compared an input budget
    against the total window it already knew was the wrong ceiling, so a
    1M-token read routed happily to gpt-5.6-luna, which accepts 922_000 of prompt
    and advertises a 1_050_000 window. The 128_000 between them is the reply's.
    """
    for model, limit, window in (
        ("gpt-5.6-luna", 922_000, 1_050_000),
        ("gpt-5.6-terra", 922_000, 1_050_000),
        ("gpt-5.6-sol", 922_000, 1_050_000),
        ("gpt-5.3-codex", 272_000, 400_000),
        ("gpt-5.4-mini", 272_000, 400_000),
    ):
        assert MODEL_CAPABILITIES[model]["context_window"] == window, model
        assert satisfies(model, {"min_context": limit}) == (True, ""), model
        assert satisfies(model, {"min_context": limit + 1}) == (
            False, "context_too_small",
        ), model
        # The window is NOT the ceiling any more, and that is the whole change.
        assert satisfies(model, {"min_context": window}) == (
            False, "context_too_small",
        ), model


def test_an_elo_that_publishes_no_input_bound_is_taken_at_its_window():
    """Absence is IGNORANCE, not a tighter limit — the module's fail-open rule.

    gpt-5.5 is the pointed case: same provider, same 1_050_000 window and same
    128_000 max_output as gpt-5.6-sol, but it declares no ``max_input_tokens``, so
    the registry has no published input bound to hold it to. Deriving one would be
    inventing a vendor fact; the honest reading is the window.
    """
    assert "max_input_tokens" not in MODEL_CAPABILITIES["gpt-5.5"]
    assert satisfies("gpt-5.5", {"min_context": 1_050_000}) == (True, "")
    assert satisfies("gpt-5.5", {"min_context": 1_050_001}) == (
        False, "context_too_small",
    )


def test_a_declared_max_input_tokens_tightens_a_registry_window():
    """The override direction that matters: an operator can only ever tighten
    the ceiling by declaring the bound the vendor publishes."""
    assert satisfies("glm-4.7", {"min_context": 190_000}) == (True, "")
    assert satisfies(
        "glm-4.7", {"min_context": 190_000}, {"max_input_tokens": 180_000}
    ) == (False, "context_too_small")


def test_a_declared_input_bound_cannot_widen_the_unsatisfiable_ceiling():
    """The pair again: a hop that would be REJECTED must not also be counted as
    evidence the request is servable."""
    hop = {
        "model": "house-model",
        "provider": "local-rail",
        "context_window": 4_000_000,
        "max_input_tokens": 100,
    }
    assert satisfies("house-model", {"min_context": 3_000_000}, hop) == (
        False, "context_too_small",
    )
    result = filter_chain([hop], {"min_context": 3_000_000})
    assert result["unsatisfiable"] == ["min_context"]
    # ...and the filter still bypasses rather than emptying the chain.
    assert result["bypassed"] is True
    assert result["eligible"] == [hop]


@pytest.mark.parametrize("floor", (0, -1, None, "", "lots", [200_000], True))
def test_a_floor_that_is_not_a_positive_size_does_not_constrain(floor):
    """A `min_context` that names no size is NOT a floor of zero and not a
    rejection either — it is a requirement the operator failed to state.

    `min_context: 0` is what a tier gets from an operator who typed the key and
    left it empty. Both readings of that value are asserted together, because the
    pair is where this drifts: `satisfies` must let every hop through AND
    `filter_chain` must not report the request unsatisfiable, or the plan would
    name a condition no hop was ever measured against.
    """
    assert satisfies("glm-4.5v", {"min_context": floor}) == (True, ""), floor

    result = filter_chain([{"model": "glm-4.5v", "provider": "zai"}],
                          {"min_context": floor})
    assert result["unsatisfiable"] == []
    assert result["rejected"] == []
    assert result["bypassed"] is False


# ---------------------------------------------------------------------------
# satisfies — unknown capabilities never reject
# ---------------------------------------------------------------------------

def test_unknown_model_returns_capability_unknown_and_passes():
    assert satisfies("no-such-model-v9", {"vision": True}) == (
        True,
        "capability_unknown",
    )


def test_unpublished_capability_on_a_known_model_is_unknown_not_a_rejection():
    ok, reason = satisfies("glm-4.7", {"vision": True}, {"vision": None})
    assert (ok, reason) == (True, "capability_unknown")


def test_declared_value_flips_a_rejection_into_a_pass():
    assert satisfies("glm-4.7", {"vision": True}) == (False, "no_vision")
    assert satisfies("glm-4.7", {"vision": True}, {"vision": True}) == (True, "")


def test_declared_context_window_flips_context_too_small_into_a_pass():
    assert satisfies("glm-4.5v", {"min_context": 200_000})[0] is False
    ok, reason = satisfies(
        "glm-4.5v", {"min_context": 200_000}, {"context_window": 262_144}
    )
    assert (ok, reason) == (True, "")


# ---------------------------------------------------------------------------
# filter_chain
# ---------------------------------------------------------------------------

def _chain():
    return [
        {"model": "glm-4.7", "provider": "zai"},
        {"model": "glm-4.6v", "provider": "zai"},
        {"model": "kimi-k3", "provider": "moonshot"},
    ]


def test_filter_chain_preserves_order_of_eligible_entries():
    result = filter_chain(_chain(), {"tool_calling": True})
    assert [entry["model"] for entry in result["eligible"]] == [
        "glm-4.7",
        "glm-4.6v",
        "kimi-k3",
    ]
    assert result["rejected"] == []
    assert result["bypassed"] is False


def test_filter_chain_leaves_eligible_entries_unchanged():
    chain = _chain()
    result = filter_chain(chain, {"tool_calling": True})
    assert result["eligible"][0] is chain[0]
    assert "reject_reason" not in result["eligible"][0]


def test_filter_chain_rejected_entries_carry_reject_reason():
    result = filter_chain(_chain(), {"vision": True})
    assert [entry["model"] for entry in result["eligible"]] == [
        "glm-4.6v",
        "kimi-k3",
    ]
    assert len(result["rejected"]) == 1
    assert result["rejected"][0]["model"] == "glm-4.7"
    assert result["rejected"][0]["reject_reason"] == "no_vision"


def test_filter_chain_does_not_mutate_the_rejected_source_entry():
    chain = _chain()
    filter_chain(chain, {"vision": True})
    assert "reject_reason" not in chain[0]


def test_filter_chain_lists_unknown_models_but_keeps_them_eligible():
    chain = [
        {"model": "glm-4.6v", "provider": "zai"},
        {"model": "mystery-elo", "provider": "somewhere"},
    ]
    result = filter_chain(chain, {"vision": True})
    assert result["unknown"] == ["mystery-elo"]
    assert [entry["model"] for entry in result["eligible"]] == [
        "glm-4.6v",
        "mystery-elo",
    ]
    assert result["bypassed"] is False


def test_an_unknown_model_named_twice_is_flagged_once():
    """``unknown`` names MODELS, not hops.

    An operator who lists the same unregistered id on two rails has one thing to
    fix, and the flag is what the console renders as "capabilities unknown for
    ..." — repeating the id there would read as two different unverified models.
    Every hop still stays eligible: fail-open is per hop, whatever the list says.
    """
    chain = [
        {"model": "mystery-elo", "provider": "somewhere"},
        {"model": "glm-4.6v", "provider": "zai"},
        {"model": "mystery-elo", "provider": "elsewhere"},
    ]
    result = filter_chain(chain, {"vision": True})
    assert result["unknown"] == ["mystery-elo"]
    assert result["eligible"] == chain


def test_requirements_that_are_not_a_mapping_constrain_nothing():
    """No requirements at all is not an impossible request.

    ``filter_chain`` reads ``requirements`` twice — once per hop through
    ``satisfies`` and once for the unsatisfiable report — and the two must agree
    that a non-mapping asks for nothing. Reporting ``min_context`` unsatisfiable
    here would name a condition the operator never wrote, which is exactly the
    lie the report exists to prevent.
    """
    chain = _chain()
    for requirements in (None, "min_context", 200_000, [("min_context", 1)]):
        result = filter_chain(chain, requirements)
        assert result["eligible"] == chain, requirements
        assert result["rejected"] == [], requirements
        assert result["unsatisfiable"] == [], requirements
        assert result["bypassed"] is False, requirements


def test_filter_chain_bypasses_when_no_elo_can_meet_the_requirement():
    chain = _chain()
    result = filter_chain(chain, {"min_context": 99_000_000})
    assert result["bypassed"] is True
    assert result["eligible"] == chain


def test_filter_chain_on_an_empty_chain_is_not_a_bypass():
    result = filter_chain([], {"vision": True})
    assert result == {
        "eligible": [],
        "rejected": [],
        "unknown": [],
        "bypassed": False,
        "unsatisfiable": [],
    }


def test_glm_52_free_is_rejected_when_tool_calling_is_required():
    # Production trap: a strong free model that hard-fails any loop which
    # sends tool definitions.
    chain = [
        {"model": "z-ai/glm-5.2:free", "provider": "openrouter"},
        {"model": "glm-4.7", "provider": "zai"},
    ]
    result = filter_chain(chain, {"tool_calling": True})
    assert [entry["model"] for entry in result["eligible"]] == ["glm-4.7"]
    assert result["rejected"][0]["model"] == "z-ai/glm-5.2:free"
    assert result["rejected"][0]["reject_reason"] == "no_tool_calling"
    assert result["bypassed"] is False


def test_glm_52_free_stays_eligible_when_tools_are_not_required():
    chain = [{"model": "z-ai/glm-5.2:free", "provider": "openrouter"}]
    result = filter_chain(chain, {"min_context": 100_000})
    assert [entry["model"] for entry in result["eligible"]] == [
        "z-ai/glm-5.2:free"
    ]
    assert result["rejected"] == []


# ---------------------------------------------------------------------------
# order_chain
# ---------------------------------------------------------------------------

def test_order_chain_sequential_is_identity():
    chain = _chain()
    assert order_chain(chain, "sequential") == chain


def test_order_chain_sequential_returns_a_new_list():
    chain = _chain()
    assert order_chain(chain, "sequential") is not chain


def test_order_chain_never_mutates_its_input_list():
    chain = _chain()
    snapshot = list(chain)
    order_chain(chain, "random", pin_primary=False, rng=random.Random(7))
    order_chain(chain, "random", pin_primary=True, rng=random.Random(7))
    assert chain == snapshot


def test_order_chain_random_with_pin_primary_fixes_index_zero():
    chain = _chain()
    for seed in range(30):
        ordered = order_chain(
            chain, "random", pin_primary=True, rng=random.Random(seed)
        )
        assert ordered[0] is chain[0]
        assert sorted(e["model"] for e in ordered) == sorted(
            e["model"] for e in chain
        )


def test_order_chain_random_with_pin_primary_still_shuffles_the_tail():
    chain = _chain()
    tails = {
        tuple(
            e["model"]
            for e in order_chain(
                chain, "random", pin_primary=True, rng=random.Random(seed)
            )[1:]
        )
        for seed in range(30)
    }
    assert len(tails) > 1


def test_order_chain_random_without_pin_primary_can_move_index_zero():
    chain = _chain()
    moved = [
        seed
        for seed in range(30)
        if order_chain(
            chain, "random", pin_primary=False, rng=random.Random(seed)
        )[0]["model"]
        != chain[0]["model"]
    ]
    assert moved


def test_order_chain_random_with_rng_none_degrades_to_sequential():
    chain = _chain()
    assert order_chain(chain, "random", pin_primary=False, rng=None) == chain


def test_order_chain_unknown_strategy_degrades_to_sequential():
    chain = _chain()
    assert order_chain(chain, "round-robin", rng=random.Random(1)) == chain


def test_order_chain_is_deterministic_for_equally_seeded_rngs():
    chain = _chain()
    first = order_chain(chain, "random", pin_primary=False, rng=random.Random(1234))
    second = order_chain(chain, "random", pin_primary=False, rng=random.Random(1234))
    assert [e["model"] for e in first] == [e["model"] for e in second]


def test_order_chain_random_on_a_single_entry_chain_is_identity():
    chain = [{"model": "glm-4.7", "provider": "zai"}]
    assert order_chain(chain, "random", pin_primary=False, rng=random.Random(3)) == chain


# ---------------------------------------------------------------------------
# derive_requirements
# ---------------------------------------------------------------------------

def test_derive_requirements_applies_the_125_percent_safety_factor():
    reqs = derive_requirements({"est_input_tokens": 200_000})
    assert reqs == {"min_context": 250_000}


def test_derive_requirements_rounds_the_safety_factor_up():
    assert derive_requirements({"est_input_tokens": 3})["min_context"] == 4
    assert derive_requirements({"est_input_tokens": 1})["min_context"] == 2
    assert derive_requirements({"est_input_tokens": 4})["min_context"] == 5


def test_derive_requirements_ignores_zero_est_input_tokens():
    assert derive_requirements({"est_input_tokens": 0}) == {}


def test_a_float_or_string_token_estimate_truncates_on_purpose():
    """The other half of the whole-hour rule, stated where it is harmless.

    A signal vector decoded from JSON can carry `est_input_tokens` as a float or
    a string, and truncating a token ESTIMATE toward zero costs nothing — the
    1.25 safety margin dwarfs the fraction. An HOUR is a position rather than an
    estimate, which is why `hours_utc`/`weekdays` refuse the same coercion (see
    test_a_fractional_hour_is_a_diagnostic_and_an_inert_window): two coercions,
    because the two values fail differently.
    """
    assert derive_requirements({"est_input_tokens": 100.9}) == {"min_context": 125}
    assert derive_requirements({"est_input_tokens": "100"}) == {"min_context": 125}
    assert derive_requirements({"est_input_tokens": "loads"}) == {}
    assert derive_requirements({"est_input_tokens": [100]}) == {}
    assert derive_requirements({}, {"min_context": "200000"}) == {
        "min_context": 200_000
    }


def test_derive_requirements_maps_the_boolean_signals():
    reqs = derive_requirements(
        {
            "needs_vision": True,
            "needs_tools": True,
            "needs_structured_output": True,
        }
    )
    assert reqs == {
        "vision": True,
        "tool_calling": True,
        "structured_output": True,
    }


def test_derive_requirements_omits_false_boolean_signals():
    assert derive_requirements({"needs_vision": False, "needs_tools": False}) == {}


def test_derive_requirements_tier_floor_takes_the_max_min_context():
    reqs = derive_requirements(
        {"est_input_tokens": 100_000}, {"min_context": 400_000}
    )
    assert reqs["min_context"] == 400_000


def test_derive_requirements_keeps_the_derived_min_context_when_it_is_higher():
    reqs = derive_requirements(
        {"est_input_tokens": 800_000}, {"min_context": 400_000}
    )
    assert reqs["min_context"] == 1_000_000


def test_derive_requirements_tier_floor_wins_on_boolean_conflict():
    reqs = derive_requirements({"needs_vision": False}, {"vision": True})
    assert reqs["vision"] is True


def test_derive_requirements_only_emits_requirement_keys():
    reqs = derive_requirements(
        {"est_input_tokens": 10, "verb_class": "hard", "needs_vision": True},
        {"min_context": 1, "flavour": "spicy"},
    )
    assert set(reqs) <= REQUIREMENT_KEYS
    assert "flavour" not in reqs


def test_derive_requirements_on_an_empty_feature_vector_is_empty():
    assert derive_requirements({}) == {}


def test_derived_requirements_feed_straight_into_satisfies():
    reqs = derive_requirements({"est_input_tokens": 200_000})
    assert satisfies("glm-4.7", reqs) == (False, "context_too_small")
    assert satisfies("glm-5.3", reqs) == (True, "")


# ---------------------------------------------------------------------------
# upstream_group / independent_rails
# ---------------------------------------------------------------------------

def test_upstream_group_collapses_nous_into_openrouter():
    assert upstream_group("nous") == "openrouter"
    assert upstream_group("openrouter") == "openrouter"


def test_upstream_group_returns_other_providers_unchanged():
    assert upstream_group("zai") == "zai"
    assert upstream_group("deepseek") == "deepseek"
    assert upstream_group("moonshot") == "moonshot"


def test_upstream_group_of_a_missing_provider_is_empty():
    assert upstream_group("") == ""
    assert upstream_group(None) == ""


def test_independent_rails_counts_groups_not_providers():
    chain = [
        {"model": "meituan/longcat-2.0:free", "provider": "nous"},
        {"model": "openrouter/free", "provider": "openrouter"},
    ]
    assert independent_rails(chain) == 1


def test_independent_rails_counts_distinct_upstreams():
    chain = [
        {"model": "meituan/longcat-2.0:free", "provider": "nous"},
        {"model": "openrouter/free", "provider": "openrouter"},
        {"model": "glm-4.7", "provider": "zai"},
    ]
    assert independent_rails(chain) == 2


def test_independent_rails_of_an_empty_chain_is_zero():
    assert independent_rails([]) == 0


def test_independent_rails_ignores_hops_without_a_provider():
    chain = [{"model": "glm-4.7", "provider": "zai"}, {"model": "mystery-elo"}]
    assert independent_rails(chain) == 1


# ---------------------------------------------------------------------------
# the rail count Python computes IS the rail count the console displays
#
# `independent_rails` exists to tell an operator whether a chain has a second
# upstream, and the console is where they read it. So the console's convention is
# not a second opinion to be compared politely: it is the definition, and it is
# parsed out of the shipped source here rather than retyped, so this asserts the
# PAIR agrees instead of asserting either side alone.
# ---------------------------------------------------------------------------

_CONSOLE_HTML = (
    Path(__file__).resolve().parents[2]
    / "webui_extension" / "hermes-smart-router" / "console.html"
)


def _console_upstream_group():
    """Return the console's ``upstreamGroup``, driven by console.html's table."""
    source = _CONSOLE_HTML.read_text(encoding="utf-8")
    body = re.search(r"const UPSTREAM = \{(.*?)\};", source, re.S)
    assert body, "console.html must declare its UPSTREAM table"
    table = dict(re.findall(r"([\w.-]+):\s*'([^']*)'", body.group(1)))
    assert table, "console.html's UPSTREAM table must have entries"
    # These two lines ARE the convention: NORMALIZE first, then fall back to the
    # normalized name when the table has no entry. Asserted so this helper cannot
    # keep agreeing with a console that has changed its mind underneath it.
    assert "String(provider == null ? '' : provider).trim().toLowerCase()" in source
    assert "return UPSTREAM[name] || name;" in source

    def group(provider):
        name = "" if provider is None else str(provider).strip().lower()
        return table.get(name, name)

    return group


def _console_rails(chain):
    """The console's ``independentRails`` over its own ``upstreamGroup``."""
    group_of = _console_upstream_group()
    groups = []
    for hop in chain:
        # `hop && hop.provider` in the console: anything that is not a mapping
        # yields no provider, and an unattributable hop is not a rail.
        group = group_of(hop.get("provider") if isinstance(hop, dict) else None)
        if group and group not in groups:
            groups.append(group)
    return len(groups)


def test_upstream_group_agrees_with_the_console_on_every_spelling():
    """A group is a COMPARISON KEY, so it is normalized whether it is mapped or
    not — `ZAI` used to come back raw and count as a rail of its own."""
    console = _console_upstream_group()
    for spelling in ("zai", "ZAI", "  ZAI  ", "Zai", "nous", "Nous", "NOUS",
                     "openrouter", "OpenRouter", " openrouter ", "deepseek",
                     "openai-codex", "  ", "", None):
        assert upstream_group(spelling) == console(spelling), spelling


def test_two_spellings_of_one_provider_are_one_rail():
    """The dangerous direction is OVER-counting: `independent_rails: 2` tells an
    operator they have a second upstream to fail over to when they have one.

    `T1 = {model: glm-4.7, provider: zai, fallback: [{model: glm-4.6, provider:
    ZAI}]}` is lint-clean — provider spelling is not something the write gate
    corrects — so nothing else in the stack was going to catch this.
    """
    chain = [
        {"model": "glm-4.7", "provider": "zai"},
        {"model": "glm-4.6", "provider": "ZAI"},
    ]
    assert independent_rails(chain) == 1
    assert independent_rails(chain) == _console_rails(chain)


def test_the_rail_count_agrees_with_the_console_on_every_chain_shape():
    for chain in (
        [{"provider": "zai"}, {"provider": "ZAI"}],
        [{"provider": "  zai  "}, {"provider": "zai"}],
        [{"provider": "nous"}, {"provider": "OpenRouter"}],
        [{"provider": "NOUS"}, {"provider": "openrouter"}, {"provider": "ZAI"}],
        [{"provider": "zai"}, {"provider": "deepseek"}],
        [{"provider": "zai"}, {}],
        # A provider of nothing but whitespace names no upstream: it used to come
        # back as "   " and count as a rail.
        [{"provider": "   "}, {"provider": "zai"}],
        # Junk in the chain — a decoded trace, a YAML list of bare model ids —
        # contributes no rail and no exception on either side.
        [{"provider": "zai"}, "glm-4.6"],
        ["glm-4.7", None],
        [],
    ):
        assert independent_rails(chain) == _console_rails(chain), chain


def test_a_provider_that_is_not_a_name_is_not_a_rail():
    """`provider: 123` names no rail here, and the console's `String(provider)`
    would make it one — the one shape the two still read differently.

    NOT asserted as agreement, because agreeing would mean adopting the answer
    that overstates redundancy. This side is deliberately the conservative one:
    a value that is not a name cannot be dialled, so it is not evidence of a
    second upstream. It is `lint()` that should be refusing it (it currently does
    not: a non-string provider passes the write gate), and the console that should
    stop coercing it.
    """
    assert upstream_group(123) == ""
    chain = [{"model": "glm-4.7", "provider": "zai"},
             {"model": "glm-4.6", "provider": 123}]
    assert independent_rails(chain) == 1


# ---------------------------------------------------------------------------
# F4 — commercial metadata must not make a model "known"
# ---------------------------------------------------------------------------

def test_capability_assertion_keys_is_the_documented_closed_set():
    assert CAPABILITY_ASSERTION_KEYS == frozenset(
        {
            "context_window",
            "max_input_tokens",
            "max_output",
            "vision",
            "tool_calling",
            "structured_output",
        }
    )


def test_billing_mode_alone_does_not_make_an_unknown_model_known():
    """F4: router.yaml mandates billing_mode on EVERY elo.

    Counting it as a capability declaration made the unknown-model warning and
    liveness's capabilities_known flag permanently dead.
    """
    assert (
        capabilities_for(
            "gpt-9-does-not-exist",
            {"model": "gpt-9-does-not-exist", "provider": "openai-codex",
             "billing_mode": "metered"},
        )
        is None
    )


def test_commercial_and_identity_metadata_alone_leaves_a_model_unknown():
    for declared in (
        {"provider": "openai-codex"},
        {"notes": "the one we always mean"},
        {"price_in": 1.0, "price_out": 2.0},
        {"price_windows": [{"hours_utc": [1, 4], "multiplier": 2.0}]},
        {"billing_mode": "plan", "notes": "x", "price_out": 3.0},
    ):
        assert capabilities_for("gpt-9-does-not-exist", declared) is None, declared


def test_one_real_capability_key_is_enough_to_make_a_model_known():
    caps = capabilities_for(
        "house-model", {"billing_mode": "metered", "context_window": 500_000}
    )
    # Known now — and the commercial field still merges, it just is not evidence.
    assert caps == {"billing_mode": "metered", "context_window": 500_000}


def test_billing_mode_only_hop_reports_capability_unknown():
    """The whole point of F4: the hop must still be flagged as unverifiable."""
    assert satisfies(
        "gpt-9-does-not-exist", {"vision": True}, {"billing_mode": "metered"}
    ) == (True, "capability_unknown")


def test_a_registry_model_is_still_known_with_only_commercial_overrides():
    caps = capabilities_for("glm-4.7", {"billing_mode": "metered"})
    assert caps["billing_mode"] == "metered"
    assert caps["context_window"] == 200_000


def test_filter_chain_lists_a_billing_mode_only_hop_as_unknown():
    chain = [
        {"model": "glm-4.6v", "provider": "zai", "billing_mode": "plan"},
        {"model": "gpt-9-does-not-exist", "provider": "openai-codex",
         "billing_mode": "metered"},
    ]
    result = filter_chain(chain, {"vision": True})
    assert result["unknown"] == ["gpt-9-does-not-exist"]


# ---------------------------------------------------------------------------
# F5 — the bypass must keep its diagnostics
# ---------------------------------------------------------------------------

def test_bypass_retains_the_per_elo_reject_reasons():
    """F5: 'nothing can meet this' is only actionable next to WHICH requirement."""
    chain = _chain()
    result = filter_chain(chain, {"min_context": 99_000_000})
    assert result["bypassed"] is True
    assert result["eligible"] == chain
    assert [(hop["model"], hop["reject_reason"]) for hop in result["rejected"]] == [
        ("glm-4.7", "context_too_small"),
        ("glm-4.6v", "context_too_small"),
        ("kimi-k3", "context_too_small"),
    ]


def test_bypass_diagnostics_are_not_excluded_from_eligible():
    """On the bypass path `rejected` is informational, NOT a removal list."""
    chain = _chain()
    result = filter_chain(chain, {"min_context": 99_000_000})
    rejected_models = {hop["model"] for hop in result["rejected"]}
    eligible_models = {hop["model"] for hop in result["eligible"]}
    assert rejected_models <= eligible_models


def test_bypass_rejected_copies_do_not_mutate_the_source_entries():
    chain = _chain()
    filter_chain(chain, {"min_context": 99_000_000})
    assert all("reject_reason" not in hop for hop in chain)


def test_the_500_file_refactor_reports_every_reason():
    """The reproduction from the review: 'refactor the auth module across 500
    files' -> est_input_tokens 2_000_012 -> min_context 2_500_015 -> every hop
    rejected. The console must be able to name the requirement nothing met.
    """
    requirements = derive_requirements({"est_input_tokens": 2_000_012})
    assert requirements["min_context"] == 2_500_015
    chain = [
        {"model": "glm-5.3", "provider": "zai"},
        {"model": "gpt-5.6-luna", "provider": "openai-codex"},
        {"model": "deepseek-v4-flash", "provider": "deepseek"},
    ]
    result = filter_chain(chain, requirements)
    assert result["bypassed"] is True
    assert len(result["rejected"]) == 3
    assert {hop["reject_reason"] for hop in result["rejected"]} == {
        "context_too_small"
    }
    assert result["unsatisfiable"] == ["min_context"]


def test_max_registered_context_agrees_with_the_ceiling_satisfies_uses():
    """The constant and the runtime helper are two spellings of one rule.

    ``MAX_REGISTERED_CONTEXT`` runs at import, before ``_as_int`` is bound, so it
    inlines the min-of-the-two rule that ``_input_ceiling`` implements for the
    request path. Two spellings is one drift away from the defect class this
    change keeps producing — the reported ceiling saying a floor is reachable that
    the filter rejects — so the agreement is asserted per entry, not sampled.
    """
    for model, entry in MODEL_CAPABILITIES.items():
        ceiling = caps_module._input_ceiling(entry)
        assert ceiling is not None, model
        assert ceiling <= entry["context_window"], model
        assert MAX_REGISTERED_CONTEXT >= ceiling, model
    assert MAX_REGISTERED_CONTEXT == max(
        caps_module._input_ceiling(entry) for entry in MODEL_CAPABILITIES.values()
    )
    # Unchanged by honouring max_input_tokens: the widest rails (mimo-v2.5,
    # mimo-v2.5-pro, gpt-5.5) publish no separate input bound.
    assert MAX_REGISTERED_CONTEXT == 1_050_000


def test_no_registered_model_can_serve_a_floor_reported_unsatisfiable():
    """The two sides of "impossible", asserted against each other.

    ``unsatisfiable`` is a claim about the whole roster and ``satisfies`` is the
    per-elo verdict; the console renders the first and the router runs the second.
    Swept just above and just below the ceiling so a stale constant cannot pass.
    """
    for floor in (
        MAX_REGISTERED_CONTEXT - 1,
        MAX_REGISTERED_CONTEXT,
        MAX_REGISTERED_CONTEXT + 1,
        922_000,
        922_001,
        272_001,
    ):
        servers = [
            model
            for model in MODEL_CAPABILITIES
            if satisfies(model, {"min_context": floor}) == (True, "")
        ]
        impossible = filter_chain(
            [{"model": model, "provider": MODEL_CAPABILITIES[model]["provider"]}
             for model in MODEL_CAPABILITIES],
            {"min_context": floor},
        )["unsatisfiable"]
        assert bool(servers) == (impossible == []), (floor, servers[:3])


def test_an_ordinary_rejection_is_not_reported_as_unsatisfiable():
    result = filter_chain(_chain(), {"min_context": 500_000})
    assert result["unsatisfiable"] == []
    assert [hop["model"] for hop in result["eligible"]] == ["kimi-k3"]


def test_unsatisfiable_is_named_even_when_a_fail_open_hop_stays_eligible():
    """The requirement is pathological whether or not an unknown elo passes."""
    result = filter_chain(
        [{"model": "mystery-elo", "provider": "somewhere"}],
        {"min_context": 3_000_000},
    )
    assert result["bypassed"] is False
    assert result["unsatisfiable"] == ["min_context"]


def test_a_declared_bigger_window_makes_the_requirement_satisfiable_again():
    result = filter_chain(
        [{"model": "house-model", "provider": "local-rail",
          "context_window": 4_000_000}],
        {"min_context": 3_000_000},
    )
    assert result["unsatisfiable"] == []
    assert result["bypassed"] is False


# ---------------------------------------------------------------------------
# The clock is injected, never read
# ---------------------------------------------------------------------------

def test_capabilities_module_never_reads_the_wall_clock():
    """Load-bearing: reading the clock here would make every routing test flaky.

    Asserted over the AST rather than the text, so the module can still DISCUSS
    ``now()`` in its docstring while never calling it.
    """
    tree = ast.parse(inspect.getsource(caps_module))
    called = set()
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
            elif isinstance(node.func, ast.Name):
                called.add(node.func.id)
        elif isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])

    assert not called & {
        "now", "utcnow", "today", "monotonic", "time", "fromtimestamp", "open",
    }
    # No IO, no state, no network: the import list is the proof. ``re`` is the
    # ISO-date shape check for price_windows_verified — pure pattern matching.
    assert imported <= {"__future__", "datetime", "random", "re", "typing"}


def test_the_reference_clocks_are_the_weekdays_they_claim():
    assert _at(MON, 0).weekday() == 0
    assert _at(WED, 0).weekday() == 2
    assert _at(FRI, 0).weekday() == 4
    assert _at(SAT, 0).weekday() == 5
    assert _at(SUN, 0).weekday() == 6


# ---------------------------------------------------------------------------
# price_multiplier — half-open [start, end) boundaries
# ---------------------------------------------------------------------------

def test_deepseek_peak_boundaries_are_half_open():
    # 01:00-04:00 and 06:00-10:00 UTC, Monday through Friday (WED here).
    expected = {
        0: 1.0, 1: 2.0, 2: 2.0, 3: 2.0, 4: 1.0, 5: 1.0,
        6: 2.0, 7: 2.0, 9: 2.0, 10: 1.0, 23: 1.0,
    }
    for hour, multiplier in expected.items():
        assert price_multiplier(
            "deepseek-v4-pro", _at(WED, hour)
        ) == multiplier, hour
        assert price_multiplier(
            "deepseek-v4-flash", _at(WED, hour)
        ) == multiplier, hour


def test_deepseek_peak_does_not_reach_the_weekend():
    """The vendor bills the whole weekend off-peak, and said so only on 2026-08-22.

    Measured 2026-08-26 on api-docs.deepseek.com: "Peak hours are 01:00 - 04:00
    and 06:00 - 10:00 UTC, Monday through Friday (all other hours are
    off-peak)". The clause is NOT in the official changelog — Wayback snapshots
    of the same URL bracket the edit between 21/08 15:13 (absent) and 24/08 17:18
    (present). Our entry was right when it was written and went stale in four
    days, which is the whole argument for checking a vendor page instead of
    trusting the last reading of it.

    Without the weekday gate the router priced 14 h/week (7 h on each weekend
    day) at 2.0x while the invoice says 1.0x. That never overbills: it routes
    away from deepseek toward rivals that are not actually cheaper, and the money
    leaves through the other provider's bill where nobody is looking for it.
    """
    for day in (SAT, SUN):
        for hour in (1, 2, 3, 6, 8, 9):
            assert price_multiplier("deepseek-v4-pro", _at(day, hour)) == 1.0, (day, hour)
            assert price_multiplier("deepseek-v4-flash", _at(day, hour)) == 1.0, (day, hour)
    # and the weekday peak is untouched
    assert price_multiplier("deepseek-v4-pro", _at(WED, 2)) == 2.0
    assert price_multiplier("deepseek-v4-pro", _at(FRI, 8)) == 2.0


def test_the_start_hour_is_inside_and_the_end_hour_is_outside():
    assert price_multiplier("deepseek-v4-pro", _at(WED, 6)) == 2.0
    assert price_multiplier("deepseek-v4-pro", _at(WED, 9, 59)) == 2.0
    assert price_multiplier("deepseek-v4-pro", _at(WED, 10)) == 1.0


def test_xiaomi_bills_flat_because_the_discount_is_plan_only():
    """The 0.8x night rate is a Token Plan credit coefficient, not a metered rate.

    Measured 2026-08-26 on the vendor's docs: the coefficient is scoped to
    「Credit 消耗系数」 of the prepaid 套餐, and the pay-as-you-go page enumerates
    its billing in five bullets — unit, cache-hit, cache-write, ASR duration,
    search calls — with zero hits for 错峰/优惠时段/时段. This install bills
    pay-as-you-go, so every hour costs the same. Carrying the window anyway made
    the router believe metered cost fell 20% for 8 h/day, i.e. real cost was
    1.25x its own estimate there.
    """
    for hour in (0, 15, 16, 20, 23):
        assert price_multiplier("mimo-v2.5", _at(WED, hour)) == 1.0, hour
        assert price_multiplier("mimo-v2.5-pro", _at(WED, hour)) == 1.0, hour
    assert "price_windows" not in MODEL_CAPABILITIES["mimo-v2.5"]


def test_a_declared_discount_window_scales_below_the_base_rate():
    """The discount MECHANISM, on a declared rail instead of a vendor promotion."""
    assert price_multiplier(DISCOUNT_RAIL, _at(WED, 15), DISCOUNT_CAPS) == 1.0
    assert price_multiplier(DISCOUNT_RAIL, _at(WED, 16), DISCOUNT_CAPS) == 0.8
    assert price_multiplier(DISCOUNT_RAIL, _at(WED, 23), DISCOUNT_CAPS) == 0.8
    # end == 24 is midnight-exclusive: the next day's 00:00 is outside.
    assert price_multiplier(DISCOUNT_RAIL, _at(WED, 0), DISCOUNT_CAPS) == 1.0


def test_zai_peak_is_gated_to_weekdays():
    for day in (MON, WED, FRI):
        assert price_multiplier("glm-5.3", _at(day, 7)) == 2.0, day
    # A weekend hour that would otherwise match bills off-peak.
    assert price_multiplier("glm-5.3", _at(SAT, 7)) == 1.0
    assert price_multiplier("glm-5.3", _at(SUN, 9)) == 1.0


def test_every_plan_covered_zai_model_carries_the_weekday_peak():
    # The set is DERIVED, not listed by hand. This test used to name four models
    # and the vendor dropped three of them from the plan's credit table on
    # 2026-08-27; a hand-written list only fails for the ids it still happens to
    # name, so it went on asserting a peak on glm-4.7 while saying nothing about
    # whether the plan roster it claimed to cover was still the plan roster. The
    # credit peak is a property of PLAN COVERAGE, so the set under test has to be
    # the plan-covered set.
    plan_zai = sorted(
        model
        for model, entry in MODEL_CAPABILITIES.items()
        if entry.get("provider") == "zai" and entry.get("billing_mode") == "plan"
    )
    assert plan_zai == ["glm-5.3", "glm-5.3-flash"], (
        "the plan's credit table lists exactly these two (read 2026-08-27); "
        "re-read the vendor before widening this"
    )
    for model in plan_zai:
        assert price_multiplier(model, _at(WED, 7)) == 2.0, model
        assert price_multiplier(model, _at(SAT, 7)) == 1.0, model


def test_a_metered_zai_model_has_no_window():
    assert price_multiplier("glm-4.6", _at(WED, 7)) == 1.0
    assert price_multiplier("glm-5.2", _at(WED, 7)) == 1.0


def test_the_plan_primary_records_the_vendor_facts_it_was_read_from():
    """glm-5.3-flash, as published 2026-08-26 and read 2026-08-27.

    Pinned because every one of these decides routing: the window feeds
    `min_context`, `vision` decides whether a screenshot can stay on the plan
    rail, and the price is LIST rather than the 50% launch promo (0.075/0.25),
    which expires 2026-09-09 16:00 UTC — a chain ordered on a discount with an
    expiry doubles in cost that morning without a config change.
    """
    entry = MODEL_CAPABILITIES["glm-5.3-flash"]
    assert entry["provider"] == "zai"
    assert entry["billing_mode"] == "plan"
    assert entry["context_window"] == 1_000_000
    assert entry["max_output"] == 131_072
    # The first plan-covered model that can see: text + image + video natively.
    assert entry["vision"] is True
    assert entry["tool_calling"] is True
    assert (entry["price_in"], entry["price_out"]) == (0.15, 0.50)
    # It is priced AND plan-covered, so it is the live example of the case
    # `_BILLING_RANK` exists for.
    assert effective_price("glm-5.3-flash", None) == pytest.approx((0.15, 0.50))


def test_the_ids_the_plan_dropped_lost_the_credit_peak_with_it():
    """peak/off-peak is a plan-CREDIT rule, so leaving the plan leaves the window.

    glm-4.7, glm-5-turbo and glm-4.6v are still purchasable metered at their
    listed prices, and a metered zai call is billed flat at every hour.  Keeping
    their 2.0x would price them as if an allowance they no longer draw on were
    still doubling, on the one rail where the number IS dollars.
    """
    for model in ("glm-4.7", "glm-5-turbo", "glm-4.6v"):
        assert MODEL_CAPABILITIES[model]["billing_mode"] == "metered", model
        assert "price_windows" not in MODEL_CAPABILITIES[model], model
        for hour in (7, 15, 23):
            assert price_multiplier(model, _at(WED, hour)) == 1.0, (model, hour)


def test_the_two_primary_rails_peak_at_the_same_hour():
    """06:00-10:00 UTC is double price on deepseek AND zai simultaneously."""
    when = _at(WED, 7)
    assert price_multiplier("deepseek-v4-pro", when) == 2.0
    assert price_multiplier("glm-5.3", when) == 2.0


def test_when_none_is_one_point_zero_everywhere():
    for model in MODEL_CAPABILITIES:
        assert price_multiplier(model) == 1.0, model
        assert price_multiplier(model, None) == 1.0, model


def test_price_multiplier_of_an_unknown_model_is_one():
    assert price_multiplier("gpt-9-does-not-exist", _at(WED, 7)) == 1.0
    assert price_multiplier("", _at(WED, 7)) == 1.0


def test_a_flat_priced_model_is_one_at_every_hour():
    for hour in range(24):
        assert price_multiplier("kimi-k3", _at(WED, hour)) == 1.0


def test_an_aware_non_utc_clock_is_converted_to_utc():
    # 04:00 in UTC-03 is 07:00 UTC, inside both primary peaks.
    local = datetime(2026, 8, 19, 4, 0, tzinfo=timezone(timedelta(hours=-3)))
    assert price_multiplier("deepseek-v4-pro", local) == 2.0
    assert price_multiplier("glm-5.3", local) == 2.0


def test_a_naive_clock_is_assumed_to_be_utc():
    assert price_multiplier("deepseek-v4-pro", datetime(2026, 8, 19, 7, 0)) == 2.0
    assert price_multiplier("deepseek-v4-pro", datetime(2026, 8, 19, 12, 0)) == 1.0


def test_an_unusable_clock_is_treated_as_no_clock():
    for junk in ("07:00", 7, [], {}, object()):
        assert price_multiplier("deepseek-v4-pro", junk) == 1.0, junk


def test_a_clock_that_cannot_be_moved_to_utc_is_no_clock():
    """Converting to UTC can RAISE, and a price is not worth a traceback.

    ``datetime.min`` in a positive UTC offset is five hours before the earliest
    instant Python can represent, so ``astimezone`` overflows on it — provoked
    here rather than patched, because the point is that the real conversion is
    what fails. Every time-dependent answer degrades to the time-agnostic one,
    which is the same reading ``when=None`` gets.
    """
    edge = datetime.min.replace(tzinfo=timezone(timedelta(hours=5)))
    with pytest.raises(OverflowError):
        edge.astimezone(timezone.utc)

    assert price_multiplier("glm-5.3", edge) == 1.0
    assert in_expensive_window("glm-5.3", edge) is False
    assert next_window_change("glm-5.3", edge) is None
    assert effective_price("deepseek-v4-pro", edge) == effective_price(
        "deepseek-v4-pro", None
    )


class _ClockShaped:
    """Clock-SHAPED, not a clock: what a decoded trace or a stub hands over."""

    tzinfo = None

    def __init__(self, hour, weekday):
        self.hour = hour
        self._weekday = weekday

    def weekday(self):
        return self._weekday


def test_a_clock_shaped_object_answering_in_range_is_read_as_a_clock():
    """The control for the test below: this object IS usable, so the 1.0s there
    come from the value being out of range and not from the shape being rejected.
    """
    # Monday 07:00 UTC — inside zai's weekday plan-credit peak.
    assert price_multiplier("glm-5.3", _ClockShaped(7, 0)) == 2.0
    # Saturday, same hour: the whole weekend bills off-peak.
    assert price_multiplier("glm-5.3", _ClockShaped(7, 5)) == 1.0


@pytest.mark.parametrize(
    "hour,weekday",
    (
        (7, "Monday"),   # a weekday that is not a number at all
        (7, 7),          # 0..6, and 7 is not a day
        (7, -1),
        (24, 0),         # 0..23, and 24 is tomorrow's midnight, not an hour
        (99, 0),
        (-1, 0),
    ),
)
def test_an_hour_or_weekday_outside_the_dial_is_no_clock(hour, weekday):
    """An impossible hour is not clamped to a possible one.

    Guessing would price the request at some OTHER hour's rate and report that as
    the answer; refusing the clock prices it at the base rate and says so through
    every time-dependent surface at once.
    """
    when = _ClockShaped(hour, weekday)
    assert price_multiplier("glm-5.3", when) == 1.0
    assert in_expensive_window("glm-5.3", when) is False
    assert next_window_change("glm-5.3", when) is None


def test_no_clock_is_the_neutral_answer_for_every_registered_model():
    """``when=None`` is time-agnostic ACROSS THE WHOLE REGISTRY, not per call.

    Swept rather than sampled: a caller that never injects a clock must see the
    base rate, no window and no scheduled change everywhere, which is what makes
    "the clock is a parameter" true rather than aspirational.
    """
    for model, entry in MODEL_CAPABILITIES.items():
        assert price_multiplier(model, None) == 1.0, model
        assert in_expensive_window(model, None) is False, model
        assert next_window_change(model, None) is None, model
        base_in = entry.get("price_in")
        base_out = entry.get("price_out")
        if base_in is None or base_out is None:
            assert effective_price(model, None) is None, model
        else:
            assert effective_price(model, None) == pytest.approx(
                (base_in, base_out)
            ), model


def test_a_declared_window_overrides_the_registry():
    # An operator correcting a stale window in YAML, no code change.
    declared = {"price_windows": [{"hours_utc": [12, 13], "multiplier": 3.0}]}
    assert price_multiplier("deepseek-v4-pro", _at(WED, 12), declared) == 3.0
    assert price_multiplier("deepseek-v4-pro", _at(WED, 7), declared) == 1.0


def test_a_malformed_declared_window_falls_back_to_the_base_rate():
    # start > end would need wrap-around arithmetic; it is a lint error instead.
    declared = {"price_windows": [{"hours_utc": [22, 3], "multiplier": 2.0}]}
    assert price_multiplier("kimi-k3", _at(WED, 23), declared) == 1.0
    assert price_window_diagnostics("kimi-k3", declared["price_windows"])


def test_price_multiplier_never_mutates_the_registry():
    declared = {"price_windows": [{"hours_utc": [0, 1], "multiplier": 5.0}]}
    price_multiplier("deepseek-v4-pro", _at(WED, 0), declared)
    assert MODEL_CAPABILITIES["deepseek-v4-pro"]["price_windows"] == [
        {"hours_utc": [1, 4], "weekdays": [0, 1, 2, 3, 4], "multiplier": 2.0},
        {"hours_utc": [6, 10], "weekdays": [0, 1, 2, 3, 4], "multiplier": 2.0},
    ]


def test_a_merged_view_cannot_mutate_the_registry_windows():
    caps = capabilities_for("deepseek-v4-pro")
    caps["price_windows"][0]["multiplier"] = 99.0
    assert MODEL_CAPABILITIES["deepseek-v4-pro"]["price_windows"][0]["multiplier"] == 2.0


# ---------------------------------------------------------------------------
# effective_price — a None price is never 0.0
# ---------------------------------------------------------------------------

def test_effective_price_scales_the_base_rate_inside_a_peak():
    assert effective_price("deepseek-v4-pro", _at(WED, 7)) == pytest.approx(
        (1.32, 3.96)
    )
    assert effective_price("deepseek-v4-pro", _at(WED, 12)) == pytest.approx(
        (0.66, 1.98)
    )


def test_effective_price_scales_the_base_rate_inside_a_discount():
    assert effective_price(DISCOUNT_RAIL, _at(WED, 20), DISCOUNT_CAPS) == pytest.approx(
        (0.112, 0.224)
    )
    assert effective_price(DISCOUNT_RAIL, _at(WED, 12), DISCOUNT_CAPS) == pytest.approx(
        (0.14, 0.28)
    )


def test_effective_price_of_a_plan_model_without_dollars_is_none_not_zero():
    """A plan model is NOT free, and an absent dollar price is not 0.0.

    glm-5.3 used to carry this case: plan-covered and unpriced, so the assertion
    could read the shipped roster.  The vendor launched its metered API on
    2026-08-27, and the roster now has no plan-covered elo without a price — so
    the case is BUILT here rather than dropped, because it is still the case
    ``effective_price`` has to get right, and the next plan-only launch will land
    on it again.  A known model with no published price, declared onto the plan
    rail: known, so the answer cannot come from the unknown-model path.
    """
    plan_no_price = {"billing_mode": "plan"}
    for when in (None, _at(WED, 7), _at(SAT, 7)):
        assert effective_price("glm-4.5-flash", when, plan_no_price) is None
    assert effective_price("glm-4.5-flash", _at(WED, 7), plan_no_price) != (0.0, 0.0)


def test_a_plan_model_that_publishes_dollars_still_reports_them():
    """And the peak scales those dollars, dimensionlessly, like any other price.

    The dollars are what a plan-LESS operator pays for the same id; what makes
    the rail cheap at the margin is the billing_mode bucket in ``cheapest_now``
    and never an absent number.  That is exactly why glm-5.3 gaining a price on
    2026-08-27 reordered nothing.
    """
    assert effective_price("glm-5.3", None) == pytest.approx((1.40, 4.40))
    assert effective_price("glm-5.3", _at(WED, 7)) == pytest.approx((2.80, 8.80))
    assert effective_price("glm-5.3", _at(SAT, 7)) == pytest.approx((1.40, 4.40))


def test_effective_price_of_an_unpriced_metered_model_is_none():
    assert effective_price("glm-4.5-flash", _at(WED, 7)) is None


def test_effective_price_of_an_unknown_model_is_none():
    assert effective_price("gpt-9-does-not-exist", _at(WED, 7)) is None


def test_a_published_zero_price_is_a_price():
    assert effective_price("glm-4.7-flash", _at(WED, 7)) == (0.0, 0.0)


def test_half_a_published_price_pair_is_no_price():
    """The missing half would have to be invented — and inventing it as 0.0 is
    exactly the coercion the design forbids."""
    assert effective_price(
        "glm-4.5-flash", _at(WED, 7), {"price_in": 1.0}
    ) is None


def test_effective_price_with_no_clock_is_the_base_rate():
    assert effective_price("deepseek-v4-pro") == pytest.approx((0.66, 1.98))
    assert effective_price("mimo-v2.5", None) == pytest.approx((0.14, 0.28))


def test_zai_peak_scales_a_plan_model_that_does_publish_dollars():
    assert effective_price("glm-5.3-flash", _at(WED, 7)) == pytest.approx(
        (0.30, 1.00)
    )
    assert effective_price("glm-5.3-flash", _at(SAT, 7)) == pytest.approx(
        (0.15, 0.50)
    )
    # LIST, not the launch promo: 0.075/0.25 expires 2026-09-09 16:00 UTC, and a
    # chain ordered on a discount with an expiry silently doubles that morning.
    assert MODEL_CAPABILITIES["glm-5.3-flash"]["price_out"] == 0.50


# ---------------------------------------------------------------------------
# in_expensive_window
# ---------------------------------------------------------------------------

def test_in_expensive_window_only_for_a_multiplier_above_one():
    assert in_expensive_window("deepseek-v4-pro", _at(WED, 7)) is True
    assert in_expensive_window("deepseek-v4-pro", _at(WED, 12)) is False
    assert in_expensive_window("glm-5.3", _at(WED, 7)) is True
    assert in_expensive_window("glm-5.3", _at(SAT, 7)) is False


def test_a_cheap_window_is_not_an_expensive_one():
    assert price_multiplier(DISCOUNT_RAIL, _at(WED, 20), DISCOUNT_CAPS) == 0.8
    assert in_expensive_window(DISCOUNT_RAIL, _at(WED, 20), DISCOUNT_CAPS) is False


def test_in_expensive_window_without_a_clock_is_false():
    assert in_expensive_window("deepseek-v4-pro") is False
    assert in_expensive_window("gpt-9-does-not-exist", _at(WED, 7)) is False


# ---------------------------------------------------------------------------
# next_window_change
# ---------------------------------------------------------------------------

#: A rail with a CHEAP window, DECLARED here instead of read from the registry.
#: No registry entry carries a discount any more: xiaomi's 0.8x was scoped to the
#: prepaid Token Plan (measured 2026-08-26; this install bills pay-as-you-go), so
#: it was removed. A mechanism test must not depend on a vendor's current
#: promotion to still have an example — that lesson cost two rounds of red on
#: 2026-08-26, first when deepseek stopped being ungated and then here.
DISCOUNT_RAIL = "discount-rail"
DISCOUNT_CAPS = {
    # A capability assertion is what makes a model KNOWN to capabilities_for();
    # commercial fields alone deliberately do not (see its docstring), so a rail
    # declared only by price would silently answer 1.0 at every hour.
    "context_window": 200_000,
    "provider": "synthetic",
    "billing_mode": "metered",
    "price_in": 0.14,
    "price_out": 0.28,
    "price_windows": [{"hours_utc": [16, 24], "multiplier": 0.8}],
}

#: A plan-covered rail that publishes NO dollar price — the case glm-5.3 carried
#: until the vendor launched its metered API on 2026-08-27. The shipped roster no
#: longer contains one, and the ordering rules that turn on it (a plan elo is
#: bucketed by billing_mode, an absent price is never 0.0, and inside a bucket an
#: unpriced elo sorts behind a priced one) still have to hold — the next plan-only
#: launch will land on them again. Declared, so the tests keep asserting the
#: MECHANISM instead of following one vendor's catalogue.
PLAN_NO_PRICE_RAIL = "plan-no-list-price"
PLAN_NO_PRICE_CAPS = {
    "context_window": 200_000,
    "provider": "synthetic",
    "billing_mode": "plan",
}

#: Weekday numbers for the reference week, matching ``datetime.weekday()``.
MONDAY, WEDNESDAY, THURSDAY = 0, 2, 3


def _change(hour: int, weekday: int, hours_ahead: int, multiplier: float) -> dict:
    """The full shape :func:`next_window_change` returns — day included."""
    return {
        "hour": hour,
        "weekday": weekday,
        "hours_ahead": hours_ahead,
        "multiplier": multiplier,
    }


def test_next_window_change_reports_the_end_of_a_peak():
    assert next_window_change("deepseek-v4-pro", _at(WED, 3)) == _change(
        4, WEDNESDAY, 1, 1.0
    )
    assert next_window_change("deepseek-v4-pro", _at(WED, 7)) == _change(
        10, WEDNESDAY, 3, 1.0
    )


def test_next_window_change_reports_the_start_of_the_next_peak():
    assert next_window_change("deepseek-v4-pro", _at(WED, 5)) == _change(
        6, WEDNESDAY, 1, 2.0
    )


def test_next_window_change_crosses_the_day_boundary():
    """The DAY is part of the answer: 01:00 tomorrow is not 01:00 today."""
    # Off-peak from 10:00; the next change is 01:00 TOMORROW, 15 hours out.
    assert next_window_change("deepseek-v4-pro", _at(WED, 10)) == _change(
        1, THURSDAY, 15, 2.0
    )
    assert next_window_change("deepseek-v4-pro", _at(WED, 23)) == _change(
        1, THURSDAY, 2, 2.0
    )
    # Inside a 16:00-00:00 discount, the change is midnight — Thursday's.
    assert next_window_change(DISCOUNT_RAIL, _at(WED, 20), DISCOUNT_CAPS) == _change(
        0, THURSDAY, 4, 1.0
    )
    # Windows begin on the hour, so minutes do not move the count.
    assert next_window_change(DISCOUNT_RAIL, _at(WED, 23, 59), DISCOUNT_CAPS) == _change(
        0, THURSDAY, 1, 1.0
    )


def test_next_window_change_crosses_the_weekend():
    """The weekday gate makes a bare hour ambiguous by up to two days.

    zai's peak is Mon-Fri only, so from Friday evening the next change is MONDAY
    06:00. Reported as the hour 6 alone that reads as "10 hours away"; the real
    answer is 58, and ``hours_ahead`` is what a countdown must use.
    """
    assert next_window_change("glm-5.3", _at(FRI, 20)) == _change(
        6, MONDAY, 58, 2.0
    )
    # The defect case: Saturday 07:00 is 47 hours from Monday 06:00, not 23.
    assert next_window_change("glm-5.3-flash", _at(SAT, 7)) == _change(
        6, MONDAY, 47, 2.0
    )
    assert next_window_change("glm-5.3", _at(SAT, 7)) == _change(
        6, MONDAY, 47, 2.0
    )
    assert next_window_change("glm-5.3", _at(SUN, 23)) == _change(
        6, MONDAY, 7, 2.0
    )


def test_next_window_change_hours_ahead_lands_on_the_hour_it_names():
    """hour/weekday and hours_ahead must describe the SAME instant."""
    for model, when, declared in (
        ("glm-5.3-flash", _at(SAT, 7), None),
        ("glm-5.3", _at(FRI, 20), None),
        ("deepseek-v4-pro", _at(WED, 10), None),
        (DISCOUNT_RAIL, _at(WED, 20), DISCOUNT_CAPS),
    ):
        change = next_window_change(model, when, declared)
        landed = when + timedelta(hours=change["hours_ahead"])
        assert (landed.hour, landed.weekday()) == (
            change["hour"], change["weekday"]
        ), model
        assert price_multiplier(model, landed, declared) == change["multiplier"], model


def test_next_window_change_of_a_flat_model_is_none():
    assert next_window_change("kimi-k3", _at(WED, 7)) is None
    assert next_window_change("gpt-5.6-luna", _at(WED, 7)) is None


def test_next_window_change_without_a_clock_or_model_is_none():
    assert next_window_change("deepseek-v4-pro") is None
    assert next_window_change("gpt-9-does-not-exist", _at(WED, 7)) is None


def test_next_window_change_of_an_all_hours_window_is_none():
    """A window covering every hour at one multiplier never changes."""
    declared = {"price_windows": [{"hours_utc": [0, 24], "multiplier": 2.0}]}
    assert price_multiplier("kimi-k3", _at(WED, 13), declared) == 2.0
    assert next_window_change("kimi-k3", _at(WED, 13), declared) is None


# ---------------------------------------------------------------------------
# order_chain — cheapest_now
# ---------------------------------------------------------------------------

def _priced_chain():
    return [
        {"model": "kimi-k3", "provider": "moonshot"},           # metered, 15.00
        {"model": "mimo-v2.5", "provider": "xiaomi"},           # metered, 0.28
        # SUBSCRIPTION, 1.20 — same dollar bucket as the two metered rails.
        {"model": "gpt-5.6-luna", "provider": "openai-codex"},
    ]


def test_cheapest_now_orders_by_effective_output_price_within_a_bucket():
    chain = [
        {"model": "kimi-k3", "provider": "moonshot"},     # metered, out 15.00
        {"model": "mimo-v2.5", "provider": "xiaomi"},     # metered, out 0.28
        {"model": "MiniMax-M3", "provider": "minimax"},   # metered, out 1.20
    ]
    ordered = order_chain(
        chain, "cheapest_now", pin_primary=False, when=_at(WED, 12)
    )
    assert [hop["model"] for hop in ordered] == [
        "mimo-v2.5", "MiniMax-M3", "kimi-k3",
    ]


def test_cheapest_now_compares_a_subscription_seat_in_dollars():
    """A rail is bucketed on the UNIT its price is quoted in, not on the seat.

    gpt-5.6-luna's 1.20 IS the per-token rate that openai-codex bills at, so it
    stays commensurable with metered mimo-v2.5's 0.28 and loses the comparison.
    Ranking a seat as already-paid instead is what freezes a chain's order at
    every hour — see the flip test below.
    """
    ordered = order_chain(
        _priced_chain(), "cheapest_now", pin_primary=False, when=_at(WED, 12)
    )
    assert [hop["model"] for hop in ordered] == [
        "mimo-v2.5", "gpt-5.6-luna", "kimi-k3",
    ]


def test_cheapest_now_subscription_versus_metered_flips_with_the_hour():
    """The shipped T2 tail — and why a seat is NOT bucketed as already-paid.

    gpt-5.6-luna is a flat 1.20 subscription seat; deepseek-v4-flash is metered
    0.66 and doubles to 1.32 inside its peak. Off-peak the metered rail is the
    cheaper token, inside the peak the seat is: one order per side of the window.
    Put the seat in a bucket ahead of metered and the two elos whose prices move
    against each other can never be compared at all — the order comes back
    identical at all 24 hours and the injected clock is decoration.
    """
    # Declared exactly as router.yaml declares them, so the override path is
    # covered too: a declared billing_mode must land in the same bucket.
    chain = [
        {"model": "gpt-5.6-luna", "provider": "openai-codex",
         "billing_mode": "subscription"},
        {"model": "deepseek-v4-flash", "provider": "deepseek",
         "billing_mode": "metered"},
    ]
    assert effective_price("gpt-5.6-luna", _at(WED, 7))[1] == 1.20
    assert effective_price("deepseek-v4-flash", _at(WED, 12))[1] == 0.66
    assert effective_price("deepseek-v4-flash", _at(WED, 7))[1] == 1.32

    assert [hop["model"] for hop in order_chain(
        chain, "cheapest_now", pin_primary=False, when=_at(WED, 12))] == [
        "deepseek-v4-flash", "gpt-5.6-luna"]
    assert [hop["model"] for hop in order_chain(
        chain, "cheapest_now", pin_primary=False, when=_at(WED, 7))] == [
        "gpt-5.6-luna", "deepseek-v4-flash"]


def test_cheapest_now_ranks_a_plan_elo_ahead_of_a_subscription_seat():
    """Only the plan rail is spent in credits, so only it leads on billing mode.

    glm-5.3 carries a 4.40 list price and is inside its 2.0x weekday peak here —
    8.80 against the seat's flat 1.20, more than seven times the number — and
    still leads, because those plan dollars are a metered SKU the operator is not
    on. The price is what a plan-LESS operator would pay for the same id; it went
    from absent to published on 2026-08-27 and the ordering did not move, which is
    the property this test exists to hold.
    """
    chain = [
        {"model": "gpt-5.6-luna", "provider": "openai-codex"},  # sub, 1.20
        {"model": "glm-5.3", "provider": "zai"},                # plan, 8.80 now
    ]
    assert effective_price("glm-5.3", _at(MON, 7))[1] == pytest.approx(8.8)
    assert [hop["model"] for hop in order_chain(
        chain, "cheapest_now", pin_primary=False, when=_at(MON, 7))] == [
        "glm-5.3", "gpt-5.6-luna"]


def test_every_billing_mode_has_an_explicit_cheapest_now_rank():
    """A mode with no rank would fall into the unknown bucket and sort LAST.

    That is a silent routing change, not a diagnostic, so the map must stay
    exhaustive over the closed set as new modes are added.
    """
    rank = caps_module._BILLING_RANK
    assert set(rank) == set(BILLING_MODES)
    assert rank["plan"] < rank["free"] < rank["metered"]
    assert max(rank.values()) < caps_module._BILLING_RANK_UNKNOWN
    # subscription and metered share a bucket: both are quoted in dollars, so the
    # price is the only thing left to separate them.
    assert rank["subscription"] == rank["metered"]


def test_cheapest_now_reorders_when_a_peak_moves_a_price():
    chain = [
        {"model": "deepseek-v4-flash", "provider": "deepseek"},  # 0.66 -> 1.32
        {"model": "MiniMax-M3", "provider": "minimax"},          # 1.20 flat
    ]
    off_peak = order_chain(chain, "cheapest_now", pin_primary=False,
                           when=_at(WED, 12))
    assert [hop["model"] for hop in off_peak] == [
        "deepseek-v4-flash", "MiniMax-M3",
    ]
    peak = order_chain(chain, "cheapest_now", pin_primary=False,
                       when=_at(WED, 7))
    assert [hop["model"] for hop in peak] == [
        "MiniMax-M3", "deepseek-v4-flash",
    ]


def test_cheapest_now_ties_keep_declared_order():
    # glm-5.2 and glm-5.1 are both metered at 4.40 out with no windows at all,
    # so the operator's declared order is the only thing left to sort on.
    when = _at(SAT, 7)
    first = [{"model": "glm-5.2", "provider": "zai"},
             {"model": "glm-5.1", "provider": "zai"}]
    second = [{"model": "glm-5.1", "provider": "zai"},
              {"model": "glm-5.2", "provider": "zai"}]
    assert effective_price("glm-5.2", when)[1] == effective_price(
        "glm-5.1", when
    )[1]
    assert [hop["model"] for hop in order_chain(
        first, "cheapest_now", pin_primary=False, when=when)] == [
        "glm-5.2", "glm-5.1"]
    assert [hop["model"] for hop in order_chain(
        second, "cheapest_now", pin_primary=False, when=when)] == [
        "glm-5.1", "glm-5.2"]


def test_cheapest_now_ties_keep_declared_order_inside_the_plan_bucket():
    """Two plan elos at one price still fall back to the declared order."""
    when = _at(SAT, 7)
    declared = {"billing_mode": "plan", "context_window": 200_000,
                "price_in": 0.60, "price_out": 2.20}
    first = [dict(declared, model="plan-a", provider="house"),
             dict(declared, model="plan-b", provider="house")]
    second = [dict(declared, model="plan-b", provider="house"),
              dict(declared, model="plan-a", provider="house")]
    assert [hop["model"] for hop in order_chain(
        first, "cheapest_now", pin_primary=False, when=when)] == [
        "plan-a", "plan-b"]
    assert [hop["model"] for hop in order_chain(
        second, "cheapest_now", pin_primary=False, when=when)] == [
        "plan-b", "plan-a"]


def test_cheapest_now_prefers_a_plan_rail_over_a_cheaper_metered_one():
    """The bucket is decided by billing_mode, NOT by the absence of a price.

    glm-5.3-flash is covered by the z.ai Coding Plan and ALSO carries a 0.50 list
    price. Compared in dollars it loses to metered mimo-v2.5 at every hour —
    1.00 against 0.28 inside zai's weekday peak, 0.50 against 0.28 outside it,
    since xiaomi bills flat — and every one of those dollars is already sunk. An
    hour already bought is the cheapest marginal token there is.

    These two are hops 1 and 3 of the SHIPPED T1 chain, so this is the live
    ordering, not a constructed one.
    """
    chain = [
        {"model": "glm-5.3-flash", "provider": "zai"},  # plan, list 0.50 out
        {"model": "mimo-v2.5", "provider": "xiaomi"},   # metered, 0.28 out
    ]
    # Monday 07:00 UTC: the plan rail at 2.0x CREDITS, mimo at its base rate.
    peak = _at(MON, 7)
    assert price_multiplier("glm-5.3-flash", peak) == 2.0
    assert effective_price("glm-5.3-flash", peak)[1] == pytest.approx(1.0)
    assert effective_price("mimo-v2.5", peak)[1] == 0.28
    assert [hop["model"] for hop in order_chain(
        chain, "cheapest_now", pin_primary=False, when=peak)] == [
        "glm-5.3-flash", "mimo-v2.5"]

    # Monday 20:00 UTC: off its peak, mimo flat as it always is now — the
    # narrowest the dollar gap ever gets, and still not a reason to move.
    off_peak = _at(MON, 20)
    assert price_multiplier("glm-5.3-flash", off_peak) == 1.0
    assert effective_price("glm-5.3-flash", off_peak)[1] == 0.5
    assert effective_price("mimo-v2.5", off_peak)[1] == pytest.approx(0.28)
    assert [hop["model"] for hop in order_chain(
        chain, "cheapest_now", pin_primary=False, when=off_peak)] == [
        "glm-5.3-flash", "mimo-v2.5"]


def test_cheapest_now_prefers_a_plan_rail_declared_second():
    """Not an artefact of declared order: the metered rail leads the chain."""
    chain = [
        {"model": "mimo-v2.5", "provider": "xiaomi"},
        {"model": "glm-5.3-flash", "provider": "zai"},  # plan, list 0.50 out
        {"model": "glm-5.3", "provider": "zai"},        # plan, list 4.40 out
    ]
    ordered = order_chain(chain, "cheapest_now", pin_primary=False,
                          when=_at(MON, 7))
    # Both plan rails first — cheaper LIST price ordering them inside the
    # bucket, which is also their plan-credit ordering (8 before 24). Note the
    # dollars being compared are both at 2.0x here (1.00 and 8.80) and mimo's
    # 0.28 beats both; the bucket is what puts them in front, not the number.
    assert [hop["model"] for hop in ordered] == [
        "glm-5.3-flash", "glm-5.3", "mimo-v2.5",
    ]


def test_cheapest_now_ranks_free_ahead_of_metered():
    """A free rail spends nothing; the cheapest metered rail still spends."""
    chain = [
        {"model": "inclusionai/ling-3.0-flash", "provider": "nous"},  # 0.0504
        {"model": "tencent/hy3:free", "provider": "nous"},            # free
    ]
    assert effective_price("inclusionai/ling-3.0-flash", _at(WED, 12))[1] > 0
    assert effective_price("tencent/hy3:free", _at(WED, 12))[1] == 0.0
    assert [hop["model"] for hop in order_chain(
        chain, "cheapest_now", pin_primary=False, when=_at(WED, 12))] == [
        "tencent/hy3:free", "inclusionai/ling-3.0-flash"]


def test_cheapest_now_ranks_every_bucket_in_marginal_cost_order():
    """plan credits, then free, then the dollar rails, then undescribable.

    ``subscription`` and ``metered`` share the dollar bucket, so mimo-v2.5's 0.28
    leads the seat's 1.20 there; an unpriced dollar rail sorts behind both of them
    and an elo with no billing mode at all sorts behind everything.
    """
    chain = [
        {"model": "mimo-v2.5", "provider": "xiaomi"},            # metered, 0.28
        {"model": "glm-4.7-flash", "provider": "zai"},           # free
        {"model": "glm-5.3-flash", "provider": "zai"},           # plan, priced
        {"model": "glm-4.5-flash", "provider": "zai"},           # metered, no price
        {"model": "gpt-5.6-luna", "provider": "openai-codex"},   # sub, 1.20
        # Known by capability assertion, but nothing describes how it is billed.
        {"model": "house-local-7b", "provider": "house",
         "context_window": 32_768},
    ]
    ordered = order_chain(chain, "cheapest_now", pin_primary=False,
                          when=_at(MON, 7))
    assert [hop["model"] for hop in ordered] == [
        "glm-5.3-flash", "glm-4.7-flash", "mimo-v2.5", "gpt-5.6-luna",
        "glm-4.5-flash", "house-local-7b",
    ]


def test_cheapest_now_never_compares_an_absent_price_numerically():
    """An unpriced elo is ordered by bucket and declared index, never as 0.0.

    Includes a HALF-priced declaration, which :func:`effective_price` reports as
    None rather than inventing the missing side — the shape that would raise if a
    None ever reached the float comparison.
    """
    half_priced = {"model": "half-priced", "provider": "house",
                   "context_window": 200_000, "billing_mode": "metered",
                   "price_in": 0.10}
    unpriced_plan = dict(PLAN_NO_PRICE_CAPS, model=PLAN_NO_PRICE_RAIL)
    chain = [
        dict(half_priced),
        {"model": "glm-4.5-flash", "provider": "zai"},   # metered, no price
        {"model": "mimo-v2.5", "provider": "xiaomi"},    # metered, 0.28
        dict(unpriced_plan),                             # plan, no price
    ]
    when = _at(WED, 12)
    assert effective_price("half-priced", when, half_priced) is None
    assert effective_price("glm-4.5-flash", when) is None
    assert effective_price(PLAN_NO_PRICE_RAIL, when, PLAN_NO_PRICE_CAPS) is None
    # Priced dollar rail first, then the two unpriced ones in DECLARED order.
    assert [hop["model"] for hop in order_chain(
        chain, "cheapest_now", pin_primary=False, when=when)] == [
        PLAN_NO_PRICE_RAIL, "mimo-v2.5", "half-priced", "glm-4.5-flash"]


def test_inside_the_plan_bucket_a_priced_elo_leads_an_unpriced_one():
    """No price is not a cheaper price: it is no information, so it sorts last.

    Both members are plan-covered, so the bucket cannot separate them and the
    price is all that is left. glm-5.3-flash publishes 0.50; the other publishes
    nothing, and nothing does not beat a number.
    """
    chain = [
        dict(PLAN_NO_PRICE_CAPS, model=PLAN_NO_PRICE_RAIL),  # plan, price None
        {"model": "glm-5.3-flash", "provider": "zai"},        # plan, 0.50 out
    ]
    ordered = order_chain(chain, "cheapest_now", pin_primary=False,
                          when=_at(MON, 7))
    assert [hop["model"] for hop in ordered] == [
        "glm-5.3-flash", PLAN_NO_PRICE_RAIL,
    ]


def test_cheapest_now_places_an_unpriced_plan_model_by_billing_rank():
    """A plan rail with no dollar price. Treated as 0.0 it would merely TIE with
    the free rail and keep declared order; by billing rank it sorts ahead of it.
    """
    chain = [
        {"model": "glm-4.7-flash", "provider": "zai"},  # free, published 0.00
        dict(PLAN_NO_PRICE_CAPS, model=PLAN_NO_PRICE_RAIL),  # plan, price None
    ]
    ordered = order_chain(chain, "cheapest_now", pin_primary=False,
                          when=_at(SAT, 7))
    assert [hop["model"] for hop in ordered] == [
        PLAN_NO_PRICE_RAIL, "glm-4.7-flash",
    ]


def test_cheapest_now_sorts_an_unpriced_metered_model_last():
    """An unpublished METERED price is a cost risk, not a freebie."""
    chain = [
        {"model": "glm-4.5-flash", "provider": "zai"},   # metered, price None
        {"model": "kimi-k3", "provider": "moonshot"},    # out 15.00
        {"model": "mimo-v2.5", "provider": "xiaomi"},    # out 0.28
    ]
    ordered = order_chain(chain, "cheapest_now", pin_primary=False,
                          when=_at(WED, 12))
    assert [hop["model"] for hop in ordered] == [
        "mimo-v2.5", "kimi-k3", "glm-4.5-flash",
    ]


def test_cheapest_now_ranks_plan_ahead_of_priced_ahead_of_unpriced_metered():
    chain = [
        {"model": "glm-4.5-flash", "provider": "zai"},  # unpriced metered
        {"model": "kimi-k3", "provider": "moonshot"},   # priced
        {"model": "glm-5.3", "provider": "zai"},        # unpriced plan
    ]
    ordered = order_chain(chain, "cheapest_now", pin_primary=False,
                          when=_at(SAT, 7))
    assert [hop["model"] for hop in ordered] == [
        "glm-5.3", "kimi-k3", "glm-4.5-flash",
    ]


def test_cheapest_now_with_no_clock_degrades_to_sequential():
    """No clock means the DECLARED order, never a guess at the hour.

    Asserted against the clocked answer as well, so this stays a real degradation
    rather than a chain that happens to be in price order already.
    """
    chain = _priced_chain()
    assert order_chain(chain, "cheapest_now", pin_primary=False) == chain
    assert order_chain(chain, "cheapest_now", pin_primary=False, when=None) == chain
    assert order_chain(chain, "cheapest_now", pin_primary=True, when=None) == chain
    assert order_chain(
        chain, "cheapest_now", pin_primary=False, when=_at(WED, 12)
    ) != chain


def test_cheapest_now_with_pin_primary_reorders_the_tail_only():
    chain = _priced_chain()
    ordered = order_chain(
        chain, "cheapest_now", pin_primary=True, when=_at(WED, 12)
    )
    assert ordered[0] is chain[0]
    # The pinned primary keeps its slot even though it is the priciest hop, and
    # the tail behind it is ordered in dollars: mimo-v2.5 0.28 before the
    # subscription seat's 1.20.
    assert [hop["model"] for hop in ordered] == [
        "kimi-k3", "mimo-v2.5", "gpt-5.6-luna",
    ]


def test_cheapest_now_returns_a_new_list_and_never_mutates_the_input():
    chain = _priced_chain()
    snapshot = list(chain)
    ordered = order_chain(chain, "cheapest_now", pin_primary=False,
                          when=_at(WED, 12))
    assert ordered is not chain
    assert chain == snapshot


def test_cheapest_now_keeps_the_original_entry_objects():
    chain = _priced_chain()
    ordered = order_chain(chain, "cheapest_now", pin_primary=False,
                          when=_at(WED, 12))
    assert {id(hop) for hop in ordered} == {id(hop) for hop in chain}


def test_cheapest_now_on_a_single_entry_chain_is_identity():
    chain = [{"model": "kimi-k3", "provider": "moonshot"}]
    assert order_chain(chain, "cheapest_now", pin_primary=False,
                       when=_at(WED, 12)) == chain


def test_cheapest_now_tolerates_junk_entries():
    chain = [{"model": "kimi-k3", "provider": "moonshot"}, "not-a-hop",
             {"model": "mimo-v2.5", "provider": "xiaomi"}]
    ordered = order_chain(chain, "cheapest_now", pin_primary=False,
                          when=_at(WED, 12))
    assert len(ordered) == 3
    assert ordered[0]["model"] == "mimo-v2.5"


def test_a_clock_does_not_change_the_other_strategies():
    chain = _priced_chain()
    assert order_chain(chain, "sequential", when=_at(WED, 7)) == chain
    assert order_chain(chain, "round-robin", when=_at(WED, 7)) == chain
    seeded = order_chain(chain, "random", pin_primary=False,
                         rng=random.Random(5), when=_at(WED, 7))
    unclocked = order_chain(chain, "random", pin_primary=False,
                            rng=random.Random(5))
    assert [hop["model"] for hop in seeded] == [
        hop["model"] for hop in unclocked
    ]


def test_fallback_strategies_is_the_documented_closed_set():
    assert FALLBACK_STRATEGIES == frozenset(
        {"sequential", "random", "cheapest_now"}
    )


# ---------------------------------------------------------------------------
# apply_time_policy
# ---------------------------------------------------------------------------

def _mixed_chain():
    return [
        {"model": "deepseek-v4-pro", "provider": "deepseek"},
        {"model": "glm-5.3", "provider": "zai"},
        {"model": "gpt-5.6-luna", "provider": "openai-codex"},
        {"model": "mimo-v2.5", "provider": "xiaomi"},
    ]


def test_avoid_peak_demotes_without_removing():
    chain = _mixed_chain()
    result = apply_time_policy(chain, {"avoid_peak": ["deepseek"]}, _at(WED, 7))
    assert [hop["model"] for hop in result["chain"]] == [
        "glm-5.3", "gpt-5.6-luna", "mimo-v2.5", "deepseek-v4-pro",
    ]
    assert result["demoted"] == ["deepseek-v4-pro"]
    assert result["promoted"] == []
    assert len(result["chain"]) == len(chain)


def test_avoid_peak_preserves_relative_order_among_the_demoted():
    chain = [
        {"model": "deepseek-v4-pro", "provider": "deepseek"},
        {"model": "gpt-5.6-luna", "provider": "openai-codex"},
        {"model": "deepseek-v4-flash", "provider": "deepseek"},
    ]
    result = apply_time_policy(chain, {"avoid_peak": ["deepseek"]}, _at(WED, 7))
    assert [hop["model"] for hop in result["chain"]] == [
        "gpt-5.6-luna", "deepseek-v4-pro", "deepseek-v4-flash",
    ]


def test_avoid_peak_does_nothing_outside_the_window():
    chain = _mixed_chain()
    result = apply_time_policy(
        chain, {"avoid_peak": ["deepseek", "zai"]}, _at(WED, 12)
    )
    assert [hop["model"] for hop in result["chain"]] == [
        hop["model"] for hop in chain
    ]
    assert result["demoted"] == []


def test_avoid_peak_matches_provider_names_case_insensitively():
    result = apply_time_policy(
        _mixed_chain(), {"avoid_peak": ["  DeepSeek "]}, _at(WED, 7)
    )
    assert result["demoted"] == ["deepseek-v4-pro"]


def test_avoid_peak_ignores_a_provider_that_is_not_in_the_chain():
    result = apply_time_policy(
        _mixed_chain(), {"avoid_peak": ["anthropic"]}, _at(WED, 7)
    )
    assert result["demoted"] == []


@pytest.mark.parametrize("provider", (None, 123, ["zai"]))
def test_avoid_peak_cannot_match_a_hop_that_names_no_provider(provider):
    """``avoid_peak`` matches by PROVIDER NAME, so a hop carrying no usable name
    is left alone — the same rule ``independent_rails`` follows when it refuses to
    count an unattributable hop as a rail.

    Reachable without a malformed registry: ``rules`` builds the primary hop as
    ``{"model": ..., "provider": output.get("provider")}``, which is None for a
    rule that names a model without one. Nothing is silently swallowed — the peak
    is still in the PRICE, which is where ``plan_chain``'s ``multipliers`` reads
    it — but a provider-keyed policy has nothing to key on, so it says nothing
    rather than guessing which rail serves the hop.
    """
    chain = [
        {"model": "glm-5.3", "provider": provider},
        {"model": "kimi-k3", "provider": "moonshot"},
    ]
    result = apply_time_policy(chain, {"avoid_peak": ["zai"]}, _at(MON, 7))
    assert [hop["model"] for hop in result["chain"]] == ["glm-5.3", "kimi-k3"]
    assert result["demoted"] == []
    assert result["peak_priced"] == []
    # The hour really is glm-5.3's peak: the policy declined to act, it was not
    # handed a cheap hour.
    assert price_multiplier("glm-5.3", _at(MON, 7)) == 2.0


def test_prefer_promotes_a_model_that_is_not_in_an_expensive_window():
    result = apply_time_policy(
        _mixed_chain(), {"prefer": ["mimo-v2.5"]}, _at(WED, 7)
    )
    assert [hop["model"] for hop in result["chain"]][0] == "mimo-v2.5"
    assert result["promoted"] == ["mimo-v2.5"]


def test_prefer_does_not_promote_a_model_inside_its_own_expensive_window():
    """Promoting an elo into its own peak would invert the intent."""
    chain = _mixed_chain()
    result = apply_time_policy(chain, {"prefer": ["glm-5.3"]}, _at(WED, 7))
    assert result["promoted"] == []
    assert [hop["model"] for hop in result["chain"]] == [
        hop["model"] for hop in chain
    ]
    # ...and the same policy DOES promote it once the peak is over.
    weekend = apply_time_policy(chain, {"prefer": ["glm-5.3"]}, _at(SAT, 7))
    assert weekend["promoted"] == ["glm-5.3"]
    assert weekend["chain"][0]["model"] == "glm-5.3"


def test_prefer_matches_model_ids_exactly():
    # MiniMax-M3 second, so a match has somewhere to move to: `promoted` reports
    # a MOVE, and a model that never left index 0 could not evidence the match.
    chain = [{"model": "kimi-k3", "provider": "moonshot"},
             {"model": "MiniMax-M3", "provider": "minimax"}]
    wrong_case = apply_time_policy(chain, {"prefer": ["minimax-m3"]}, _at(WED, 7))
    assert wrong_case["promoted"] == []
    assert [hop["model"] for hop in wrong_case["chain"]] == [
        "kimi-k3", "MiniMax-M3",
    ]
    exact = apply_time_policy(chain, {"prefer": ["MiniMax-M3"]}, _at(WED, 7))
    assert exact["promoted"] == ["MiniMax-M3"]
    assert [hop["model"] for hop in exact["chain"]] == ["MiniMax-M3", "kimi-k3"]


def test_time_policy_without_a_clock_is_a_no_op():
    chain = _mixed_chain()
    result = apply_time_policy(
        chain, {"avoid_peak": ["deepseek", "zai"], "prefer": ["mimo-v2.5"]}
    )
    assert [hop["model"] for hop in result["chain"]] == [
        hop["model"] for hop in chain
    ]
    assert result["demoted"] == []
    assert result["promoted"] == []


def test_time_policy_tolerates_junk():
    chain = _mixed_chain()
    for policy in (None, [], "avoid everything", {"avoid_peak": "deepseek"},
                   {"prefer": 3}, {}):
        result = apply_time_policy(chain, policy, _at(WED, 7))
        assert [hop["model"] for hop in result["chain"]] == [
            hop["model"] for hop in chain
        ], policy


def test_time_policy_never_mutates_its_input_list():
    chain = _mixed_chain()
    snapshot = [hop["model"] for hop in chain]
    apply_time_policy(
        chain, {"avoid_peak": ["deepseek", "zai"], "prefer": ["mimo-v2.5"]},
        _at(WED, 7),
    )
    assert [hop["model"] for hop in chain] == snapshot


def test_time_policy_on_an_empty_chain_is_empty():
    assert apply_time_policy([], {"avoid_peak": ["zai"]}, _at(WED, 7)) == {
        "chain": [], "demoted": [], "promoted": [], "peak_priced": [],
    }


def test_both_primary_rails_demoted_at_once_still_leaves_a_chain():
    """The real case the feature exists for: 07:00 UTC on a Wednesday is peak on
    deepseek AND zai simultaneously, so avoid_peak names both — and the chain
    must not empty out. A non-peak elo rises to the front instead.
    """
    chain = _mixed_chain()
    result = apply_time_policy(
        chain, {"avoid_peak": ["deepseek", "zai"]}, _at(WED, 7)
    )
    assert result["chain"], "a cost policy must never empty a chain"
    assert len(result["chain"]) == len(chain)
    assert result["chain"][0]["model"] == "gpt-5.6-luna"
    assert not in_expensive_window(result["chain"][0]["model"], _at(WED, 7))
    assert result["demoted"] == ["deepseek-v4-pro", "glm-5.3"]
    assert [hop["model"] for hop in result["chain"]] == [
        "gpt-5.6-luna", "mimo-v2.5", "deepseek-v4-pro", "glm-5.3",
    ]


def test_a_chain_of_nothing_but_peaking_rails_keeps_every_hop():
    """Every hop matched, so demotion has nowhere to move anything.

    The permutation is the identity, which the report says outright: `demoted` is
    empty because no elo moved, and `peak_priced` still names both because both
    ARE charging more. Reporting them as demoted would describe a reordering that
    never happened.
    """
    chain = [
        {"model": "deepseek-v4-pro", "provider": "deepseek"},
        {"model": "glm-5.3", "provider": "zai"},
    ]
    result = apply_time_policy(
        chain, {"avoid_peak": ["deepseek", "zai"]}, _at(WED, 7)
    )
    assert [hop["model"] for hop in result["chain"]] == [
        "deepseek-v4-pro", "glm-5.3",
    ]
    assert result["demoted"] == []
    assert result["peak_priced"] == ["deepseek-v4-pro", "glm-5.3"]


# ---------------------------------------------------------------------------
# apply_time_cap
# ---------------------------------------------------------------------------

def test_time_cap_excludes_an_over_cap_elo():
    """The DOLLAR rail over the ceiling goes; the plan rail beside it stays.

    glm-5.3's 2.0x is a plan-CREDIT multiplier, so a dollar ceiling has nothing to
    say about it — it is reported as exempt and keeps its slot (see
    test_time_cap_does_not_evict_a_plan_credit_rail for the full argument).
    """
    chain = _mixed_chain()
    result = apply_time_cap(chain, 1.5, _at(WED, 7))
    assert [hop["model"] for hop in result["chain"]] == [
        "glm-5.3", "gpt-5.6-luna", "mimo-v2.5",
    ]
    assert result["capped"] == [
        {"model": "deepseek-v4-pro", "multiplier": 2.0},
    ]
    assert result["cap_exempt"] == [
        {"model": "glm-5.3", "multiplier": 2.0, "billing_mode": "plan"},
    ]
    assert result["bypassed"] is False


def test_time_cap_is_a_ceiling_not_a_strict_bound():
    result = apply_time_cap(_mixed_chain(), 2.0, _at(WED, 7))
    assert [hop["model"] for hop in result["chain"]] == [
        "deepseek-v4-pro", "glm-5.3", "gpt-5.6-luna", "mimo-v2.5",
    ]
    assert result["capped"] == []
    # Nothing is over the ceiling, so nothing needed exempting: `cap_exempt`
    # reports an exemption that MATTERED, not every plan rail in the chain.
    assert result["cap_exempt"] == []


def test_time_cap_bypasses_rather_than_emptying_the_chain():
    """A cost control must never be able to cause an outage."""
    chain = [
        {"model": "deepseek-v4-pro", "provider": "deepseek"},
        {"model": "deepseek-v4-flash", "provider": "deepseek"},
    ]
    result = apply_time_cap(chain, 1.5, _at(WED, 7))
    assert result["bypassed"] is True
    assert [hop["model"] for hop in result["chain"]] == [
        "deepseek-v4-pro", "deepseek-v4-flash",
    ]
    # Diagnostics are RETAINED on the bypass path, same as the capability filter.
    assert result["capped"] == [
        {"model": "deepseek-v4-pro", "multiplier": 2.0},
        {"model": "deepseek-v4-flash", "multiplier": 2.0},
    ]
    assert result["cap_exempt"] == []


def test_time_cap_without_a_clock_is_no_cap():
    chain = _mixed_chain()
    result = apply_time_cap(chain, 1.0)
    assert [hop["model"] for hop in result["chain"]] == [
        hop["model"] for hop in chain
    ]
    assert result["capped"] == []
    assert result["bypassed"] is False


def test_an_absent_or_junk_max_multiplier_is_no_cap():
    chain = _mixed_chain()
    for cap in (None, "loads", True, [2.0]):
        result = apply_time_cap(chain, cap, _at(WED, 7))
        assert [hop["model"] for hop in result["chain"]] == [
            hop["model"] for hop in chain
        ], cap
        assert result["capped"] == [], cap


def test_time_cap_never_touches_a_cheap_window():
    result = apply_time_cap(
        [{"model": "mimo-v2.5", "provider": "xiaomi"},
         {"model": "kimi-k3", "provider": "moonshot"}],
        1.0, _at(WED, 20),
    )
    assert [hop["model"] for hop in result["chain"]] == [
        "mimo-v2.5", "kimi-k3",
    ]
    assert result["capped"] == []


def test_time_cap_never_mutates_its_input_list():
    chain = _mixed_chain()
    snapshot = [hop["model"] for hop in chain]
    apply_time_cap(chain, 1.5, _at(WED, 7))
    assert [hop["model"] for hop in chain] == snapshot


def test_time_cap_on_an_empty_chain_is_empty():
    assert apply_time_cap([], 1.5, _at(WED, 7)) == {
        "chain": [], "capped": [], "cap_exempt": [], "bypassed": False,
    }


def test_time_cap_keeps_a_hop_it_cannot_price():
    result = apply_time_cap(
        [{"model": "mystery-elo", "provider": "somewhere"}, "junk"],
        1.0, _at(WED, 7),
    )
    assert len(result["chain"]) == 2
    assert result["capped"] == []


# ---------------------------------------------------------------------------
# registry / window diagnostics  (F14: this must be callable from lint)
# ---------------------------------------------------------------------------

def test_every_diagnostic_is_shaped_for_lint_to_append_verbatim(monkeypatch):
    monkeypatch.setitem(
        caps_module.MODEL_CAPABILITIES, "broken-elo", {"provider": "nowhere"}
    )
    problems = registry_diagnostics()
    assert problems
    assert all(problem.startswith("model '") for problem in problems)
    assert any(
        problem.startswith("model 'broken-elo': ") for problem in problems
    )


def test_registry_diagnostics_reports_a_bad_window_without_raising(monkeypatch):
    monkeypatch.setitem(
        caps_module.MODEL_CAPABILITIES,
        "wrapping-elo",
        {
            "provider": "nowhere", "context_window": 1000,
            "billing_mode": "metered", "vision": False,
            "tool_calling": True, "structured_output": True,
            "price_windows": [{"hours_utc": [22, 3], "multiplier": 2.0}],
        },
    )
    problems = registry_diagnostics()
    assert any("'hours_utc'" in problem for problem in problems)


def test_an_entry_that_is_not_a_mapping_is_one_diagnostic_not_a_crash(monkeypatch):
    """A registry entry hand-edited down to a bare value gets ONE message.

    "Diagnostics, never exceptions" is the contract, and reading fields off a
    string is how that gets broken: ``entry.get`` raises on it, and treating
    ``set("metered")`` as a field list would bury the real defect under one
    "unrecognized field" per letter. So the entry is reported and skipped.
    """
    monkeypatch.setitem(caps_module.MODEL_CAPABILITIES, "typo-elo", "metered")
    problems = registry_diagnostics()
    assert [
        problem for problem in problems if "'typo-elo'" in problem
    ] == ["model 'typo-elo': entry is not a mapping"]


def test_an_unknown_billing_mode_names_the_mode_it_found(monkeypatch):
    """``billing_mode`` is the UNIT every price comparison keys on.

    A mode outside :data:`BILLING_MODES` reaches ``cheapest_now`` as the
    undescribable bucket and steps the time cap aside — quietly, and correctly,
    because there is nothing else it could do with a unit it cannot read. Lint is
    where that gets caught, so the message has to name the value to fix.
    """
    monkeypatch.setitem(
        caps_module.MODEL_CAPABILITIES,
        "yearly-elo",
        {
            "provider": "nowhere", "context_window": 1000,
            "billing_mode": "yearly", "vision": False,
            "tool_calling": True, "structured_output": True,
        },
    )
    assert "yearly" not in BILLING_MODES
    assert [
        problem for problem in registry_diagnostics() if "'yearly-elo'" in problem
    ] == ["model 'yearly-elo': unknown billing_mode 'yearly'"]


def test_an_unrecognized_field_is_a_diagnostic_because_nothing_reads_it(
    monkeypatch,
):
    """A misspelt field is a value no consumer will ever read, and lint is the
    only place that difference is visible.

    Asserted through the price rather than on the message alone: ``price_input``
    makes an entry LOOK priced in the file while ``effective_price`` still reports
    None, and a rail whose cost cannot be read is a rail ``cheapest_now`` sorts
    last. The name of the field is the whole fix, so the message carries it.
    """
    monkeypatch.setitem(
        caps_module.MODEL_CAPABILITIES,
        "misspelt-elo",
        {
            "provider": "nowhere", "context_window": 1000,
            "billing_mode": "metered", "vision": False,
            "tool_calling": True, "structured_output": True,
            "price_input": 1.0, "price_output": 2.0,
        },
    )
    assert [
        problem for problem in registry_diagnostics() if "'misspelt-elo'" in problem
    ] == [
        "model 'misspelt-elo': unrecognized field 'price_input'",
        "model 'misspelt-elo': unrecognized field 'price_output'",
    ]
    assert effective_price("misspelt-elo", _at(WED, 7)) is None


def test_price_window_diagnostics_accepts_absent_windows():
    assert price_window_diagnostics("kimi-k3", None) == []


def test_price_window_diagnostics_accepts_every_shipped_window():
    for model, entry in MODEL_CAPABILITIES.items():
        assert price_window_diagnostics(
            model, entry.get("price_windows")
        ) == [], model


def test_price_window_diagnostics_rejects_a_midnight_crossing_window():
    problems = price_window_diagnostics(
        "x", [{"hours_utc": [22, 3], "multiplier": 2.0}]
    )
    assert len(problems) == 1
    assert "two entries" in problems[0]


def test_price_window_diagnostics_rejects_out_of_range_hours():
    assert price_window_diagnostics(
        "x", [{"hours_utc": [6, 25], "multiplier": 2.0}]
    )
    assert price_window_diagnostics(
        "x", [{"hours_utc": [-1, 6], "multiplier": 2.0}]
    )
    # 24 is the legal midnight-exclusive end.
    assert price_window_diagnostics(
        "x", [{"hours_utc": [16, 24], "multiplier": 0.8}]
    ) == []


@pytest.mark.parametrize(
    "hours",
    ([-1, 6], [6, 25], [22, 3], [0, 0], [16.5, 24], [6, 9.5], "6-10", [6], None),
)
def test_a_malformed_window_reports_the_value_that_actually_offended(hours):
    """The message names what the operator WROTE, never an example of it.

    One message has to serve a negative start, an end past midnight, a reversed
    pair and a fractional hour, so it states the rule for all of them — but it
    used to illustrate the rule with a hardcoded "16.5 is not an hour boundary",
    which sent the operator who typed `[-1, 6]` hunting for a fractional hour
    that appeared nowhere in their config. A diagnostic naming a value nobody
    wrote costs more than it saves.
    """
    window = {"hours_utc": hours, "multiplier": 2.0}
    problems = price_window_diagnostics("kimi-k3", [window])
    assert len(problems) == 1
    assert repr(hours) in problems[0], problems[0]
    # The rule the value broke is still spelled out...
    assert f"0 <= start < end <= {caps_module._HOURS_IN_DAY}" in problems[0]
    # ...and no value the operator did not write is.
    if "16.5" not in repr(hours):
        assert "16.5" not in problems[0]


def test_price_window_diagnostics_reports_overlapping_windows():
    problems = price_window_diagnostics(
        "x",
        [{"hours_utc": [6, 10], "multiplier": 2.0},
         {"hours_utc": [9, 12], "multiplier": 3.0}],
    )
    assert problems == ["model 'x': price_windows entries overlap"]


def test_adjacent_half_open_windows_do_not_overlap():
    assert price_window_diagnostics(
        "x",
        [{"hours_utc": [1, 4], "multiplier": 2.0},
         {"hours_utc": [4, 6], "multiplier": 3.0}],
    ) == []


def test_windows_on_disjoint_weekdays_do_not_overlap():
    assert price_window_diagnostics(
        "x",
        [{"hours_utc": [6, 10], "weekdays": [0, 1, 2, 3, 4], "multiplier": 2.0},
         {"hours_utc": [6, 10], "weekdays": [5, 6], "multiplier": 0.5}],
    ) == []


def test_price_window_diagnostics_rejects_bad_weekdays_and_multipliers():
    assert price_window_diagnostics(
        "x", [{"hours_utc": [6, 10], "weekdays": [7], "multiplier": 2.0}]
    )
    assert price_window_diagnostics(
        "x", [{"hours_utc": [6, 10], "weekdays": "monday", "multiplier": 2.0}]
    )
    assert price_window_diagnostics(
        "x", [{"hours_utc": [6, 10], "multiplier": 0}]
    )
    assert price_window_diagnostics(
        "x", [{"hours_utc": [6, 10], "multiplier": "double"}]
    )


def test_price_window_diagnostics_rejects_a_malformed_container():
    assert price_window_diagnostics("x", {"hours_utc": [6, 10]}) == [
        "model 'x': price_windows must be a list"
    ]
    assert price_window_diagnostics("x", ["06:00-10:00"])


def test_a_weekday_gated_window_is_inert_when_the_gate_is_malformed():
    declared = {
        "price_windows": [
            {"hours_utc": [6, 10], "weekdays": ["monday"], "multiplier": 2.0}
        ]
    }
    assert price_multiplier("kimi-k3", _at(MON, 7), declared) == 1.0


# ---------------------------------------------------------------------------
# whole-hour window shapes: lint and the running path read one rule
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hours", ([16.5, 24], [6, 9.5], [0.25, 6], [6, 10.75]))
def test_a_fractional_hour_is_a_diagnostic_and_an_inert_window(hours):
    """A window that lints CLEAN and then runs as different hours is the worst
    of the two outcomes, so a fractional hour is refused like every other
    malformed shape.

    Asserted on BOTH sides at once, and that is the point of the test: the same
    window must be reported by lint AND absent from the price the router charges.
    `[16.5, 24]` used to truncate to `[16, 24)`, starting half an hour early with
    nothing said about it; `[6, 9.5]` used to end at 9, silently excluding the
    hour the operator meant to include.
    """
    window = {"hours_utc": hours, "multiplier": 2.0}
    problems = price_window_diagnostics("kimi-k3", [window])
    assert len(problems) == 1
    assert "'hours_utc'" in problems[0]

    # ...and the running path agrees: the window is absent, not reinterpreted.
    declared = {"price_windows": [window]}
    for hour in range(24):
        assert price_multiplier("kimi-k3", _at(WED, hour), declared) == 1.0, hour
    assert next_window_change("kimi-k3", _at(WED, 7), declared) is None


def test_a_whole_hour_written_as_a_float_is_still_that_hour():
    """YAML decides `16.0` is a float; the operator still wrote hour 16.

    Nothing is lost truncating it, so it is accepted — the rule is "whole hour",
    not "int-typed".
    """
    window = {"hours_utc": [16.0, 24], "multiplier": 2.0}
    assert price_window_diagnostics("kimi-k3", [window]) == []
    declared = {"price_windows": [window]}
    assert price_multiplier("kimi-k3", _at(WED, 15), declared) == 1.0
    assert price_multiplier("kimi-k3", _at(WED, 16), declared) == 2.0


@pytest.mark.parametrize("weekdays", ([0, 1.5], [True], [0, None]))
def test_a_weekday_that_is_not_a_whole_number_is_a_diagnostic_and_an_inert_gate(
    weekdays,
):
    """`[0, 1.5]` is a typo, not Monday and Tuesday — same rule as the hours."""
    window = {"hours_utc": [6, 10], "weekdays": weekdays, "multiplier": 2.0}
    problems = price_window_diagnostics("kimi-k3", [window])
    assert len(problems) == 1
    assert "'weekdays'" in problems[0]
    declared = {"price_windows": [window]}
    for day in (MON, WED, SAT):
        assert price_multiplier("kimi-k3", _at(day, 7), declared) == 1.0, day


@pytest.mark.parametrize(
    "hours",
    (
        [float("inf"), 24],
        [0, float("inf")],
        [float("nan"), 6],
        ["16.5", 24],
        [True, 10],
        [None, 10],
        [[6], 10],
    ),
)
def test_an_hour_that_is_not_a_whole_number_is_a_diagnostic_not_an_exception(hours):
    """Whatever the shape, a diagnostic — and never an exception out of lint.

    `int(float("inf"))` RAISES OverflowError, so the whole-hour rule closes a
    crash as well as a silently rewritten window; `True` is not an hour, and the
    other shapes are the ordinary junk a hand-edited YAML file produces.
    """
    window = {"hours_utc": hours, "multiplier": 2.0}
    assert price_window_diagnostics("kimi-k3", [window])
    assert price_multiplier("kimi-k3", _at(WED, 7), {"price_windows": [window]}) == 1.0


@pytest.mark.parametrize(
    "hours", ("6-10", [6], [6, 10, 14], (), {"start": 6, "end": 10})
)
def test_hours_utc_that_is_not_a_pair_is_a_diagnostic_and_an_inert_window(hours):
    """The shape is a two-element sequence or it is nothing — same both sides."""
    window = {"hours_utc": hours, "multiplier": 2.0}
    assert price_window_diagnostics("kimi-k3", [window])
    assert price_multiplier("kimi-k3", _at(WED, 7), {"price_windows": [window]}) == 1.0


def test_a_window_with_an_unusable_multiplier_prices_at_the_base_rate():
    """Lint reports it and the running path ignores it — one reading, again."""
    for multiplier in (0, -1.0, "double", None, float("nan"), float("inf"),
                       float("-inf"), "nan", "inf"):
        windows = [{"hours_utc": [6, 10], "multiplier": multiplier}]
        assert price_window_diagnostics("kimi-k3", windows), multiplier
        assert price_multiplier(
            "kimi-k3", _at(WED, 7), {"price_windows": windows}
        ) == 1.0, multiplier


# ---------------------------------------------------------------------------
# a NON-FINITE number is a diagnostic, never a value
# ---------------------------------------------------------------------------

# `.inf`/`.nan` are legal YAML, so an operator can put either on any numeric key
# in router.yaml. Both defeat the arithmetic in a way no other junk does: int()
# RAISES on them, and every comparison against nan is False.
_NON_FINITE = (float("inf"), float("-inf"), float("nan"), "inf", "-inf", "nan",
               " Infinity ", "NaN")


@pytest.mark.parametrize("value", _NON_FINITE)
def test_a_non_finite_number_never_becomes_a_number(value):
    """Both coercions refuse it, on both routes into them.

    `_as_int` used to RAISE here — OverflowError on ±inf, ValueError on nan — and
    it sits under `satisfies` and `derive_requirements`, i.e. the request path,
    where `rules.plan_chain`'s defensive except does not catch OverflowError.
    `_as_float` used to PROPAGATE it, which is worse than raising: nan compares
    False against everything, so the value survives every guard silently.
    """
    assert caps_module._as_int(value) is None, value
    assert caps_module._as_float(value) is None, value
    assert caps_module._as_whole_number(value) is None, value


@pytest.mark.parametrize("value", _NON_FINITE)
def test_a_non_finite_declared_window_is_a_flagged_unknown_not_a_crash(value):
    """The reviewer's reproduction: `context_window: .inf` on a tier hop.

    It linted CLEAN and then took the whole routing decision down. Two invariants
    broke at once — the write gate accepted a config that misroutes, and a
    capability filter broke routing outright — so the fix has to satisfy both: no
    exception, AND the hop is reported as unverifiable rather than silently
    trusted.
    """
    hop = {"model": "house-model", "provider": "local-rail", "context_window": value}
    assert satisfies("house-model", {"min_context": 5_000}, hop) == (
        True, "capability_unknown",
    )
    result = filter_chain([hop], {"min_context": 5_000})
    assert result["unknown"] == ["house-model"], value
    assert result["bypassed"] is False
    assert result["eligible"] == [hop]
    # Same for the input bound, which is the other half of `_input_ceiling`.
    both = dict(hop, context_window=500_000, max_input_tokens=value)
    assert satisfies("house-model", {"min_context": 5_000}, both) == (True, "")
    assert caps_module._input_ceiling(both) == 500_000


@pytest.mark.parametrize("value", _NON_FINITE)
def test_a_non_finite_token_estimate_is_discarded_not_raised(value):
    """`derive_requirements` runs per request; it may not raise on junk input."""
    assert derive_requirements({"est_input_tokens": value}) == {}
    assert derive_requirements({}, {"min_context": value}) == {}
    # A usable estimate beside the junk floor still survives.
    assert derive_requirements(
        {"est_input_tokens": 1_000}, {"min_context": value}
    ) == {"min_context": 1_250}


def test_the_whole_time_layer_survives_a_non_finite_multiplier():
    """nan is the one value that passes a `> ceiling` guard by being unordered.

    A nan multiplier used to lint clean (`nan <= 0` is False) and then make the
    elo permanently un-capped, un-peak-priced and unorderable — a routing change
    with no diagnostic anywhere. Now it is one diagnostic that lint and every
    running stage read the same way.
    """
    hop = {
        "model": "house-model", "provider": "local-rail",
        "billing_mode": "metered", "context_window": 500_000,
        "price_in": 1.0, "price_out": 2.0,
        "price_windows": [{"hours_utc": [6, 10], "multiplier": float("nan")}],
    }
    assert price_window_diagnostics("house-model", hop["price_windows"])
    assert price_multiplier("house-model", _at(WED, 7), hop) == 1.0
    assert in_expensive_window("house-model", _at(WED, 7), hop) is False
    assert effective_price("house-model", _at(WED, 7), hop) == (1.0, 2.0)
    capped = apply_time_cap([hop], 1.5, _at(WED, 7))
    assert (capped["capped"], capped["cap_exempt"]) == ([], [])
    assert capped["chain"] == [hop]


def test_a_price_declared_without_any_capability_stays_unknown():
    """One gate, not two: F4's rule decides pricing visibility as well."""
    declared = {"price_in": 1.0, "price_out": 2.0, "billing_mode": "metered"}
    assert effective_price("house-model", _at(WED, 7), declared) is None
    # Declaring one real capability makes the declared price usable.
    with_capability = dict(declared, context_window=500_000)
    assert effective_price(
        "house-model", _at(WED, 7), with_capability
    ) == (1.0, 2.0)


def test_avoid_peak_is_per_elo_not_per_provider():
    """A same-provider elo with FLAT pricing costs no more, so it is not demoted.

    glm-5.3 bills plan credits at 2x during the weekday peak; metered glm-4.6 is
    flat at every hour. Demoting glm-4.6 would degrade the route and save nothing.
    """
    chain = [
        {"model": "glm-5.3", "provider": "zai"},
        {"model": "glm-4.6", "provider": "zai"},
        {"model": "gpt-5.6-luna", "provider": "openai-codex"},
    ]
    result = apply_time_policy(chain, {"avoid_peak": ["zai"]}, _at(WED, 7))
    assert result["demoted"] == ["glm-5.3"]
    assert [hop["model"] for hop in result["chain"]] == [
        "glm-4.6", "gpt-5.6-luna", "glm-5.3",
    ]


def test_cheapest_now_sorts_an_undescribable_elo_last_without_dropping_it():
    """Ordering is not eligibility: an elo we cannot price still gets a slot."""
    chain = [
        {"model": "ghost-model", "provider": "other-rail",
         "billing_mode": "plan"},
        {"model": "kimi-k3", "provider": "moonshot"},
    ]
    ordered = order_chain(chain, "cheapest_now", pin_primary=False,
                          when=_at(WED, 12))
    assert [hop["model"] for hop in ordered] == ["kimi-k3", "ghost-model"]


def test_time_cap_bypasses_when_only_unnamed_hops_would_survive():
    """A chain of hops nothing can name is not a route, so the cap gives way."""
    result = apply_time_cap(
        [{"model": "deepseek-v4-pro", "provider": "deepseek"},
         {"provider": "deepseek"}],
        1.0, _at(WED, 7),
    )
    assert result["bypassed"] is True
    assert [hop.get("model") for hop in result["chain"]] == [
        "deepseek-v4-pro", None,
    ]
    assert result["capped"] == [{"model": "deepseek-v4-pro", "multiplier": 2.0}]
    assert result["cap_exempt"] == []


# ---------------------------------------------------------------------------
# a `time_cap` is a DOLLAR ceiling: one model of the unit across the module
# ---------------------------------------------------------------------------

def _t1_chain():
    """The shipped T1 chain: plan primary, subscription seat, metered third rail.

    Kept inline rather than read out of router.yaml — this is a unit test of the
    rule, and the tier is here only because it is the shape the rule got wrong.
    """
    return [
        {"model": "glm-5.3-flash", "provider": "zai", "billing_mode": "plan"},
        {"model": "gpt-5.6-luna", "provider": "openai-codex",
         "billing_mode": "subscription"},
        {"model": "mimo-v2.5", "provider": "xiaomi", "billing_mode": "metered"},
    ]


def test_time_cap_does_not_evict_a_plan_credit_rail():
    """T1's shipped shape at 07:00 UTC on a weekday — the case the cap got wrong.

    glm-5.3-flash's 2.0x is a PLAN-CREDIT multiplier: it doubles a draw against an
    allowance already bought and adds nothing to any dollar invoice, so
    `max_multiplier: 1.5` — a dollar ceiling — has nothing to say about it.
    Evicting it pushed every trivial mechanical edit in that four-hour block onto
    a metered rail to avoid a cost that is already sunk, which is the trade
    `cheapest_now`'s billing buckets refuse one function over.
    """
    result = apply_time_cap(_t1_chain(), 1.5, _at(WED, 7))
    assert [hop["model"] for hop in result["chain"]] == [
        "glm-5.3-flash", "gpt-5.6-luna", "mimo-v2.5",
    ]
    assert result["capped"] == []
    assert result["bypassed"] is False
    # Silence would be the other half of the bug: the exemption is REPORTED, with
    # the multiplier the operator's 1.5 was compared against.
    assert result["cap_exempt"] == [
        {"model": "glm-5.3-flash", "multiplier": 2.0, "billing_mode": "plan"},
    ]


def test_the_same_cap_still_removes_a_dollar_rail_beside_the_plan_one():
    """What a `time_cap` still DOES on a plan-primary tier, in one assertion."""
    chain = [
        {"model": "glm-5.3-flash", "provider": "zai", "billing_mode": "plan"},
        {"model": "deepseek-v4-flash", "provider": "deepseek",
         "billing_mode": "metered"},
    ]
    result = apply_time_cap(chain, 1.5, _at(WED, 7))
    assert [hop["model"] for hop in result["chain"]] == ["glm-5.3-flash"]
    assert [c["model"] for c in result["capped"]] == ["deepseek-v4-flash"]
    assert [e["model"] for e in result["cap_exempt"]] == ["glm-5.3-flash"]
    assert result["bypassed"] is False


def test_the_shipped_t1_cap_removes_nothing_at_any_hour():
    """`apply_time_cap`'s prediction for T1, pinned so the docstring cannot rot.

    With glm-5.3-flash exempt on its unit, T1's roster has nothing left that can
    exceed 1.5 at any hour: gpt-5.6-luna is flat and mimo-v2.5's only window is
    xiaomi's 0.8x DISCOUNT. So the cap removes nothing today — it is insurance
    against a future dollar-priced hop — and the one thing it shows an operator is
    the plan primary's weekday credit peak, in `cap_exempt`.
    """
    for day in (MON, WED, FRI, SAT, SUN):
        for hour in range(24):
            result = apply_time_cap(_t1_chain(), 1.5, _at(day, hour))
            label = (day, hour)
            assert result["capped"] == [], label
            assert result["bypassed"] is False, label
            assert [hop["model"] for hop in result["chain"]] == [
                hop["model"] for hop in _t1_chain()
            ], label
            in_zai_peak = day in (MON, WED, FRI) and 6 <= hour < 10
            assert [e["model"] for e in result["cap_exempt"]] == (
                ["glm-5.3-flash"] if in_zai_peak else []
            ), label


@pytest.mark.parametrize("mode", sorted(BILLING_MODES) + ["not-a-mode"])
def test_the_cap_and_the_cheapest_now_bucket_share_one_model_of_the_unit(mode):
    """`_BILLING_RANK` decides the bucket AND whether a dollar cap may speak.

    Two readings of "what unit does this rail bill in" would be two answers, and
    a chain ordered on one model of cost and filtered on another is the defect
    this test exists to prevent. So the assertion is driven off the same table the
    ordering uses, not off a list of modes copied into the test.
    """
    hop = {
        "model": "house-model", "provider": "somewhere",
        "context_window": 200_000, "billing_mode": mode,
        "price_in": 1.0, "price_out": 2.0,
        "price_windows": [{"hours_utc": [6, 10], "multiplier": 3.0}],
    }
    flat = {"model": "kimi-k3", "provider": "moonshot"}
    result = apply_time_cap([hop, flat], 1.5, _at(WED, 7))

    bills_dollars = (
        caps_module._BILLING_RANK.get(mode) == caps_module._BUCKET_DOLLARS
    )
    if bills_dollars:
        assert [c["model"] for c in result["capped"]] == ["house-model"]
        assert result["cap_exempt"] == []
        assert [h["model"] for h in result["chain"]] == ["kimi-k3"]
    else:
        assert result["capped"] == []
        assert [h["model"] for h in result["chain"]] == [
            "house-model", "kimi-k3",
        ]
        expected_mode = mode if mode in BILLING_MODES else "unknown"
        assert result["cap_exempt"] == [
            {"model": "house-model", "multiplier": 3.0,
             "billing_mode": expected_mode},
        ]
    assert result["bypassed"] is False


def test_an_all_plan_chain_at_peak_never_needs_the_bypass():
    """Unit-awareness can only ADD survivors, so it makes the bypass rarer."""
    chain = [
        {"model": "glm-5.3", "provider": "zai"},
        {"model": "glm-5.3-flash", "provider": "zai"},
    ]
    result = apply_time_cap(chain, 1.0, _at(WED, 7))
    assert result["bypassed"] is False
    assert result["capped"] == []
    assert [hop["model"] for hop in result["chain"]] == [
        "glm-5.3", "glm-5.3-flash",
    ]
    assert [e["model"] for e in result["cap_exempt"]] == [
        "glm-5.3", "glm-5.3-flash",
    ]
    assert all(e["billing_mode"] == "plan" for e in result["cap_exempt"])


def test_a_cap_cannot_drop_an_elo_whose_billing_mode_nothing_can_describe():
    """Unknown fails OPEN and says so: dropping it could empty a chain."""
    result = apply_time_cap(
        [{"model": "mystery-elo", "provider": "nowhere"},
         {"model": "kimi-k3", "provider": "moonshot"}],
        0.5, _at(WED, 12),
    )
    assert [hop["model"] for hop in result["chain"]] == ["mystery-elo"]
    assert [c["model"] for c in result["capped"]] == ["kimi-k3"]
    assert result["cap_exempt"] == [
        {"model": "mystery-elo", "multiplier": 1.0, "billing_mode": "unknown"},
    ]


# The hours that matter on the shipped registry: both primary rails peaking,
# deepseek alone, flat, xiaomi's discount, the weekend (zai off, deepseek on),
# and no clock at all.
_TIME_MATRIX = (
    _at(WED, 7), _at(WED, 2), _at(WED, 12), _at(WED, 20), _at(SAT, 7), None,
)


def _cap_matrix_chains():
    return (
        _t1_chain(),
        _mixed_chain(),
        # every hop metered and peaking at once — the bypass path
        [{"model": "deepseek-v4-pro", "provider": "deepseek"},
         {"model": "deepseek-v4-flash", "provider": "deepseek"}],
        # one dollar rail, one credit rail, both peaking
        [{"model": "deepseek-v4-pro", "provider": "deepseek"},
         {"model": "glm-5.3", "provider": "zai"}],
        # a hop nothing can name beside one nothing can describe
        [{"provider": "zai"}, {"model": "mystery-elo", "provider": "nowhere"}],
    )


def test_the_cap_report_and_the_chain_it_returns_never_disagree():
    """The invariant, asserted as an AGREEMENT rather than from one side.

    `capped` is what the console renders as removed and `cap_exempt` as kept, so
    each has to match the chain that will actually be attempted — at every hour,
    for every ceiling, including the bypass path where `capped` is retained as a
    diagnostic and nothing is removed after all.
    """
    for chain in _cap_matrix_chains():
        declared_of = {
            hop["model"]: hop for hop in chain if hop.get("model")
        }
        named = set(declared_of)
        for cap in (0.5, 1.0, 1.5, 2.0, 3.0, None, "junk"):
            for when in _TIME_MATRIX:
                result = apply_time_cap(chain, cap, when)
                label = (cap, when, sorted(named))
                kept = [
                    hop["model"] for hop in result["chain"] if hop.get("model")
                ]
                capped = [entry["model"] for entry in result["capped"]]
                exempt = [entry["model"] for entry in result["cap_exempt"]]

                # A cost control must not be able to cause an outage.
                assert result["chain"], label
                # One elo, one verdict.
                assert not set(capped) & set(exempt), label
                # Nothing leaves the chain unaccounted for.
                assert named == set(kept) | set(capped), label
                # An exempt elo is still attemptable — that is what exempt means.
                assert set(exempt) <= set(kept), label

                if result["bypassed"]:
                    assert result["chain"] == list(chain), label
                    assert capped, label
                else:
                    # A capped elo is GONE from the chain that will run.
                    assert not set(capped) & set(kept), label

                ceiling = caps_module._as_float(cap)
                if when is None or ceiling is None:
                    assert (capped, exempt) == ([], []), label
                    assert result["chain"] == list(chain), label
                    continue

                # Every reported multiplier is the one the registry answers with,
                # every verdict is over the ceiling, and the unit decides which
                # list it landed in.
                for entry in result["capped"] + result["cap_exempt"]:
                    model = entry["model"]
                    assert entry["multiplier"] == price_multiplier(
                        model, when, declared_of[model]
                    ), (label, model)
                    assert entry["multiplier"] > ceiling, (label, model)
                for entry in result["capped"]:
                    assert caps_module._billing_mode_of(
                        entry["model"], declared_of[entry["model"]]
                    ) in ("metered", "subscription"), (label, entry)
                for entry in result["cap_exempt"]:
                    assert entry["billing_mode"] not in (
                        "metered", "subscription"
                    ), (label, entry)


# ---------------------------------------------------------------------------
# `demoted` is a MOVE, `peak_priced` is a PRICE
# ---------------------------------------------------------------------------

def test_the_shipped_t3_shape_reports_the_price_and_not_a_move():
    """T3/T4: both named providers are ALREADY the trailing hops.

    Demotion preserves relative order, so the permutation is the identity. The
    old report named both elos as demoted at 07:00 UTC while the chain was
    byte-identical to 15:00 UTC — the console renders that as "moved to the end",
    a claim about an order nothing changed. `peak_priced` says the true thing
    (they are charging double) and `demoted` stays empty because nothing moved.
    """
    chain = [
        {"model": "gpt-5.6-terra", "provider": "openai-codex"},
        {"model": "deepseek-v4-pro", "provider": "deepseek"},
        {"model": "glm-5.3", "provider": "zai"},
    ]
    result = apply_time_policy(
        chain, {"avoid_peak": ["deepseek", "zai"]}, _at(WED, 7)
    )
    assert [hop["model"] for hop in result["chain"]] == [
        hop["model"] for hop in chain
    ]
    assert result["demoted"] == []
    assert result["promoted"] == []
    assert result["peak_priced"] == ["deepseek-v4-pro", "glm-5.3"]

    # ...and the fix router.yaml's own T3 note recommends — a flat-priced hop
    # behind the pair — makes the same policy report a real move.
    with_flat = chain + [{"model": "mimo-v2.5", "provider": "xiaomi"}]
    moved = apply_time_policy(
        with_flat, {"avoid_peak": ["deepseek", "zai"]}, _at(WED, 7)
    )
    assert [hop["model"] for hop in moved["chain"]] == [
        "gpt-5.6-terra", "mimo-v2.5", "deepseek-v4-pro", "glm-5.3",
    ]
    assert moved["demoted"] == ["deepseek-v4-pro", "glm-5.3"]
    assert moved["peak_priced"] == ["deepseek-v4-pro", "glm-5.3"]


_HOURS_IN_WEEK = 168


def _week() -> list:
    """Every hour of the reference week, Mon 00:00Z .. Sun 23:00Z (168 of them).

    A time-dependent claim sampled at one or two hours is the defect this file
    keeps catching: 07:00Z and 15:00Z between them miss every hour where T3's
    `avoid_peak` actually reorders, because both zai and deepseek peak at 07:00
    on a weekday and neither peaks at 15:00. The whole week is 168 cheap calls.
    """
    return [
        datetime(2026, 8, MON, tzinfo=UTC) + timedelta(hours=step)
        for step in range(_HOURS_IN_WEEK)
    ]


def test_the_shipped_t3_policy_reorders_for_15_hours_of_the_week():
    """The claim router.yaml used to make — "REORDERS NOTHING", "the permutation
    is the IDENTITY", verified at 07:00Z and 15:00Z — swept across the week.

    It is wrong for 15 of 168 hours (8.93%), and what changes is which model
    serves the SECOND hop of every hard task: metered deepseek-v4-pro out,
    plan-billed glm-5.3 in. The identity holds only where BOTH named providers
    peak together, which is the four weekday hours 07:00Z happens to land in.

    The two lists are read off the returned chain here, not off the match, so this
    pins the agreement the policy comment got wrong rather than one side of it.
    """
    chain = [
        {"model": "gpt-5.6-terra", "provider": "openai-codex"},
        {"model": "deepseek-v4-pro", "provider": "deepseek"},
        {"model": "glm-5.3", "provider": "zai"},
    ]
    declared = [hop["model"] for hop in chain]
    seen: dict = {}
    for when in _week():
        result = apply_time_policy(chain, {"avoid_peak": ["deepseek", "zai"]}, when)
        key = (
            tuple(hop["model"] for hop in result["chain"]),
            tuple(result["demoted"]),
            tuple(result["peak_priced"]),
        )
        seen.setdefault(key, []).append((when.weekday(), when.hour))

    # Both vendors peak MON-FRI only (deepseek narrowed to weekdays on
    # 2026-08-22), so the pair separates exactly where deepseek peaks alone:
    # 01:00-04:00 Mon-Fri, 15 hours. The weekend is quiet for both.
    reordered = (
        ("gpt-5.6-terra", "glm-5.3", "deepseek-v4-pro"),
        ("deepseek-v4-pro",),
        ("deepseek-v4-pro",),
    )
    both_peaking = (tuple(declared), (), ("deepseek-v4-pro", "glm-5.3"))
    quiet = (tuple(declared), (), ())
    assert set(seen) == {reordered, both_peaking, quiet}
    assert len(seen[reordered]) == 15
    assert len(seen[both_peaking]) == 20
    assert len(seen[quiet]) == 133
    assert sorted(seen[reordered]) == sorted(
        [(day, hour) for day in range(5) for hour in (1, 2, 3)]
    )
    assert sorted(seen[both_peaking]) == sorted(
        [(day, hour) for day in range(5) for hour in (6, 7, 8, 9)]
    )
    # The identity IS what 07:00Z on a weekday reports — the comment's sample was
    # true about its hour and false about the week.
    assert (_at(MON, 7).weekday(), 7) in seen[both_peaking]
    # Saturday 07:00 used to be deepseek peaking alone, which is why the old
    # sample landed in `reordered`. Since 2026-08-22 the vendor bills the whole
    # weekend off-peak, so the same hour is now quiet for both.
    assert (_at(SAT, 7).weekday(), 7) in seen[quiet]


def test_the_shipped_t2_tail_flips_on_deepseeks_window_alone():
    """T2's `cheapest_now` note framed the flip on the deepseek/zai OVERLAP.

    It cannot be that: the zai plan rail is the pinned primary, so zai's window
    never reaches the ordered tail, and the tail is [flat 1.20 seat, 0.66->1.32
    metered rail] whose order depends on deepseek-v4-flash's multiplier and nothing
    else. The flipped set is therefore BOTH deepseek windows every day — 01:00-04:00
    and 06:00-10:00, 49 of 168 hours — not the four hours zai overlaps, and it does
    not thin out at the weekend when zai stops peaking.

    The primary is glm-5.3-flash since 2026-08-27 and the measurement is unchanged,
    which is the argument's own point: WHICH plan rail is pinned cannot matter to a
    tail it never enters.
    """
    chain = [
        {"model": "glm-5.3-flash", "provider": "zai", "billing_mode": "plan"},
        {"model": "gpt-5.6-luna", "provider": "openai-codex",
         "billing_mode": "subscription"},
        {"model": "deepseek-v4-flash", "provider": "deepseek",
         "billing_mode": "metered"},
    ]
    flipped: list = []
    for when in _week():
        ordered = [
            hop["model"] for hop in order_chain(
                chain, "cheapest_now", pin_primary=True, when=when
            )
        ]
        assert ordered[0] == "glm-5.3-flash", when
        # The order and the price that produced it, asserted together: the tail is
        # ascending effective output price, whichever way round it came out.
        prices = [effective_price(model, when)[1] for model in ordered[1:]]
        assert prices == sorted(prices), (when, ordered, prices)
        if ordered[1] == "gpt-5.6-luna":
            flipped.append((when.weekday(), when.hour))
            assert in_expensive_window("deepseek-v4-flash", when), when
        else:
            assert not in_expensive_window("deepseek-v4-flash", when), when

    assert len(flipped) == 35
    assert sorted(flipped) == sorted(
        [(day, hour) for day in range(5) for hour in (1, 2, 3, 6, 7, 8, 9)]
    )
    # Every one of those hours is deepseek's alone; zai's Mon-Fri window overlaps
    # 20 of them and explains none of them.
    assert len([1 for day, hour in flipped if day < 5 and hour in (6, 7, 8, 9)]) == 20


def test_the_shipped_t2_pin_is_redundant_today_and_says_what_it_protects():
    """`pin_primary: true` changes NOTHING on this roster, at any of 168 hours.

    Found by mutation: flipping the shipped `pin_primary` to false broke no test,
    and the reason is not a missing assertion — it is that the pin is currently
    doing nothing. `cheapest_now` buckets by billing_mode first, so a plan-covered
    primary already leads every dollar-priced hop whether or not it is pinned.

    That is worth pinning in both directions. If this test starts failing, the pin
    has become load-bearing, which is exactly the moment an operator wants to know
    rather than the moment to delete a line that "does nothing".

    What the pin protects is the OTHER shape: a primary in the dollars bucket, where
    the tail can genuinely overtake it. Asserted below so the knob's purpose is
    recorded next to the proof that it is idle.
    """
    shipped = [
        {"model": "glm-5.3-flash", "provider": "zai", "billing_mode": "plan"},
        {"model": "gpt-5.6-luna", "provider": "openai-codex",
         "billing_mode": "subscription"},
        {"model": "deepseek-v4-flash", "provider": "deepseek",
         "billing_mode": "metered"},
    ]
    for when in _week():
        pinned = [hop["model"] for hop in order_chain(
            shipped, "cheapest_now", pin_primary=True, when=when)]
        loose = [hop["model"] for hop in order_chain(
            shipped, "cheapest_now", pin_primary=False, when=when)]
        assert pinned == loose, when
        assert pinned[0] == "glm-5.3-flash", when

    # The shape the pin is for: declare the same primary METERED and its 0.50 out
    # is suddenly comparable, so mimo-v2.5 at 0.28 overtakes it — unless pinned.
    dollar_primary = [
        {"model": "glm-5.3-flash", "provider": "zai", "billing_mode": "metered"},
        {"model": "mimo-v2.5", "provider": "xiaomi", "billing_mode": "metered"},
    ]
    when = _at(SAT, 12)
    assert [hop["model"] for hop in order_chain(
        dollar_primary, "cheapest_now", pin_primary=False, when=when)] == [
        "mimo-v2.5", "glm-5.3-flash"]
    assert [hop["model"] for hop in order_chain(
        dollar_primary, "cheapest_now", pin_primary=True, when=when)] == [
        "glm-5.3-flash", "mimo-v2.5"]


def test_peak_priced_names_only_the_providers_avoid_peak_named():
    """It is this policy's match, not a survey of every peak in the chain."""
    result = apply_time_policy(_mixed_chain(), {"avoid_peak": ["zai"]}, _at(WED, 7))
    assert result["peak_priced"] == ["glm-5.3"]
    assert result["demoted"] == ["glm-5.3"]
    # deepseek-v4-pro is charging 2.0x at this hour too; this tier did not name it,
    # so the policy has nothing to say about it and does not pretend otherwise.
    assert in_expensive_window("deepseek-v4-pro", _at(WED, 7))


def test_prefer_reports_nothing_when_the_preferred_elo_is_already_first():
    """`promoted` is the mirror of `demoted`: a position, not a match."""
    chain = _mixed_chain()
    result = apply_time_policy(
        chain, {"prefer": ["deepseek-v4-pro"]}, _at(WED, 12)
    )
    assert [hop["model"] for hop in result["chain"]] == [
        hop["model"] for hop in chain
    ]
    assert result["promoted"] == []
    assert result["demoted"] == []


def test_time_policy_tolerates_a_junk_hop_in_the_chain():
    """Positions are tracked, so an unusable hop keeps its slot and its silence."""
    chain = [
        "junk",
        {"model": "deepseek-v4-pro", "provider": "deepseek"},
        {"provider": "zai"},
    ]
    result = apply_time_policy(chain, {"avoid_peak": ["deepseek"]}, _at(WED, 7))
    assert result["chain"] == [chain[0], chain[2], chain[1]]
    assert result["demoted"] == ["deepseek-v4-pro"]
    assert result["peak_priced"] == ["deepseek-v4-pro"]


def test_time_policy_moves_every_instance_of_a_repeated_model():
    """A model id is reported once; both of its hops still move."""
    chain = [
        {"model": "deepseek-v4-pro", "provider": "deepseek"},
        {"model": "glm-4.6", "provider": "zai"},
        {"model": "deepseek-v4-pro", "provider": "deepseek"},
    ]
    result = apply_time_policy(chain, {"avoid_peak": ["deepseek"]}, _at(WED, 7))
    assert [hop["model"] for hop in result["chain"]] == [
        "glm-4.6", "deepseek-v4-pro", "deepseek-v4-pro",
    ]
    assert result["demoted"] == ["deepseek-v4-pro"]


def test_both_time_stages_return_exactly_their_documented_keys():
    """Every branch, including the no-ops: a shape that drifts between the
    working path and the degrade path is a key a consumer reads as absent."""
    policy_shape = {"chain", "demoted", "promoted", "peak_priced"}
    cap_shape = {"chain", "capped", "cap_exempt", "bypassed"}
    peaking = [{"model": "deepseek-v4-pro", "provider": "deepseek"},
               {"model": "deepseek-v4-flash", "provider": "deepseek"}]

    for result in (
        apply_time_policy(_mixed_chain(), {"avoid_peak": ["zai"]}, _at(WED, 7)),
        apply_time_policy(_mixed_chain(), {"avoid_peak": ["zai"]}),   # no clock
        apply_time_policy(_mixed_chain(), "not a mapping", _at(WED, 7)),
        apply_time_policy([], {"avoid_peak": ["zai"]}, _at(WED, 7)),
    ):
        assert set(result) == policy_shape

    for result in (
        apply_time_cap(_mixed_chain(), 1.5, _at(WED, 7)),
        apply_time_cap(_mixed_chain(), 1.5),                         # no clock
        apply_time_cap(_mixed_chain(), "junk", _at(WED, 7)),
        apply_time_cap([], 1.5, _at(WED, 7)),
        apply_time_cap(peaking, 1.5, _at(WED, 7)),                   # bypass path
    ):
        assert set(result) == cap_shape


def test_every_move_the_policy_reports_is_a_move_the_returned_chain_made():
    """The agreement, asserted from the chain rather than from the match.

    Whatever the policy matched, `demoted` may only name an elo that ended up
    LATER than it started and `promoted` only one that ended up EARLIER — and an
    identity permutation must report neither. `peak_priced` is checked the other
    way round: it must be a superset of `demoted` (a move implies a price match)
    and every elo in it must really be in an expensive window right now.
    """
    policies = (
        {"avoid_peak": ["deepseek", "zai"]},
        {"avoid_peak": ["zai"]},
        {"avoid_peak": ["deepseek"], "prefer": ["mimo-v2.5"]},
        {"avoid_peak": ["deepseek", "zai"], "prefer": ["glm-5.3"]},
        {"prefer": ["deepseek-v4-pro"]},
        {},
    )
    for chain in _cap_matrix_chains():
        declared_of = {hop["model"]: hop for hop in chain if hop.get("model")}
        source = [hop["model"] for hop in chain if hop.get("model")]
        for policy in policies:
            avoid = {
                name.strip().lower() for name in policy.get("avoid_peak", [])
            }
            for when in _TIME_MATRIX:
                result = apply_time_policy(chain, policy, when)
                label = (policy, when, source)
                ordered = [
                    hop["model"] for hop in result["chain"] if hop.get("model")
                ]

                # Never a filter: the chain is always a permutation.
                assert len(result["chain"]) == len(chain), label
                assert sorted(ordered) == sorted(source), label

                first_in = {model: source.index(model) for model in source}
                first_out = {model: ordered.index(model) for model in ordered}
                for model in result["demoted"]:
                    assert first_out[model] > first_in[model], (label, model)
                for model in result["promoted"]:
                    assert first_out[model] < first_in[model], (label, model)
                if ordered == source:
                    assert result["demoted"] == [], label
                    assert result["promoted"] == [], label

                assert set(result["demoted"]) <= set(result["peak_priced"]), label
                for model in result["peak_priced"]:
                    entry = declared_of[model]
                    assert in_expensive_window(model, when, entry), (label, model)
                    assert entry.get("provider", "").lower() in avoid, (label, model)


# ---------------------------------------------------------------------------
# anthropic — first-party ids, flat priced at every hour
# ---------------------------------------------------------------------------

# (id, (price_in, price_out)) in DECLARED registry order — the order
# `cheapest_now` falls back to when two entries cost the same.
_ANTHROPIC_PRICES = (
    ("claude-fable-5", (10.00, 50.00)),
    ("claude-opus-5", (5.00, 25.00)),
    ("claude-opus-4-8", (5.00, 25.00)),
    ("claude-sonnet-5", (3.00, 15.00)),
    ("claude-haiku-4-5", (1.00, 5.00)),
)

# Everything but haiku, which is the one 200K entry.
_ANTHROPIC_1M = (
    "claude-fable-5", "claude-opus-5", "claude-opus-4-8", "claude-sonnet-5",
)


def test_every_anthropic_model_resolves_through_capabilities_for():
    for model, _ in _ANTHROPIC_PRICES:
        entry = capabilities_for(model)
        assert entry is not None, model
        assert entry["provider"] == "anthropic", model
        assert entry["billing_mode"] == "metered", model


def test_the_prefixed_rail_id_is_left_unknown_on_purpose():
    """Same weights, different rail — so not these prices, and not registered.

    router.example.yaml reaches `us.anthropic.claude-opus-5` through
    `copilot-acp`, a seat that need not bill the first-party per-token rate.
    Registering it here would assert that price anyway, so its
    capability_unknown warning is CORRECT and stays; an operator who knows their
    own seat's terms declares it per elo in router.yaml.
    """
    assert capabilities_for("us.anthropic.claude-opus-5") is None
    assert satisfies(
        "us.anthropic.claude-opus-5", {"min_context": 200_000, "vision": True}
    ) == (True, "capability_unknown")


def test_no_anthropic_entry_trips_the_registry_diagnostics():
    for model, _ in _ANTHROPIC_PRICES:
        entry = MODEL_CAPABILITIES[model]
        # Deliberately window-free: no published peak/off-peak split.
        assert "price_windows" not in entry, model
        assert price_window_diagnostics(model, entry.get("price_windows")) == []
    assert [
        problem for problem in registry_diagnostics() if "claude" in problem
    ] == []


def test_anthropic_models_accept_vision_and_a_200k_min_context():
    for model in _ANTHROPIC_1M:
        assert satisfies(
            model, {"vision": True, "min_context": 200_000}
        ) == (True, ""), model
    # haiku's window is exactly 200K, and equal passes.
    assert satisfies(
        "claude-haiku-4-5", {"vision": True, "min_context": 200_000}
    ) == (True, "")


def test_haiku_rejects_a_400k_min_context_and_the_1m_entries_do_not():
    assert satisfies("claude-haiku-4-5", {"min_context": 400_000}) == (
        False, "context_too_small"
    )
    for model in _ANTHROPIC_1M:
        assert satisfies(model, {"min_context": 400_000}) == (True, ""), model


def test_anthropic_prices_are_the_same_at_every_hour():
    """No `price_windows`, so the flat pair comes back for all 24 hours.

    Asserted on a weekday and on a Saturday because the two shipped peaks are
    hour-gated and one of them is weekday-gated; neither touches these entries.
    """
    for model, expected in _ANTHROPIC_PRICES:
        assert effective_price(model, None) == expected, model
        for day in (WED, SAT):
            for hour in range(24):
                when = _at(day, hour)
                assert effective_price(model, when) == expected, (model, day, hour)


def test_no_anthropic_model_is_ever_in_an_expensive_window():
    for model, _ in _ANTHROPIC_PRICES:
        assert in_expensive_window(model, None) is False, model
        for day in (WED, SAT):
            for hour in range(24):
                assert in_expensive_window(model, _at(day, hour)) is False, (
                    model, day, hour
                )


def test_cheapest_now_sorts_the_anthropic_entries_by_output_price():
    # 07:00 Wednesday is peak on both primary rails and moves nothing here.
    chain = [{"model": model, "provider": "anthropic"}
             for model, _ in _ANTHROPIC_PRICES]
    ordered = order_chain(
        chain, "cheapest_now", pin_primary=False, when=_at(WED, 7)
    )
    assert [hop["model"] for hop in ordered] == [
        "claude-haiku-4-5", "claude-sonnet-5",
        "claude-opus-5", "claude-opus-4-8",       # 25.00 tie, declared order
        "claude-fable-5",
    ]


def test_the_two_opus_entries_tie_and_keep_declared_order():
    when = _at(WED, 7)
    assert effective_price("claude-opus-5", when) == effective_price(
        "claude-opus-4-8", when
    )
    first = [{"model": "claude-opus-5", "provider": "anthropic"},
             {"model": "claude-opus-4-8", "provider": "anthropic"}]
    second = [{"model": "claude-opus-4-8", "provider": "anthropic"},
              {"model": "claude-opus-5", "provider": "anthropic"}]
    assert [hop["model"] for hop in order_chain(
        first, "cheapest_now", pin_primary=False, when=when)] == [
        "claude-opus-5", "claude-opus-4-8"]
    assert [hop["model"] for hop in order_chain(
        second, "cheapest_now", pin_primary=False, when=when)] == [
        "claude-opus-4-8", "claude-opus-5"]


def test_an_anthropic_elo_sorts_inside_the_dollars_bucket():
    """Metered dollars, so behind a plan-credit rail and behind a free one."""
    chain = [
        {"model": "claude-haiku-4-5", "provider": "anthropic"},  # metered, 5.00
        {"model": "tencent/hy3:free", "provider": "nous"},       # free
        {"model": "glm-5.3-flash", "provider": "zai"},           # plan credits
    ]
    ordered = order_chain(
        chain, "cheapest_now", pin_primary=False, when=_at(WED, 7)
    )
    assert [hop["model"] for hop in ordered] == [
        "glm-5.3-flash", "tencent/hy3:free", "claude-haiku-4-5",
    ]


def test_without_safety_margin_is_the_exact_inverse_direction_of_the_recorded_headroom():
    """One ratio, two directions. A second constant elsewhere would be free to drift.

    FLOOR here against ceil there, deliberately: both must err toward LESS room, or a
    token count that satisfied the min_context requirement could fail the prompt budget.
    """
    from router.capabilities import without_safety_margin, _with_safety_margin

    assert without_safety_margin(500) == 400          # 500 * 4 // 5
    assert without_safety_margin(0) == 0
    assert without_safety_margin(-5) == 0, "a nonsense window is no room, never negative"
    # Round-tripping can only shrink, never grow — that is what "both err toward less"
    # means, and it is the property that keeps the two checks consistent.
    for tokens in (1, 7, 128, 999, 65_536, 1_000_000):
        assert without_safety_margin(_with_safety_margin(tokens)) >= tokens - 1
        assert without_safety_margin(tokens) <= tokens


def test_copilot_rows_registered_and_clean_with_only_published_multiplier_recorded():
    """The five rows from card t_03b03eeb exist, lint clean, and the one
    published Copilot premium-request multiplier (gpt-5.4 = 6) is on record."""
    for model in (
        "claude-opus-5.5", "claude-sonnet-5.5", "claude-haiku-5.5",
        "gpt-6.1-sol", "gpt-5.4",
    ):
        assert model in caps_module.MODEL_CAPABILITIES
    assert caps_module.registry_diagnostics() == []
    assert "multiplier 6" in caps_module.MODEL_CAPABILITIES["gpt-5.4"]["notes"]
    assert "2026-10-09" in caps_module.MODEL_CAPABILITIES["gpt-5.4"]["notes"]
