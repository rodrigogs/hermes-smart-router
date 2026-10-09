"""Rule matching engine — first-match over ordered Table 1.

Pure: no IO, no state, no model calls. Deterministic.
Reads blocked_model boolean, never writes it.

Capability filtering, time-windowed pricing and fallback ordering (plan_chain)
stay pure too. The only impurities are INJECTED: a ``random.Random`` for the
``random`` strategy and a ``datetime`` clock (``when``) for everything
time-dependent. Same inputs plus the same rng and the same clock always produce
the same plan. This module never reads the wall clock — it does not even import
``datetime`` at run time — and ``when=None`` means time-agnostic, reported as
such rather than guessed at.

plan_chain applies its stages in ONE fixed order, and the order is load-bearing:

    capability filter  ->  time_cap  ->  time_policy  ->  fallback_strategy

Membership is decided first, position second. A reordering stage running before a
filtering stage could promote an elo that the next stage then removes, and the
trace would show a promotion that never took effect.
"""

from __future__ import annotations

import copy
import inspect
import random
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

if TYPE_CHECKING:  # pragma: no cover - annotations only, never imported at run time
    # The clock is a PARAMETER, so ``datetime`` is needed for the annotation and
    # for nothing else. Importing it under TYPE_CHECKING is what makes "this
    # module cannot read a clock" a property of the file rather than a promise in
    # a docstring.
    from datetime import datetime

# Capability registry lives in a sibling module. It is imported defensively so
# this engine still loads (and degrades to pre-capability behaviour) when the
# registry is absent — every capability code path below is skipped when
# ``_caps is None``.
#
# RELATIVE FIRST, and the order is load-bearing. Hermes loads this plugin as
# ``hermes_plugins.<slug>.router.rules`` (see the package switch in the plugin
# ``__init__``), where ``router`` is NOT a top-level package and the absolute
# name cannot resolve. Trying absolute first there meant ImportError -> ``_caps
# is None`` -> ``plan_chain`` degrading to ``_unfiltered_plan``: the whole
# capability, time_cap, time_policy and cheapest_now layer silently inert in the
# only shape production uses, with nothing but a degrade flag to show for it.
# The absolute name stays as the SECOND attempt because the direct source-loading
# test harnesses put ``router`` on ``sys.path`` as a top-level package. ``None``
# stays as the third, for a genuinely absent registry — but it is no longer
# reachable merely by how the module was loaded.
try:
    from . import capabilities as _caps
except ImportError:  # pragma: no cover - flat layout, or registry not installed
    try:
        from router import capabilities as _caps  # flat-layout fallback
    except ImportError:  # pragma: no cover - registry not installed
        _caps = None  # type: ignore[assignment]

# Signal vocabulary, imported for ONE reason: lint validates every
# ``when.<field>`` name against it. The list is NOT mirrored here — a mirror is
# how the linter and the extractor drift, and a drifted mirror rejects a
# legitimate field at the write gate. Without the module the field-name check is
# skipped (see :func:`_known_when_fields`).
#
# Relative first for the reason above: under the package shape the absolute name
# raised, and lint quietly stopped checking ``when`` field names at the one gate
# that is supposed to be fail-closed.
try:
    from . import signals as _signals
except ImportError:  # pragma: no cover - flat layout, or signals not installed
    try:
        from router import signals as _signals  # flat-layout fallback
    except ImportError:  # pragma: no cover - signals not installed
        _signals = None  # type: ignore[assignment]

# The third external table this module reads is the rule-id → cause map, and it
# is the only one NOT resolved here: ``router.adapter`` owns it
# (``_RULE_ID_CAUSES``, applied by ``adapter._cause_from_rule``) and imports THIS
# module at its own module scope, so an import back at this scope would build one
# half of the pair against a half-initialised other half. It is fetched at call
# time instead — see :func:`_cause_labeller`. Not mirrored here, for the reason
# the signal vocabulary above is not mirrored: a copy drifts, and this particular
# copy did (see :func:`_determine_cause`).

# ---------------------------------------------------------------------------
# Closed operator set — extend ONLY by adding a row, never a new operator family
# ---------------------------------------------------------------------------
#
# Context-conditional routing (est_input_tokens, needs_vision, ...) needs NO new
# operator: those are ordinary signal fields compared with the existing
# gt/gte/lt/lte/eq predicates. Time-conditional routing (utc_hour, utc_weekday)
# needs none either, for the same reason. Do NOT "helpfully" add a range/size
# operator — the set below is closed, and rules.lint() rejects anything outside
# it.

# All recognized operators
_VALID_OPS: Set[str] = {
    "eq", "ne", "in", "nin", "gt", "gte", "lt", "lte",
    "contains", "starts_with", "ends_with", "matches",
}

# Operator FAMILIES, used only by shadow detection: each family describes a
# region of one field's value space that can be compared for containment.
# Anything outside them (contains/starts_with/ends_with/matches) describes a
# region this module refuses to reason about — see :func:`_is_shadowed`.
_COMPARISON_OPS: Set[str] = {"gt", "gte", "lt", "lte"}
_INCLUDE_OPS: Set[str] = {"eq", "in"}
_EXCLUDE_OPS: Set[str] = {"ne", "nin"}

# Regex `matches` is gated to this single field
_MATCHES_ALLOWED_FIELD = "verb_class"

# Closed output set
_VALID_OUTPUT_KEYS: Set[str] = {"profile", "model", "provider", "deny", "action"}

# ---------------------------------------------------------------------------
# Tier fallback strategy / capability vocabulary
# ---------------------------------------------------------------------------

_DEFAULT_FALLBACK_STRATEGY = "sequential"

# Mirrors of the capability registry's closed sets, used only when the registry
# is unavailable so lint stays a fail-closed gate either way.
_FALLBACK_STRATEGIES = frozenset({"sequential", "random", "cheapest_now"})
_FALLBACK_REQUIREMENT_KEYS = frozenset(
    {"min_context", "vision", "tool_calling", "structured_output"}
)
_FALLBACK_BILLING_MODES = frozenset({"plan", "subscription", "metered", "free"})

# The requirement keys that are booleans; `min_context` is the only numeric one.
# Lint checks requirement VALUES against this split, because a requirement whose
# value has the wrong type is silently DISCARDED at plan time (`_as_int` returns
# None), i.e. a floor an operator believes they set and never got.
_BOOL_REQUIREMENT_KEYS = frozenset({"vision", "tool_calling", "structured_output"})

# Closed KEY sets for the two clock knobs, held to the same standard as
# ``fallback_strategy``. A closed strategy set exists so a typo is refused at the
# write gate instead of degrading to sequential at run time; a typo'd KEY is
# worse than a typo'd value, because `time_policy: {avoid_peek: [...]}` and
# `time_cap: {max_multipler: 1.5}` read in the file as active cost controls,
# lint clean, and do nothing whatsoever — the plan reports `demoted: []` /
# `capped: []` and the operator finds out from an invoice.
_TIME_CAP_KEYS = frozenset({"max_multiplier"})
_TIME_POLICY_KEYS = frozenset({"avoid_peak", "prefer"})

# Keys on a tier/hop mapping that are routing or identity, never a capability
# declaration. Everything else is handed to capabilities_for() as a per-elo
# override; the registry filters out fields it does not recognize, so this
# module does not have to mirror the capability field list. ``time_cap`` and
# ``time_policy`` are routing knobs like ``fallback_strategy``: they say when to
# use an elo, never what it can do. Mirrored by ``service._NON_CAPABILITY_KEYS``.
_NON_CAPABILITY_KEYS: Set[str] = {
    "model", "provider", "fallback", "fallback_strategy",
    "pin_primary", "requirements", "time_cap", "time_policy",
}

# Tier aliases are Tn, n >= 1. Matching the real shape rather than a bare "T"
# prefix is what keeps a concrete model id that happens to start with a capital
# T (``Titan-70B``) from being rejected as an unknown tier.
_TIER_NAME_RE = re.compile(r"^T\d+$")

# `when` fields the CALLER injects that no signal produces. `blocked_model` is
# the blocklist boolean threaded through match(); `utc_hour`/`utc_weekday` are
# the clock features and are declared by signals.INJECTED_FEATURE_NAMES, so they
# arrive through _known_when_fields() instead of being named twice.
_INJECTED_WHEN_FIELDS = frozenset({"blocked_model"})

# Bounds for the injected clock features. A `when` clause outside them is a row
# that can never match: the feature is an int in [0, 23] / [0, 6] by
# construction. `lt`/`lte` may name the one-past-the-end value, because that is
# how the half-open `[start, end)` window an operator is copying from reads.
_UTC_HOUR_MAX = 23
_UTC_WEEKDAY_MAX = 6

# A price cap below 1.0 would exclude every flat-priced elo at every hour, so the
# cap could only ever empty the chain and bypass itself. Rejected at the gate.
_MIN_TIME_CAP = 1.0

# Fixed seed for explain()'s dry-run preview (see explain docstring).
_PREVIEW_SEED = 0

# Whether the installed capabilities.order_chain accepts an injected clock.
# Resolved once, at import: a stale registry then loses only the time-relative
# ORDERING instead of the whole plan (the alternative — a TypeError inside
# plan_chain — degrades every stage, including capability filtering).
try:
    _ORDER_CHAIN_ACCEPTS_WHEN = (
        _caps is not None
        and "when" in inspect.signature(_caps.order_chain).parameters
    )
except (AttributeError, TypeError, ValueError):  # pragma: no cover - odd registry
    _ORDER_CHAIN_ACCEPTS_WHEN = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def match(
    features: Dict[str, Any],
    blocked_model: bool,
    rules: List[Dict[str, Any]],
    default: Dict[str, Any],
    tiers: Dict[str, Dict[str, str]],
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Run Table 1 top-down first-match.

    Returns (output, matched_rule_id). output always has at least one key.
    matched_rule_id is None when the default fired.

    Args:
        features: flat signal dict from signals.extract()
        blocked_model: boolean from blocklist pre-filter
        rules: list of rule dicts from router.yaml (id, when, then)
        default: default routing dict
        tiers: {T1: {model, provider}, ...}
    """
    for rule in rules:
        # A rule the operator disabled (console: 'Desativar esta regra') is
        # dead by declaration, not by condition: it never fires, and it cannot
        # stand in the way of the rows behind it. Only the literal boolean
        # False disables — a missing or truthy field keeps the rule live.
        if rule.get("enabled") is False:
            continue
        when = rule.get("when", {})
        if _all_clauses_match(when, features, blocked_model):
            output = dict(rule.get("then", {}))
            if not output:
                continue
            # Resolve tier aliases in model field
            output = _resolve_tiers(output, tiers)
            return output, rule["id"]

    # Fall-through: default
    output = dict(default)
    output = _resolve_tiers(output, tiers)
    return output, None


def resolve_tiers(output: Dict[str, Any], tiers: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """Resolve Tn aliases in output.model against Table 2."""
    return _resolve_tiers(output, tiers)


def plan_chain(
    output: Dict[str, Any],
    features: Dict[str, Any],
    *,
    rng: Optional[random.Random] = None,
    when: Optional["datetime"] = None,
) -> Dict[str, Any]:
    """Build the effective attempt chain for an already tier-resolved output.

    Pure and deterministic: the only impurities are the injected ``rng`` and the
    injected clock ``when``. ``when=None`` is time-agnostic — every
    time-dependent stage is a no-op, ``cheapest_now`` degrades to sequential, and
    the plan says so — never a wall-clock read.

    Stages, in the one shipped order (see the module docstring for why it cannot
    be permuted)::

        [primary] + fallback hops
          -> capability filter   (membership: derived requirements + tier floor)
          -> time_cap            (membership: price ceiling at `when`)
          -> time_policy         (position: avoid_peak / prefer at `when`)
          -> fallback_strategy   (position: sequential / random / cheapest_now)

    Returns, always::

        {chain, requirements, rejected, unknown, bypassed, unsatisfiable,
         strategy, strategy_declared, strategy_degraded, strategy_degraded_reason,
         pin_primary, independent_rails,
         time_agnostic, time_cap_bypassed, capped, demoted, promoted, multipliers}

    plus ``utc_hour`` / ``utc_weekday`` when a clock was injected, and
    ``time_cap`` when the tier declares one. Those three are OMITTED rather than
    set to None: a JSON consumer reads ``Number(null)`` as ``0``, so a null hour
    would render as midnight and a null cap as a ceiling of 0x — the exact class
    of silent wrongness ``time_agnostic`` exists to prevent.

    ``unsatisfiable`` names the requirement keys NO available model could meet,
    carried straight through from the filter. It is a different fact from
    ``rejected``: "this request is pathological" has to be distinguishable from
    "these particular elos were rejected" without an operator reconstructing it
    from three coincidental ``context_too_small`` reasons.

    ``strategy`` is the strategy that ACTUALLY RAN. A declared strategy that
    could not run (``random`` with no rng, ``cheapest_now`` with no clock, a
    strategy outside the closed set) degrades to sequential and is reported as
    ``strategy_declared`` + ``strategy_degraded: True`` + a reason, because a
    sequential chain labelled "random" is indistinguishable from a routing bug.

    ``pin_primary`` is reported for the same reason: the console's strategy
    wording defaults it to True when absent, so a shuffled primary would be
    described as pinned.

    When the capability registry is unavailable (or misbehaves) this degrades to
    the declared chain in declared order with requirements {} and bypassed
    False — i.e. exactly the pre-capability routing behaviour.
    """
    chain = _build_chain(output)
    declared_strategy = _strategy_of(output)
    pin_primary = _pin_primary_of(output)
    cap = _time_cap_of(output)

    # One reading of the clock, normalised once: an unusable `when` (a string
    # decoded from a trace, a bare date) is EXACTLY the no-clock case, so every
    # stage and every reported key agree about which hour — if any — this plan was
    # made at. Deciding it per stage is how `time_agnostic: True` would end up
    # sitting next to a cap that fired.
    clock = when if _clock_parts(when) is not None else None

    if _caps is None:
        # No registry: no filtering, no reordering — today's behaviour exactly.
        return _unfiltered_plan(chain, declared_strategy, pin_primary, cap, clock)

    try:
        requirements = _caps.derive_requirements(features or {}, _tier_floor_of(output))
        filtered = _caps.filter_chain(chain, requirements)
        eligible = filtered.get("eligible") or chain

        # Membership first: the cap decides WHICH elos are attemptable at this
        # hour, so it runs before anything decides their order.
        capped = _apply_time_cap(eligible, cap, clock)
        # Position second, over the set that will actually be attempted.
        policy = _apply_time_policy(capped["chain"], output.get("time_policy"), clock)
        strategy, degraded_reason = _effective_strategy(declared_strategy, rng, clock)
        ordered = _order_chain(policy["chain"], strategy, pin_primary, rng, clock)
        rails = _caps.independent_rails(ordered)
        multipliers = _multipliers_for(ordered, capped["capped"], clock)
    except (AttributeError, TypeError, ValueError, KeyError):
        # Defensive: a stale/partial registry must never break routing. Report
        # the declared chain instead of raising into the request path.
        return _unfiltered_plan(chain, declared_strategy, pin_primary, cap, clock)

    plan: Dict[str, Any] = {
        "chain": ordered,
        "requirements": requirements,
        "rejected": list(filtered.get("rejected") or []),
        "unknown": list(filtered.get("unknown") or []),
        "bypassed": bool(filtered.get("bypassed", False)),
        # Carried, not re-derived: the filter owns "no model could ever meet
        # this", and a second implementation here would be a second answer.
        "unsatisfiable": list(filtered.get("unsatisfiable") or []),
        "strategy": strategy,
        "strategy_declared": declared_strategy,
        "strategy_degraded": bool(degraded_reason),
        "strategy_degraded_reason": degraded_reason,
        "pin_primary": pin_primary,
        "independent_rails": rails,
        "time_cap_bypassed": bool(capped["bypassed"]),
        "capped": list(capped["capped"]),
        "demoted": list(policy["demoted"]),
        "promoted": list(policy["promoted"]),
        # Charging more at this hour, whether or not the order changed. An
        # ``avoid_peak`` whose matched elos are already the trailing hops moves
        # nothing, so ``demoted`` is empty while this still names them.
        "peak_priced": list(policy["peak_priced"]),
        "multipliers": multipliers,
    }
    plan.update(_clock_keys(clock))
    if cap is not None:
        plan["time_cap"] = {"max_multiplier": cap}
    return plan


def explain(
    task: str,
    features: Dict[str, Any],
    blocked_model: bool,
    rules: List[Dict[str, Any]],
    default: Dict[str, Any],
    tiers: Dict[str, Dict[str, str]],
    rng: Optional[random.Random] = None,
    when: Optional["datetime"] = None,
) -> Dict[str, Any]:
    """Full transparency: run match() and return the decision trace.

    Returns {matched_rule_id, output, matched_clauses, cause, chain_plan}.
    cause is from the closed set (see spec). chain_plan is plan_chain()'s
    result, so the console and CLI can show which elos were eligible, which
    were rejected and why, and which fallback strategy applied.

    ``rng`` is passed straight through to plan_chain. When it is None explain
    uses a FIXED-seed random.Random(0): the dry-run preview must be stable
    across reloads, so an operator refreshing the console does not see the
    random-strategy order churn. Production callers inject a request-derived
    seed instead, so real traffic actually spreads across the tail.

    ``when`` is passed straight through too, and is NOT defaulted to a live
    clock: a trace is a diagnostic, and a diagnostic that invents an input has
    invented its answer. With no clock the preview is time-agnostic and labelled
    as such (``chain_plan.time_agnostic``), so a caller that wants the plan for a
    given hour must say which hour.
    """
    output, rule_id = match(features, blocked_model, rules, default, tiers)
    matched_clauses: Dict[str, Any] = {}
    cause = "default_fallthrough"

    if rule_id is not None:
        for rule in rules:
            if rule["id"] == rule_id:
                matched_clauses = _matching_clauses(
                    rule.get("when", {}), features, blocked_model
                )
                break
        # Determine cause from the matched rule
        cause = _determine_cause(rule_id, output)
    else:
        cause = "default_fallthrough"

    # Drill down: blocklist is the most specific
    if output.get("deny"):
        cause = "blocklist_veto"
    # default_fallthrough stays as-is — classifier hasn't fired yet at pure-core stage

    preview_rng = rng if rng is not None else random.Random(_PREVIEW_SEED)

    try:
        chain_plan = plan_chain(output, features, rng=preview_rng, when=when)
    except Exception:  # noqa: BLE001 - a trace must never break the decision
        # explain() is a diagnostic surface (console, CLI, decision log). A
        # planner that blows up must degrade to "no plan" rather than take the
        # operator's only view of the router down with it.
        chain_plan = _empty_chain_plan()

    return {
        "matched_rule_id": rule_id,
        "output": output,
        "matched_clauses": matched_clauses,
        "cause": cause,
        "chain_plan": chain_plan,
    }


# ---------------------------------------------------------------------------
# Config validation (lint)
# ---------------------------------------------------------------------------

def lint(config: Dict[str, Any]) -> List[str]:
    """Validate router.yaml. Returns list of error strings (empty = valid).

    Fail-closed: any error means the config is invalid.
    Checks:
      - rule 'enabled' must be boolean when present; a disabled rule never
        fires (match skips it) and never shadows or is shadowed, but is still
        schema-validated — disabling must not be a hatch past the write gate
      - mandatory default present
      - rules have required fields (id, when, then)
      - rule ids unique
      - when clauses use closed operators
      - when clause FIELD NAMES are known signals (a typo is a dead rule)
      - when.utc_hour / when.utc_weekday bounds are reachable
      - then clauses use closed output keys
      - matches op gated to allowlisted field
      - then.model / default.model name a real tier when they name a tier at all
      - dead/shadowed row detection (by condition, not by key set)
      - tiers T1-T4 present
      - every tier declares its own model + provider
      - tier fallback_strategy / pin_primary / billing_mode / requirements
        (keys AND value types) / time_cap / time_policy (keys AND value shapes)
      - tier fallback hops are {model, provider} mappings
      - declared price_windows are well formed and non-overlapping

    lint() returns HARD errors only: service.py runs it before every write, so
    anything returned here blocks the write. Advisory findings (shared upstream,
    model unknown to the capability registry, a cap that will bypass, a defect in
    the capability registry itself) are legitimate configs an operator may want
    to ship, so they live in lint_warnings() instead.
    """
    errors: List[str] = []

    # yaml.safe_load() may legally yield scalars, lists, or None. Lint is the
    # fail-closed boundary for that external input: return diagnostics rather
    # than leaking a Python type error through the CLI.
    if not isinstance(config, dict):
        return ["router.yaml root must be a mapping"]
    if not config:
        return ["router.yaml not loaded or empty"]

    if "default" not in config:
        errors.append("missing mandatory 'default' routing")

    tiers_cfg = config.get("tiers")
    if not isinstance(tiers_cfg, dict):
        errors.append("missing 'tiers' mapping")
        tiers_cfg = {}
    else:
        for tn in ("T1", "T2", "T3", "T4"):
            if tn not in tiers_cfg:
                errors.append(f"missing tier {tn}")

    errors.extend(_lint_tier_shapes(tiers_cfg))
    errors.extend(_lint_global_price_windows(config))
    errors.extend(_lint_blocklist_shape(config))
    errors.extend(_lint_alignment(config))

    # The default is the route EVERY fall-through takes, so an unresolvable tier
    # alias there misroutes more traffic than any single rule can. The identical
    # check has always existed for rules; this is the same check, symmetrically.
    default_cfg = config.get("default")
    if isinstance(default_cfg, dict):
        default_model = default_cfg.get("model")
        if _is_dangling_tier_alias(default_model, tiers_cfg):
            errors.append(
                f"default: 'model' references unknown tier '{default_model}'"
            )

    rules_raw = config.get("rules", [])
    if not isinstance(rules_raw, list):
        errors.append("'rules' must be a list")
        return errors
    rules: List[Dict[str, Any]] = rules_raw
    # Empty rules with a default is valid — everything falls through to default.

    # Resolved once for the whole file: the vocabulary cannot change mid-lint.
    known_fields = _known_when_fields()

    seen_ids: Set[str] = set()
    for i, rule in enumerate(rules):
        if not isinstance(rule, dict):
            errors.append(f"rule[{i}] must be a mapping")
            continue
        rid = rule.get("id")
        if not rid:
            errors.append(f"rule[{i}] missing 'id'")
            continue
        if rid in seen_ids:
            errors.append(f"duplicate rule id '{rid}'")
        seen_ids.add(rid)

        # enabled is a switch, and only the literal boolean False turns it off
        # (match() tests `is False`). A truthy string would read as 'on' while
        # looking like a setting someone intended — a typo with a face.
        enabled_flag = rule.get("enabled")
        if enabled_flag is not None and not isinstance(enabled_flag, bool):
            errors.append(f"rule '{rid}': 'enabled' must be boolean")

        when = rule.get("when")
        if not when or not isinstance(when, dict):
            errors.append(f"rule '{rid}': missing or invalid 'when'")
            continue

        then = rule.get("then")
        if not then or not isinstance(then, dict):
            errors.append(f"rule '{rid}': missing or invalid 'then'")
            continue

        # Validate when clauses
        for field, condition in when.items():
            # A field no signal produces is a DEAD ROW, not a runtime error:
            # _all_clauses_match returns False for an absent feature, so a typo'd
            # `need_vision` reads in the file as working policy and simply never
            # fires. Lint is the only place the two can be told apart.
            if known_fields is not None and field not in known_fields:
                errors.append(f"rule '{rid}': 'when.{field}' is not a known signal")
            if not isinstance(condition, dict):
                errors.append(f"rule '{rid}': 'when.{field}' must be an op map")
                continue
            for op, val in condition.items():
                if op not in _VALID_OPS:
                    errors.append(
                        f"rule '{rid}': 'when.{field}' uses unknown operator '{op}'"
                    )
                if op == "matches" and field != _MATCHES_ALLOWED_FIELD:
                    errors.append(
                        f"rule '{rid}': 'matches' operator only allowed on "
                        f"'{_MATCHES_ALLOWED_FIELD}', found on '{field}'"
                    )
            errors.extend(_lint_clock_bounds(rid, field, condition))

        # Validate then output keys
        for key in then:
            if key not in _VALID_OUTPUT_KEYS:
                errors.append(f"rule '{rid}': 'then.{key}' not in closed output set")
            if key == "model" and _is_dangling_tier_alias(then[key], tiers_cfg):
                errors.append(
                    f"rule '{rid}': 'then.model' references unknown tier "
                    f"'{then[key]}'"
                )
            if key == "deny" and not isinstance(then[key], bool):
                errors.append(f"rule '{rid}': 'then.deny' must be boolean")

    # Detect shadowed rows — a later rule every one of whose matching feature
    # vectors ALSO matches an earlier rule, so first-match means it can never
    # fire. Decided from the conditions themselves; see _is_shadowed for why an
    # undecidable pair is silence rather than an error.
    for finding in _shadowed_pairs(rules):
        errors.append(finding["message"])

    return errors


# ---------------------------------------------------------------------------
# alignment: block (session-alignment hook, docs/research/2026-10-session-alignment-hook.md §5)
# ---------------------------------------------------------------------------
# Closed vocabulary, table-driven: every key the spec names has ONE entry, and a
# key outside its table fails the lint (a typo is a dead knob, not a setting).
# Nothing reads this block yet (F4+); the lint only keeps the schema honest.

_ALIGN_MODES = ("shadow", "alert", "enforce")
_ALIGN_SCOPES = ("kanban", "chat", "subagent", "delegate_profile")
_ALIGN_ALERT_ON = ("block", "adjust", "shadow_block")
_ALIGN_DELIVER = ("auto", "steer", "pre_tool_call")
_ALIGN_BILLING = ("free", "metered", "plan", "premium", "subscription")
_ALIGN_TIERS = ("T1", "T2", "T3", "T4")


def _al_bool(v: Any) -> Optional[str]:
    return None if isinstance(v, bool) else "must be boolean"


def _al_int(lo: int, hi: Optional[int] = None) -> Callable[[Any], Optional[str]]:
    def check(v: Any) -> Optional[str]:
        if isinstance(v, bool) or not isinstance(v, int) or v < lo or (hi is not None and v > hi):
            return f"must be an integer >= {lo}" + (f" and <= {hi}" if hi is not None else "")
        return None
    return check


def _al_num(lo: float, hi: Optional[float] = None) -> Callable[[Any], Optional[str]]:
    def check(v: Any) -> Optional[str]:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < lo or (
            hi is not None and v > hi
        ):
            return f"must be a number >= {lo}" + (f" and <= {hi}" if hi is not None else "")
        return None
    return check


def _al_enum(vals: Tuple[str, ...]) -> Callable[[Any], Optional[str]]:
    def check(v: Any) -> Optional[str]:
        return None if v in vals else f"must be one of {', '.join(vals)}"
    return check


def _al_enum_list(vals: Tuple[str, ...]) -> Callable[[Any], Optional[str]]:
    def check(v: Any) -> Optional[str]:
        if not isinstance(v, list) or not v or any(x not in vals for x in v):
            return f"must be a non-empty list drawn from {', '.join(vals)}"
        return None
    return check


def _al_str(v: Any) -> Optional[str]:
    return None if isinstance(v, str) and v.strip() else "must be a non-empty string"


def _al_str_list(v: Any) -> Optional[str]:
    if not isinstance(v, list) or not v or any(_al_str(x) for x in v):
        return "must be a non-empty list of non-empty strings"
    return None


def _al_null_str(v: Any) -> Optional[str]:
    return None if v is None or isinstance(v, str) else "must be null or a string"


def _al_pct_list(v: Any) -> Optional[str]:
    if (
        not isinstance(v, list)
        or not v
        or any(isinstance(x, bool) or not isinstance(x, int) or not 1 <= x <= 100 for x in v)
        or any(a >= b for a, b in zip(v, v[1:]))
    ):
        return "must be a non-empty strictly ascending list of integers in 1..100"
    return None


def _al_chain(v: Any) -> Optional[str]:
    """judge.chain: a non-empty list of {model, provider[, billing_mode]} hops."""
    if not isinstance(v, list) or not v:
        return "must be a non-empty list"
    for i, hop in enumerate(v):
        errs = _lint_align_node(f"[{i}]", hop, _AL_JUDGE_HOP)
        if errs:
            return errs[0]
        if "model" not in hop or "provider" not in hop:
            return f"[{i}] must declare 'model' and 'provider'"
    return None


_AL_CONF = _al_num(0, 1)
_AL_TRIGGER: Dict[str, Any] = {
    "pct": _al_pct_list,
    "min_iterations": _al_int(1),
    "goal_turn_pct": _al_int(1, 100),
    "fallback_max_iterations": _al_int(1),
    "cooldown_iterations": _al_int(0),
    "max_evaluations_per_session": _al_int(1),
}
_AL_OVERRIDE: Dict[str, Any] = {
    "enabled": _al_bool,
    **_AL_TRIGGER,
    "min_confidence": {"adjust": _AL_CONF, "block": _AL_CONF},
}
_AL_JUDGE_HOP: Dict[str, Any] = {
    "model": _al_str,
    "provider": _al_str,
    "billing_mode": _al_enum(_ALIGN_BILLING),
}
_AL_SPEC: Dict[str, Any] = {
    "enabled": _al_bool,
    "mode": _al_enum(_ALIGN_MODES),
    "scopes": _al_enum_list(_ALIGN_SCOPES),
    "trigger": _AL_TRIGGER,
    "overrides": {
        "tiers": {t: _AL_OVERRIDE for t in _ALIGN_TIERS},
        "profiles": {},  # open: keyed by profile name, each entry is _AL_OVERRIDE
    },
    "judge": {
        "chain": _al_chain,
        "require_distinct_provider": _al_bool,
        "timeout_seconds": _al_num(0.001),
        "max_input_tokens": _al_int(1),
        "max_output_tokens": _al_int(1),
        "temperature": _al_num(0),
        "confirm_block_with_second_judge": _al_bool,
    },
    "thresholds": {
        "adjust_min_confidence": _AL_CONF,
        "block_min_confidence": _AL_CONF,
        "require_evidence": _al_bool,
        "adjust_limit": _al_int(0),
        "allow_direct_block": _al_bool,
    },
    "actions": {
        "adjust": {"deliver": _al_enum(_ALIGN_DELIVER), "comment": _al_bool},
        "block": {"kanban_block": _al_bool, "comment": _al_bool},
    },
    "alert": {
        "on": _al_enum_list(_ALIGN_ALERT_ON),
        "channels": _al_str_list,
        "include_reasons": _al_bool,
        "include_transcript_excerpt": _al_bool,
        "quiet_hours": _al_null_str,
    },
    "privacy": {
        "redact": _al_bool,
        "max_transcript_chars_to_judge": _al_int(1),
        "allow_providers": _al_str_list,
    },
    "budget": {
        "max_judge_calls_per_day": _al_int(0),
        "breaker": {"threshold": _al_int(1), "cooldown_seconds": _al_int(1)},
    },
    "log": {"path": _al_str},
}


def _lint_align_node(path: str, value: Any, spec: Any) -> List[str]:
    """Walk ``value`` against ``spec``: dict spec = closed key table, callable = leaf."""
    if callable(spec):
        msg = spec(value)
        return [f"{path} {msg}"] if msg else []
    if not isinstance(value, dict):
        return [f"{path} must be a mapping"]
    errors: List[str] = []
    for raw_key, sub in value.items():
        # PyYAML (YAML 1.1) reads a bare `on:` as the boolean True, so the spec's
        # own `alert.on` arrives as key True. Same key, spelled by the parser.
        key = "on" if raw_key is True else raw_key
        if path == "alignment.overrides.profiles":
            errors.extend(_lint_align_node(f"{path}.{key}", sub, _AL_OVERRIDE))
        elif key not in spec:
            errors.append(f"{path}.{key} is not a known alignment key")
        else:
            errors.extend(_lint_align_node(f"{path}.{key}", sub, spec[key]))
    return errors


def _lint_alignment(config: Dict[str, Any]) -> List[str]:
    """Hard-error checks for the top-level ``alignment`` block (absent = fine)."""
    if "alignment" not in config:
        return []
    return _lint_align_node("alignment", config["alignment"], _AL_SPEC)


def _shadowed_pairs(rules: List[Any]) -> Iterator[Dict[str, Any]]:
    """Every shadowed (earlier, later) row pair, in report order, as findings.

    SHARED by :func:`lint` and :func:`lint_findings` — the write gate and the
    jump-target surface must agree about WHICH rows shadow WHICH, or the
    console's "Ver regra N" button could point at a pair lint() never named.
    The message string is built here, ONCE, so the two surfaces cannot drift.

    The guards mirror lint()'s per-rule validation: a row that is not a mapping
    or carries no id was already reported as its own error, and shadow analysis
    skips it so one malformed row cannot mask other errors.

    Yields the same findings :func:`lint_findings` returns; :func:`lint` keeps
    only the ``message``.
    """
    for i in range(len(rules)):
        for j in range(i + 1, len(rules)):
            ri, rj = rules[i], rules[j]
            if not isinstance(ri, dict) or not isinstance(rj, dict):
                continue
            if not ri.get("id") or not rj.get("id"):
                continue
            # A disabled row cannot fire, so it cannot kill the row behind it
            # and nothing can be dead because of it: skip it on BOTH sides.
            # This is what makes the console's disable button resolve a shadow
            # finding — the pair lint used to report must go quiet.
            if ri.get("enabled") is False or rj.get("enabled") is False:
                continue
            earlier_when = ri.get("when")
            later_when = rj.get("when")
            if not isinstance(earlier_when, dict) or not isinstance(later_when, dict):
                continue
            if _is_shadowed(earlier_when, later_when):
                yield {
                    "code": "shadowed",
                    "later_index": j,
                    "later_id": rj["id"],
                    "earlier_index": i,
                    "earlier_id": ri["id"],
                    "message": (
                        f"rule '{rj['id']}' is shadowed by earlier rule '{ri['id']}'"
                    ),
                }


def lint_findings(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Structured findings next to :func:`lint` — never a second write gate.

    lint() returns the strings that block an apply; this returns, for the
    errors that NAME a rule, the coordinates an operator surface needs to jump
    to it. Currently every finding is a shadowed pair, so the shape is:

      {code: 'shadowed', later_index, later_id, earlier_index, earlier_id,
       message}

    ``later_*`` names the row that can never fire (the one an operator must
    reorder or merge); ``earlier_*`` the row that shadows it. ``message`` is the
    exact string :func:`lint` reports for the same pair — that is what lets
    ``service.py`` align the findings list against lint()'s error list without
    touching the write gate.

    Same fail-closed guards as lint() for the same reason: a config that is not
    a mapping, or whose rules are not a list, has no row to point at.
    """
    if not isinstance(config, dict):
        return []
    if not config:
        return []
    rules_raw = config.get("rules", [])
    if not isinstance(rules_raw, list):
        return []
    return list(_shadowed_pairs(rules_raw))


def lint_warnings(config: Dict[str, Any]) -> List[str]:
    """Advisory findings that must NOT block a write.

    Split from lint() on purpose: lint() is the fail-closed write gate, so a
    string returned from there stops the operator's apply. These findings are
    about redundancy quality, not validity — a tier whose first two hops share
    an upstream still routes, a model missing from the capability registry is
    only unverifiable rather than wrong, and a time_cap that will bypass at some
    hour is a cost control the operator may knowingly be shipping anyway.

    The capability registry's own self-check is folded in here too, and it is
    reported FIRST and unconditionally — before the per-tier findings and before
    the config-shape guards below. A registry defect is a property of the router,
    not of the operator's file, so it has to surface even for a config this
    function otherwise has nothing to say about.

    Returns a list of warning strings (empty = nothing to report).
    """
    warnings: List[str] = _registry_warnings()

    if not isinstance(config, dict):
        return warnings

    # Before the tiers guard: a malformed ban row is worth reporting even when the
    # tier table is the thing that is broken, and it does not depend on tiers.
    warnings.extend(_blocklist_row_warnings(config))

    tiers_cfg = config.get("tiers")
    if not isinstance(tiers_cfg, dict):
        return warnings

    for tn, tier in tiers_cfg.items():
        if not isinstance(tier, dict):
            continue  # shape errors are lint()'s job, not ours
        chain = _tier_chain(tier)

        # Two hops behind the same upstream are one rail, not two.
        if len(chain) >= 2:
            first = _upstream_group(str(chain[0].get("provider") or ""))
            second = _upstream_group(str(chain[1].get("provider") or ""))
            if first and first == second:
                warnings.append(
                    f"tier '{tn}': first two hops share upstream '{first}' "
                    f"— no independent fallback"
                )

        # Unverifiable capabilities: neither the registry nor the YAML knows.
        if _caps is None:
            continue
        seen_models: Set[str] = set()
        for hop in chain:
            model = hop.get("model")
            if not isinstance(model, str) or model in seen_models:
                continue
            seen_models.add(model)
            # provider/model are identity, not capability: a hop that only names
            # its provider still "declares no capabilities".
            declared = _declared_capabilities(hop)
            try:
                known = _caps.capabilities_for(model, declared or None)
            except (AttributeError, TypeError, ValueError):
                continue
            if known is None:
                warnings.append(
                    f"tier '{tn}': model '{model}' is unknown to the capability "
                    f"registry and declares no capabilities"
                )

        warnings.extend(_time_warnings(tn, tier, chain))

    warnings.extend(_fallback_chain_warnings(config, tiers_cfg))
    warnings.extend(_classifier_provider_warnings(config, tiers_cfg))

    return warnings


def _classifier_provider_warnings(
    config: Dict[str, Any], tiers_cfg: Dict[str, Any]
) -> List[str]:
    """The Stage 1 classifier's provider must not be a tier PRIMARY's provider.

    The classifier sits on the critical path of every delegation; sharing a
    provider with the primaries means one quota or outage takes out routing and
    the work it routes at once. Advisory (not a write-gate error) so a config
    that knowingly keeps one rail still applies. Checks the flat pair and
    chain[0], the only hops that dispatch today.
    """
    clf = config.get("classifier")
    if not isinstance(clf, dict):
        return []
    primaries = {
        str(t["provider"])
        for t in tiers_cfg.values()
        if isinstance(t, dict) and isinstance(t.get("provider"), str)
    }
    hops = [("classifier", clf)]
    chain = clf.get("chain")
    if isinstance(chain, list) and chain and isinstance(chain[0], dict):
        hops.append(("classifier.chain[0]", chain[0]))
    return [
        f"{label}: provider '{h['provider']}' is also a tier primary provider "
        f"({', '.join(sorted(primaries))}) — classifier shares a failure domain "
        f"with the primaries"
        for label, h in hops
        if isinstance(h.get("provider"), str) and h["provider"] in primaries
    ]


def _lint_blocklist_shape(config: Dict[str, Any]) -> List[str]:
    """HARD errors for the COARSE shape of the ``blocklist:`` section.

    This is a write gate, and the gap it closes is a fail-OPEN on the one
    component whose whole job is to refuse. Measured on the shipped policy with
    ``blocklist: off`` appended: ``lint`` returned ``[]``, so ``/status`` reported
    ``valid: True`` and ``plan``/``apply`` would have PERSISTED it — after which
    ``Blocklist.__init__`` raised ``AttributeError``, ``adapter.route`` died,
    ``_route_task`` swallowed it at DEBUG, every delegation answered ``bad_args``,
    and every manual ban was unenforced.

    COARSE ONLY, and the line is deliberate:

      * The four container shapes are errors — a mapping under ``blocklist``, a
        list for ``manual_ban`` and ``fallback_chain``, a mapping for
        ``auto_breaker``. Get one of these wrong and the section as a whole cannot
        mean anything.
      * A malformed ROW inside those lists is NOT an error. A ban row naming no
        model is a documented shippable input (it bans every model), and
        ``test_plugin_status_drops_a_ban_row_it_cannot_name`` pins that a row the
        UI cannot name still round-trips. Per-row problems are reported by
        ``lint_warnings`` and skipped by ``_ban_row`` at the match site.
    """
    errors: List[str] = []
    blocklist = config.get("blocklist")
    if blocklist is None:
        return errors  # absent is valid: no bans, breaker off
    if not isinstance(blocklist, dict):
        return [
            f"'blocklist' must be a mapping, got {type(blocklist).__name__} "
            f"— a blocklist that cannot be read blocks nothing"
        ]
    for key in ("manual_ban", "fallback_chain"):
        value = blocklist.get(key)
        if value is not None and not isinstance(value, list):
            errors.append(
                f"blocklist.{key} must be a list, got {type(value).__name__}"
            )
    breaker = blocklist.get("auto_breaker")
    if breaker is not None and not isinstance(breaker, dict):
        errors.append(
            f"blocklist.auto_breaker must be a mapping, got "
            f"{type(breaker).__name__}"
        )
    return errors


def _blocklist_row_warnings(config: Dict[str, Any]) -> List[str]:
    """Advisory findings for ``manual_ban`` rows that cannot ban anything.

    Skipped at the match site by ``blocklist._ban_row`` rather than raising, so
    without this they were silently ignored — an operator who typed
    ``- glm-5.3`` instead of ``- {model: glm-5.3}`` got no ban and no complaint.

    ADVISORY, not an error: the row shape has a documented shippable edge (a row
    naming no model bans every model), so the write gate must not refuse the file
    over one line it can simply not honour.
    """
    blocklist = config.get("blocklist")
    if not isinstance(blocklist, dict):
        return []
    bans = blocklist.get("manual_ban")
    if not isinstance(bans, list):
        return []
    warnings: List[str] = []
    for i, ban in enumerate(bans):
        if not isinstance(ban, dict):
            warnings.append(
                f"blocklist.manual_ban[{i}] must be a mapping "
                f"({{model, provider, reason}}), got {type(ban).__name__} "
                f"— this row bans nothing"
            )
            continue
        for key in ("model", "provider"):
            value = ban.get(key, "")
            if not isinstance(value, str):
                warnings.append(
                    f"blocklist.manual_ban[{i}]: '{key}' must be a string, got "
                    f"{type(value).__name__} — this row bans nothing"
                )
    return warnings


def _fallback_chain_warnings(
    config: Dict[str, Any], tiers_cfg: Dict[str, Any],
) -> List[str]:
    """Tier members missing from ``blocklist.fallback_chain``.

    That list is documented as the union of every tier member in tier order, and
    it has to be REGENERATED BY HAND whenever ``tiers`` changes — nothing else
    checked that, and the console's tier editor does not do it. So this reports the
    drift the operator cannot otherwise see.

    ADVISORY, not a hard error, and the severity is the point. The list used to be
    the ONLY source the blocklist veto would substitute from, which made a missing
    member a total refusal: ``Blocklist.fallback_for`` walks the list POSITIONALLY,
    so a banned model absent from it had no position to walk from and the turn was
    denied outright. ``adapter._reachable_replacement`` now searches the planned
    chain and the tier's own declared hops FIRST, so a gap here costs the
    cross-tier escape hatch rather than the turn — a quality finding, and blocking
    a write over it would strand the operator outside the guarded path over
    something their policy still routes fine without.

    Reported per missing model rather than as one summary line so the string names
    what to add, and sorted so the report is stable across processes.
    """
    blocklist_cfg = config.get("blocklist")
    if not isinstance(blocklist_cfg, dict):
        return []
    chain = blocklist_cfg.get("fallback_chain")
    if not isinstance(chain, list):
        # A missing or malformed list is lint()'s shape question, not ours; with
        # no list to compare against there is no drift to report.
        return []
    listed = {model for model in chain if isinstance(model, str)}

    missing: List[str] = []
    for tier in tiers_cfg.values():
        if not isinstance(tier, dict):
            continue
        for hop in _tier_chain(tier):
            model = hop.get("model")
            if isinstance(model, str) and model and model not in listed:
                missing.append(model)
    return [
        f"blocklist.fallback_chain does not list tier member '{model}' "
        f"— it cannot be substituted for, and cannot be escaped to"
        for model in sorted(set(missing))
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _registry_warnings() -> List[str]:
    """``capabilities.registry_diagnostics()``, folded into the advisory channel.

    This is that check's production caller, and it needs one: a diagnostic
    nobody calls is a diagnostic that does not exist. YAML-declared
    ``price_windows`` are validated per tier by :func:`_lint_price_windows`, but
    a malformed or overlapping window shipped in ``MODEL_CAPABILITIES`` itself is
    invisible to every other gate — and the router and the console then price the
    same hour differently, each taking a different one of two matching windows.

    Advisory rather than blocking, unlike the YAML window check: the defect is in
    the router's own registry, not in the operator's file, so refusing their
    write for it would strand them outside the guarded path over something they
    cannot fix from YAML. The strings are already shaped
    ``model '<id>': <defect>``, so they append verbatim.
    """
    fn = getattr(_caps, "registry_diagnostics", None) if _caps else None
    if not callable(fn):
        return []
    try:
        return [str(problem) for problem in fn() or []]
    except Exception:  # noqa: BLE001 - an optional self-check must not break lint
        # A registry too broken to describe itself must not break the lint
        # report that would have told the operator everything else.
        #
        # Broad HERE and only here: `fn` is an OPTIONAL hook looked up by name on
        # whatever object `_caps` happens to be, so its failure modes are not
        # ours to enumerate. A narrow tuple would say "this self-check may fail in
        # these four ways", which is a claim about foreign code we cannot make; the
        # contract we can state is "however this self-check fails, it must not take
        # the lint path — and therefore the whole write gate — down with it".
        # Everywhere else in this module the exception set IS knowable (our own
        # parsing of the operator's YAML), so it stays narrow: a broad catch there
        # would swallow real defects instead of reporting them as diagnostics.
        return []


def _requirement_keys() -> frozenset:
    """Closed requirement key set, from the registry when it is importable."""
    if _caps is not None:
        keys = getattr(_caps, "REQUIREMENT_KEYS", None)
        if isinstance(keys, (frozenset, set)):
            return frozenset(keys)
    return _FALLBACK_REQUIREMENT_KEYS


def _billing_modes() -> frozenset:
    """Closed billing mode set, from the registry when it is importable."""
    if _caps is not None:
        modes = getattr(_caps, "BILLING_MODES", None)
        if isinstance(modes, (frozenset, set)):
            return frozenset(modes)
    return _FALLBACK_BILLING_MODES


def _fallback_strategies() -> frozenset:
    """Closed strategy set, from the registry when it is importable.

    Read from the registry (which owns ``order_chain``) so lint can never reject
    a strategy the orderer supports, or accept one it silently degrades.
    """
    if _caps is not None:
        strategies = getattr(_caps, "FALLBACK_STRATEGIES", None)
        if isinstance(strategies, (frozenset, set)):
            return frozenset(strategies)
    return _FALLBACK_STRATEGIES


def _known_when_fields() -> Optional[frozenset]:
    """Every legal ``when.<field>`` name, or None when it cannot be known.

    ``signals`` owns the list (extracted + injected clock features) and it is
    imported, never mirrored: a mirror drifts, and a drifted mirror rejects a
    legitimate field at the write gate. ``blocked_model`` is added here because
    it is threaded through ``match()`` rather than produced by ``extract()``.

    None means "do not check" — without ``signals`` there is no canonical list,
    and guessing one would risk blocking a valid config, which is the failure
    mode this whole check is trying to avoid.
    """
    if _signals is None:
        return None
    names = getattr(_signals, "KNOWN_FEATURE_NAMES", None)
    if not isinstance(names, (frozenset, set)) or not names:
        return None
    return frozenset(names) | _INJECTED_WHEN_FIELDS


def _is_dangling_tier_alias(value: Any, tiers_cfg: Dict[str, Any]) -> bool:
    """True when ``value`` looks like a tier alias that the table does not have.

    A name the table DOES carry resolves, whatever it is spelled like, so custom
    tier names keep working. Only a Tn-shaped name that is absent is an error:
    matching the real alias shape rather than a bare "T" prefix is what stops a
    concrete model id such as ``Titan-70B`` from being read as a broken tier
    reference.
    """
    if not isinstance(value, str) or not value:
        return False
    if isinstance(tiers_cfg, dict) and value in tiers_cfg:
        return False
    return bool(_TIER_NAME_RE.match(value))


def _upstream_group(provider: str) -> str:
    """Upstream group for a provider.

    Delegates to the registry, which knows the reseller aliases. Without it,
    fall back to provider identity: that still catches the literal
    same-provider pair without duplicating the alias table here.
    """
    if _caps is not None:
        try:
            return str(_caps.upstream_group(provider))
        except (AttributeError, TypeError, ValueError):
            return provider
    return provider


def _empty_chain_plan() -> Dict[str, Any]:
    """The plan shape with nothing in it — the last-resort degraded default.

    Time-agnostic by construction: nothing was planned, so no hour can be claimed
    for it. ``utc_hour``/``utc_weekday``/``time_cap`` are absent for the reason
    plan_chain documents — a null there reads as 0 in a JSON consumer.
    """
    return {
        "chain": [],
        "requirements": {},
        "rejected": [],
        "unknown": [],
        "bypassed": False,
        "unsatisfiable": [],
        "strategy": _DEFAULT_FALLBACK_STRATEGY,
        "strategy_declared": _DEFAULT_FALLBACK_STRATEGY,
        "strategy_degraded": False,
        "strategy_degraded_reason": "",
        "pin_primary": True,
        "independent_rails": 0,
        "time_agnostic": True,
        "time_cap_bypassed": False,
        "capped": [],
        "demoted": [],
        "promoted": [],
        "peak_priced": [],
        "multipliers": {},
    }


def _unfiltered_plan(
    chain: List[Dict[str, Any]],
    declared_strategy: str,
    pin_primary: bool,
    cap: Optional[float],
    when: Optional["datetime"],
) -> Dict[str, Any]:
    """Declared chain in declared order — the pre-capability plan shape.

    independent_rails degrades to a distinct-provider count because upstream
    grouping (reseller aliases) is knowledge that lives in the registry.
    ``unsatisfiable`` is empty for the same reason: without the registry there is
    no largest-registered-window to compare a floor against, and an empty list
    claims nothing rather than claiming nothing is wrong.

    Every time-dependent field reports the honest nothing: no stage ran, so no
    elo was capped, demoted, promoted or repriced, and the strategy that ran is
    sequential. A declared strategy other than sequential is therefore a REPORTED
    degrade — without the registry there is no orderer to run it, and a plan that
    claimed otherwise would be describing an order nobody produced.
    """
    reason = (
        ""
        if declared_strategy == _DEFAULT_FALLBACK_STRATEGY
        else "the capability registry is unavailable, so nothing was reordered"
    )
    plan: Dict[str, Any] = {
        "chain": chain,
        "requirements": {},
        "rejected": [],
        "unknown": [],
        "bypassed": False,
        "unsatisfiable": [],
        "strategy": _DEFAULT_FALLBACK_STRATEGY,
        "strategy_declared": declared_strategy,
        "strategy_degraded": bool(reason),
        "strategy_degraded_reason": reason,
        "pin_primary": pin_primary,
        "independent_rails": len({hop.get("provider") for hop in chain}),
        "time_cap_bypassed": False,
        "capped": [],
        "demoted": [],
        "promoted": [],
        "peak_priced": [],
        "multipliers": {},
    }
    plan.update(_clock_keys(when))
    if cap is not None:
        plan["time_cap"] = {"max_multiplier": cap}
    return plan


def _declared_capabilities(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Per-elo capability overrides declared on a tier or fallback hop."""
    return {
        key: value
        for key, value in entry.items()
        if key not in _NON_CAPABILITY_KEYS
    }


def _tier_chain(tier: Dict[str, Any]) -> List[Dict[str, Any]]:
    """[primary hop] + declared fallback hops for one tier mapping."""
    chain: List[Dict[str, Any]] = []
    if isinstance(tier.get("model"), str):
        primary = {"model": tier["model"], "provider": tier.get("provider")}
        primary.update(_declared_capabilities(tier))
        chain.append(primary)
    fallback = tier.get("fallback")
    if isinstance(fallback, list):
        for hop in fallback:
            if isinstance(hop, dict) and hop.get("model"):
                chain.append(dict(hop))
    return chain


def with_global_price_windows(config: Any) -> Any:
    """Return ``config`` with the top-level ``price_windows`` overlay merged in.

    The overlay is the writable, model-keyed price-window table at the top of
    ``router.yaml`` (spec t_c90c5336). It is applied to every tier primary and
    fallback hop whose model it names, and it sits BELOW a per-elo declaration
    the hop already carries: a hop that declares its own ``price_windows`` keeps
    it, so the precedence is ``tier[].price_windows`` (declared) > this overlay >
    the code registry. Routing reads the windows through the same ``declared``
    override channel a per-elo declaration uses, so no consumer learns a new
    path.

    Returns a DEEP COPY when there is an overlay to apply — never the caller's
    object, so a plan can mutate the returned view without writing back into the
    parsed config — and the caller's own object (unmutated) when there is none,
    so the hot routing path pays nothing when the operator has not written an
    overlay. A malformed overlay (not a mapping) is a no-op here: ``lint()`` is
    the gate that refuses it, and nothing reaches the routing path without
    passing the gate.
    """
    if not isinstance(config, dict):
        return config
    overlay = config.get("price_windows")
    # An empty overlay (``{}``) is "no overlay": returning the caller's object
    # avoids a deep copy on the hot routing path for the state the console's
    # write path can leave behind (posting an edited policy with no windows).
    if not isinstance(overlay, dict) or not overlay:
        return config
    result = copy.deepcopy(config)
    tiers = result.get("tiers")
    if isinstance(tiers, dict):
        for tier in tiers.values():
            if isinstance(tier, dict):
                _inject_overlay_windows(tier, overlay)
                for hop in tier.get("fallback") or []:
                    if isinstance(hop, dict):
                        _inject_overlay_windows(hop, overlay)
    return result


def _inject_overlay_windows(hop: Dict[str, Any], overlay: Dict[str, Any]) -> None:
    """Inject one model's overlay windows into ``hop``, unless it declares its own.

    Two spellings are accepted because the provenance field grew a sibling out of
    the plain list: ``model: [window, ...]`` (the list) and ``model:
    {price_windows: [...], price_windows_verified: 'YYYY-MM-DD'}`` (the extended
    form). Both inject the same ``price_windows`` list into ``declared``; the
    extended form also carries the human confirmation date so the catalogue can
    serve it. The injection is a COPY so routing can never mutate the parsed
    overlay, and a hop that already carries ``price_windows`` is left alone — the
    per-elo declaration is the deliberate local exception.
    """
    model = hop.get("model")
    if not isinstance(model, str) or not model:
        return
    if "price_windows" in hop:
        return
    entry = overlay.get(model)
    if entry is None:
        return
    if isinstance(entry, dict):
        windows = entry.get("price_windows")
        if windows is not None:
            hop["price_windows"] = copy.deepcopy(windows)
        if "price_windows_verified" in entry:
            hop["price_windows_verified"] = entry["price_windows_verified"]
        return
    hop["price_windows"] = copy.deepcopy(entry)


#: The registry field names a tier or hop may DECLARE, when capabilities.py can be
#: asked. Local fallback for a checkout whose sibling module predates the export.
_FALLBACK_REGISTRY_FIELDS = frozenset({
    "provider", "context_window", "max_input_tokens", "max_output", "vision",
    "tool_calling", "structured_output", "billing_mode", "price_in",
    "price_out", "price_windows", "price_windows_verified", "notes",
})


def _registry_fields() -> frozenset:
    """Field names ``capabilities._declared_overrides`` will actually keep.

    Read through the module-level ``_caps`` handle with ``getattr`` +
    ``isinstance``, exactly like ``_requirement_keys`` and ``_billing_modes``: a
    direct ``import router.capabilities`` fails under Hermes' ``hermes_plugins.
    <slug>`` package shape, which
    ``TestCapabilityLayerIsLiveUnderHermesPluginPackageShape`` pins.
    """
    if _caps is not None:
        fields = getattr(_caps, "_REGISTRY_FIELDS", None)
        if isinstance(fields, (frozenset, set)):
            return frozenset(fields)
    return _FALLBACK_REGISTRY_FIELDS


def _lint_declared_capabilities(label: str, entry: Dict[str, Any]) -> List[str]:
    """Hard errors for capability keys the registry will silently DROP.

    ``_declared_capabilities`` is EXCLUSION-based: everything not in
    ``_NON_CAPABILITY_KEYS`` is harvested and handed to ``capabilities_for`` as a
    per-elo override — and ``capabilities._declared_overrides`` then keeps only
    ``_REGISTRY_FIELDS`` and drops the rest without a word. Neither ``lint`` nor
    ``lint_warnings`` said anything, so a typo was invisible in both channels.

    Measured on the real ``router.yaml``: ``visssion: True`` and
    ``min_context: 128000`` on a REGISTERED T3 hop both linted clean. The first is
    a one-letter typo that silently un-declares vision on the hop the operator was
    trying to describe. The second is worse, because the v2 spec's own hop example
    uses it: ``min_context`` is a REQUIREMENT (a floor, belonging under
    ``requirements:``), not a registry field, and declaring it on a hop asserts
    nothing at all — the spec promised a ``min_context -> context_window`` alias
    that was never implemented and must not be, since it would re-merge the two
    vocabularies the code deliberately separated (a floor is not a ceiling).

    Scoped to ``tiers`` deliberately. ``classifier:`` legitimately carries
    ``temperature``/``max_tokens``/``timeout_seconds``, which are not capability
    declarations and are validated elsewhere.
    """
    allowed = _NON_CAPABILITY_KEYS | _registry_fields()
    unknown = sorted(key for key in entry if isinstance(key, str) and key not in allowed)
    return [
        f"{label} declares unknown capability key '{key}'" for key in unknown
    ]


def _lint_tier_shapes(tiers_cfg: Dict[str, Any]) -> List[str]:
    """Hard-error checks for the per-tier routing knobs."""
    errors: List[str] = []
    # Unreachable from lint(), which reports a non-mapping `tiers` itself and then
    # normalises it to {} before calling here. Kept because `tiers_cfg: Any` is
    # what a caller may hand a private helper, and left uncovered on purpose: a
    # test for it would assert that the guard exists, not any behaviour it holds.
    if not isinstance(tiers_cfg, dict):  # pragma: no cover - lint() normalises first
        return errors

    strategies = _fallback_strategies()
    for tn, tier in tiers_cfg.items():
        if not isinstance(tier, dict):
            errors.append(f"tier '{tn}' must be a mapping")
            continue

        # The tier's OWN elo, checked with the same rigour as its fallback hops.
        # Without this, _resolve_tiers fills a missing model from the alias
        # itself, so deleting one line from T2 lints clean and routes 100% of
        # that tier's traffic to a model literally named "T2".
        for key in ("model", "provider"):
            if key not in tier:
                errors.append(f"tier '{tn}': missing '{key}'")
            elif not isinstance(tier[key], str) or not tier[key].strip():
                errors.append(f"tier '{tn}': '{key}' must be a non-empty string")

        errors.extend(_lint_declared_capabilities(f"tier '{tn}':", tier))

        # The isinstance guard is not decoration: `x in frozenset` raises
        # TypeError for an unhashable x, and YAML can legally produce a list or a
        # mapping here. lint() is the write gate, so it has to REPORT that, not
        # raise through the operator's apply.
        if "fallback_strategy" in tier:
            declared = tier["fallback_strategy"]
            if not isinstance(declared, str) or declared not in strategies:
                errors.append(
                    f"tier '{tn}': 'fallback_strategy' must be one of "
                    f"{', '.join(sorted(strategies))}"
                )

        if "pin_primary" in tier and not isinstance(tier["pin_primary"], bool):
            errors.append(f"tier '{tn}': 'pin_primary' must be boolean")

        errors.extend(_lint_billing_mode(f"tier '{tn}'", tier))
        errors.extend(_lint_requirements(tn, tier))
        errors.extend(_lint_time_knobs(tn, tier))

        errors.extend(_lint_tier_fallback(tn, tier))
        errors.extend(_lint_price_windows(tier))

    return errors


def _lint_tier_fallback(tn: str, tier: Dict[str, Any]) -> List[str]:
    """Hard-error checks for ONE tier's ``fallback`` list and each hop in it.

    Extracted from :func:`_lint_tier_shapes`, which was the deepest function in the
    repo at seven levels of real nesting where the next was five — the whole depth
    came from this one block. Signature mirrors its siblings ``_lint_requirements``
    and ``_lint_time_knobs`` so the caller reads as a flat list of checks.

    Behaviour-preserving, and the shape change is the point: the old
    ``if fallback is not None: if not isinstance(...): ... else: ...`` pair became a
    single ``if isinstance(...) / elif is not None``, which removes one level without
    touching a verdict.
    """
    errors: List[str] = []
    fallback = tier.get("fallback")
    if not isinstance(fallback, list):
        if fallback is not None:
            errors.append(f"tier '{tn}': 'fallback' must be a list")
        return errors

    for i, hop in enumerate(fallback):
        label = f"tier '{tn}': fallback[{i}]"
        if (
            not isinstance(hop, dict)
            or not hop.get("model")
            or not hop.get("provider")
        ):
            errors.append(f"{label} must be a mapping with 'model' and 'provider'")
        # A hop's own knobs, checked with the same rigour as the tier's: the planner
        # reads a hop's declaration with the same weight as the tier's, so a knob
        # validated on one and not the other is a gap the operator cannot see.
        # Reported alongside a shape defect rather than instead of it — lint returns
        # every diagnostic it has, not the first. Guarded because a non-dict hop was
        # already reported above and `'k' in 7` raises.
        if not isinstance(hop, dict):
            continue
        # IDENTITY, held to the tier's own standard — the same symmetry
        # _lint_billing_mode exists for. A hop's model and provider were checked for
        # TRUTHINESS only while the tier's own must be a non-empty string, so
        # `model: 4.7` (what YAML makes of an unquoted glm-4.7) was refused on a
        # tier and passed the gate on a hop. It is not an inert typo: _build_chain
        # keeps the hop, the capability filter reads its id as "" and lets it through
        # on the FAIL-OPEN unknown path, and ``unknown`` cannot even name it — that
        # list collects string ids — so the one flag whose job is to make a fail-open
        # loud stays silent and the router attempts a rail whose model id is a float.
        # This gate is the only place it is visible. A missing or blank value is
        # already reported above, so only a truthy-but-unusable one is named here;
        # reporting both would be two errors for one defect.
        for key in ("model", "provider"):
            value = hop.get(key)
            if value and (not isinstance(value, str) or not value.strip()):
                errors.append(f"{label}: '{key}' must be a non-empty string")
        errors.extend(_lint_billing_mode(label, hop))
        errors.extend(_lint_declared_capabilities(label, hop))
    return errors


def _lint_billing_mode(label: str, entry: Dict[str, Any]) -> List[str]:
    """Validate the ``billing_mode`` on ONE elo — a tier's own, or a fallback hop.

    One implementation for both, called with the label of whoever declared it, so
    the two can never be held to different standards. They were: the tier's mode
    was checked and a hop's was not, and a hop declares it for the same reason a
    tier does — ``resolve_tiers`` hands every hop's declarations to
    ``capabilities._declared_overrides``, where a declared mode OVERRIDES the
    registry's correct one. So ``billing_mode: meterd`` on a hop is not an inert
    typo: ``capabilities._billing_rank`` finds no such mode, drops that elo into
    the unknown bucket, and ``cheapest_now`` — whose OUTER sort key is that bucket
    — sorts it last. Measured on a `cheapest_now` tier with `pin_primary: false`,
    primary gpt-5.5 (subscription, $30.00/1M out) and one hop glm-4.7-flashx
    (metered, $0.40/1M out): with the typo lint returned [] and the chain came
    back gpt-5.5 first, i.e. the cost strategy demoted the rail that was 75x
    cheaper on output, for one missing character in a file that read as correct.

    Value-shaped like every other closed set in this gate: the isinstance guard
    comes first because ``x in frozenset`` raises TypeError for an unhashable x
    and YAML can legally produce a list or a mapping here, and lint() is the write
    gate — that has to come back as a diagnostic, never as a raise through the
    operator's apply.
    """
    if "billing_mode" not in entry:
        return []
    mode = entry["billing_mode"]
    if isinstance(mode, str) and mode in _billing_modes():
        return []
    return [f"{label}: 'billing_mode' must be one of {sorted(_billing_modes())}"]


def _lint_requirements(tn: str, tier: Dict[str, Any]) -> List[str]:
    """Key AND value checks for one tier's ``requirements`` floor.

    Values matter as much as keys: a requirement whose type is wrong is silently
    DISCARDED at plan time (``min_context: "lots"`` reads back as None), so the
    operator gets a floor they believe they set and never had. A discarded floor
    fails in the direction of routing to a model that cannot serve the request,
    which is exactly what the floor exists to prevent.
    """
    if "requirements" not in tier:
        return []
    reqs = tier["requirements"]
    if not isinstance(reqs, dict):
        return [f"tier '{tn}': 'requirements' must be a mapping"]

    errors: List[str] = []
    for key, value in reqs.items():
        if key not in _requirement_keys():
            errors.append(
                f"tier '{tn}': 'requirements.{key}' not in closed requirement set"
            )
            continue
        if key == "min_context":
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                errors.append(
                    f"tier '{tn}': 'requirements.min_context' must be a "
                    f"positive integer"
                )
        elif key in _BOOL_REQUIREMENT_KEYS and not isinstance(value, bool):
            errors.append(f"tier '{tn}': 'requirements.{key}' must be a boolean")
    return errors


def _lint_time_knobs(tn: str, tier: Dict[str, Any]) -> List[str]:
    """Key AND value checks for ``time_cap`` and ``time_policy``.

    KEYS are a closed set, exactly like ``fallback_strategy``'s values. An
    unrecognised key is a HARD error naming the key, because the failure it
    causes is invisible everywhere else: ``time_policy: {avoid_peek: [deepseek]}``
    and ``time_cap: {max_multipler: 1.5}`` are one character from a working cost
    control, read in the file as if they were one, pass a fail-closed gate, and
    then do nothing at all — the plan reports ``demoted: []`` / ``capped: []``
    and the tier keeps billing at 2.0x through every peak. Silently ignoring an
    unknown key is what makes that possible, so it is not silently ignored.

    ``max_multiplier`` must be >= 1.0: a cap below the base rate excludes every
    flat-priced elo at every hour, so it could only ever empty the chain and
    bypass itself — a cost control that is guaranteed not to control cost. The
    module honours such a value literally; the gate is where it is refused.
    """
    errors: List[str] = []

    if "time_cap" in tier:
        cap_cfg = tier["time_cap"]
        if not isinstance(cap_cfg, dict):
            errors.append(f"tier '{tn}': 'time_cap' must be a mapping")
        else:
            for key in cap_cfg:
                if key not in _TIME_CAP_KEYS:
                    errors.append(
                        f"tier '{tn}': 'time_cap.{key}' not in closed "
                        f"time_cap key set"
                    )
            if "max_multiplier" in cap_cfg:
                cap = _as_number(cap_cfg["max_multiplier"])
                if cap is None or cap < _MIN_TIME_CAP:
                    errors.append(
                        f"tier '{tn}': 'time_cap.max_multiplier' must be a "
                        f"number >= 1.0"
                    )

    if "time_policy" in tier:
        policy = tier["time_policy"]
        if not isinstance(policy, dict):
            errors.append(f"tier '{tn}': 'time_policy' must be a mapping")
        else:
            for key in policy:
                if key not in _TIME_POLICY_KEYS:
                    errors.append(
                        f"tier '{tn}': 'time_policy.{key}' not in closed "
                        f"time_policy key set"
                    )
            if "avoid_peak" in policy and not _is_name_list(policy["avoid_peak"]):
                errors.append(
                    f"tier '{tn}': 'time_policy.avoid_peak' must be a list of "
                    f"provider names"
                )
            if "prefer" in policy and not _is_name_list(policy["prefer"]):
                errors.append(
                    f"tier '{tn}': 'time_policy.prefer' must be a list of model names"
                )

    return errors


def _lint_price_windows(tier: Dict[str, Any]) -> List[str]:
    """Validate every ``price_windows`` list this tier declares in YAML.

    Delegated to ``capabilities.price_window_diagnostics`` — the same function the
    registry self-check uses — because ``price_windows`` is overridable per elo
    and a second implementation of the window rules is a second set of answers.
    Its strings are already shaped ``model '<id>': <defect>``, so they are
    appended verbatim. Overlapping windows are a HARD error here: with two
    matching multipliers the winner would be an accident of list order.
    """
    fn = getattr(_caps, "price_window_diagnostics", None) if _caps else None
    if not callable(fn):
        return []
    errors: List[str] = []
    for hop in _tier_chain(tier):
        if "price_windows" not in hop:
            continue
        model = hop.get("model")
        try:
            errors.extend(
                fn(model if isinstance(model, str) else "", hop["price_windows"])
            )
        except (AttributeError, TypeError, ValueError):
            continue
    return errors


def _lint_global_price_windows(config: Dict[str, Any]) -> List[str]:
    """Hard-error checks for the top-level ``price_windows`` overlay block.

    The overlay is a write target (it sits in :data:`service._HOT_KEYS`), so its
    defects are REFUSED here rather than applied half-way: a malformed window
    must never become a silent multiplier. The window-shape rules are delegated
    to ``capabilities.price_window_diagnostics`` — the same validator the
    registry is held to — and the confirmation date to
    ``capabilities.verified_date_diagnostics``, so the overlay can never accept
    a window or a date the registry would refuse. Two spellings are accepted,
    mirroring :func:`_inject_overlay_windows`: the plain list and the extended
    ``{price_windows: [...], price_windows_verified: 'YYYY-MM-DD'}`` form. Absent
    (None) is legal and clean — an operator with no overlay has nothing to lint.
    """
    overlay = config.get("price_windows")
    if overlay is None:
        return []
    if not isinstance(overlay, dict):
        return ["top-level price_windows must be a mapping of model id -> windows"]
    errors: List[str] = []
    windows_fn: Any = getattr(_caps, "price_window_diagnostics", None) if _caps else None
    verified_fn: Any = getattr(_caps, "verified_date_diagnostics", None) if _caps else None
    for model, entry in overlay.items():
        if isinstance(entry, dict):
            # Extended form. A mapping without 'price_windows' is a bare window or
            # a typo — neither is the documented shape, so it is refused rather
            # than silently treated as flat pricing.
            if "price_windows" not in entry:
                errors.append(
                    f"model '{model}': overlay entry must be a list of windows "
                    f"or a mapping with 'price_windows'"
                )
                continue
            windows = entry["price_windows"]
            verified = entry.get("price_windows_verified")
            unknown = set(entry) - {"price_windows", "price_windows_verified"}
            for field in sorted(unknown, key=str):
                errors.append(f"model '{model}': unrecognized overlay field '{field}'")
        else:
            windows = entry
            verified = None
        if callable(windows_fn):
            errors.extend(windows_fn(model, windows))
        if verified is not None and callable(verified_fn):
            errors.extend(verified_fn(model, verified))
    return errors


def _time_warnings(
    tn: str,
    tier: Dict[str, Any],
    chain: List[Dict[str, Any]],
) -> List[str]:
    """Advisory findings about a tier's time knobs — never write-blocking.

    Each one describes a config that is valid and does something other than what
    it reads like:

    * a ``time_cap`` every elo can exceed at some hour is a cap that will bypass
      itself rather than shed cost, and the operator finds out from an invoice;
    * an ``avoid_peak`` provider absent from the tier is a typo or a stale copy —
      the knob is inert either way;
    * ``cheapest_now`` over elos that publish no dollar price cannot compare
      dollars, so it ranks by billing mode alone. That is the documented
      behaviour, not a bug, but it is not what "cheapest" reads like.

    Needs the registry (prices and windows live there); returns [] without it.
    """
    if _caps is None or not chain:
        return []

    warnings: List[str] = []
    cap = _time_cap_of(tier)
    if cap is not None and all(_exceeds_cap_at_some_hour(hop, cap) for hop in chain):
        warnings.append(
            f"tier '{tn}': every elo is in an expensive window at some hour "
            f"— time_cap will bypass"
        )

    policy = tier.get("time_policy")
    if isinstance(policy, dict) and _is_name_list(policy.get("avoid_peak")):
        present = {
            _upstream_provider(hop.get("provider"))
            for hop in chain
            if isinstance(hop.get("provider"), str)
        }
        for name in policy["avoid_peak"]:
            if _upstream_provider(name) not in present:
                warnings.append(
                    f"tier '{tn}': 'time_policy.avoid_peak' names provider "
                    f"'{name}', absent from this tier"
                )

    if _strategy_of(tier) == "cheapest_now" and not any(
        _has_dollar_price(hop) for hop in chain
    ):
        warnings.append(
            f"tier '{tn}': 'cheapest_now' with no priced elo degrades to "
            f"billing_mode rank only"
        )

    return warnings


def _exceeds_cap_at_some_hour(hop: Dict[str, Any], cap: float) -> bool:
    """Whether any window declared for ``hop`` is priced above ``cap``.

    Reads the declared windows rather than sampling 168 hours: the multiplier a
    window carries IS the answer, and an elo with no window is flat-priced at 1.0
    and can never exceed a cap of at least 1.0.
    """
    model = hop.get("model")
    if not isinstance(model, str) or not model:
        return False
    try:
        caps = _caps.capabilities_for(model, _declared_capabilities(hop) or None)
    except (AttributeError, TypeError, ValueError):
        return False
    if not isinstance(caps, dict):
        return False
    windows = caps.get("price_windows")
    if not isinstance(windows, list):
        return False
    for window in windows:
        if not isinstance(window, dict):
            continue
        multiplier = _as_number(window.get("multiplier"))
        if multiplier is not None and multiplier > cap:
            return True
    return False


def _has_dollar_price(hop: Dict[str, Any]) -> bool:
    """Whether this elo publishes a per-token price the router can compare.

    Asked through ``effective_price`` so "priced" means exactly what
    ``cheapest_now`` means by it — never a re-derivation that could disagree.
    """
    model = hop.get("model")
    if not isinstance(model, str) or not model:
        return False
    try:
        priced = _caps.effective_price(model, None, _declared_capabilities(hop) or None)
    except (AttributeError, TypeError, ValueError):
        return False
    return priced is not None


def _upstream_provider(provider: Any) -> str:
    """Provider name normalised for comparison — case and whitespace only.

    Matches ``capabilities.apply_time_policy``'s provider matching, which is
    case-insensitive, so lint agrees with the stage it is warning about. Upstream
    GROUPING is deliberately not applied: ``avoid_peak: [nous]`` names a
    provider's windows, not a rail's redundancy.
    """
    return provider.strip().lower() if isinstance(provider, str) else ""


def _is_name_list(value: Any) -> bool:
    """Whether ``value`` is a non-empty list of non-empty name strings."""
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def _lint_clock_bounds(rid: str, field: str, condition: Any) -> List[str]:
    """Reject a clock clause no legal hour / weekday could ever satisfy.

    ``utc_hour`` is an int in 0..23 and ``utc_weekday`` an int in 0..6 by
    construction, so a bound outside those is a row that never fires — the same
    invisible-dead-row failure the field-name check exists for. ``lt``/``lte`` may
    name the one-past-the-end value (``lt: 24``), because that is how the half-open
    ``[start, end)`` price window an operator is copying from reads, and it is
    perfectly satisfiable.
    """
    if field == "utc_hour":
        limit, label = _UTC_HOUR_MAX, "0..23"
    elif field == "utc_weekday":
        limit, label = _UTC_WEEKDAY_MAX, "0..6"
    else:
        return []
    # Unreachable from lint(), which reports a non-mapping condition as "must be an
    # op map" and `continue`s before it reaches this check. Same treatment as the
    # guard in _lint_tier_shapes: the caller already normalised.
    if not isinstance(condition, dict):  # pragma: no cover - lint() reports it first
        return []

    for op, value in condition.items():
        if op not in _VALID_OPS:
            continue  # already reported as an unknown operator
        ceiling = limit + 1 if op in ("lt", "lte") else limit
        for candidate in (value if isinstance(value, list) else [value]):
            number = _as_number(candidate)
            if number is None or not 0 <= number <= ceiling:
                return [
                    f"rule '{rid}': 'when.{field}' must be bounded to {label}"
                ]
    return []


def _strategy_of(output: Dict[str, Any]) -> str:
    """Declared fallback strategy, defaulting to sequential."""
    strategy = output.get("fallback_strategy", _DEFAULT_FALLBACK_STRATEGY)
    if not isinstance(strategy, str):
        return _DEFAULT_FALLBACK_STRATEGY
    return strategy


def _pin_primary_of(output: Dict[str, Any]) -> bool:
    """Declared pin_primary, defaulting to True (keep the tier's own elo first)."""
    pin = output.get("pin_primary", True)
    return pin if isinstance(pin, bool) else True


def _time_cap_of(output: Dict[str, Any]) -> Optional[float]:
    """Declared price ceiling as a float, or None for "no cap".

    Reads the documented ``{max_multiplier: N}`` mapping. A bare number is
    tolerated because this function is handed whatever a caller assembled, and
    coercing here costs less than a TypeError in the request path — but a bare
    number is NOT a spelling a tier can use: ``_resolve_tiers`` carries
    ``time_cap`` only when it is a mapping, and lint refuses ``time_cap: 1.5``
    outright. Fail-closed in both directions, deliberately.

    An unusable value is no cap at all: lint refuses it at the write gate, and a
    cost control that cannot be parsed must not become a cost control of zero.
    """
    raw = output.get("time_cap")
    if isinstance(raw, dict):
        raw = raw.get("max_multiplier")
    return _as_number(raw)


def _tier_floor_of(output: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The tier's requirement floor, reduced to what can only TIGHTEN.

    ``requirements`` is documented as a floor, and a floor can raise a
    requirement, never lower one. A boolean requirement of False cannot tighten
    anything — ``satisfies()`` only ever constrains on a True requirement — but
    ``derive_requirements`` unions the floor by OVERWRITING, so passing
    ``{vision: false}`` through would erase a signal-derived ``vision: True`` and
    silently let a screenshot route to a blind model. Falsy booleans are therefore
    dropped here, where the floor is assembled, rather than trusted downstream.

    ``min_context`` needs no guard: ``derive_requirements`` takes the MAX of the
    two, which is already floor semantics.
    """
    floor = output.get("requirements")
    if not isinstance(floor, dict):
        return None
    keys = _requirement_keys()
    tightening = {
        key: value
        for key, value in floor.items()
        if key in keys and not (key in _BOOL_REQUIREMENT_KEYS and not value)
    }
    return tightening or None


def _effective_strategy(
    declared: str,
    rng: Optional[random.Random],
    when: Optional["datetime"],
) -> Tuple[str, str]:
    """Return (strategy that can actually run, degrade reason or "").

    ``order_chain`` degrades to sequential rather than reaching for global
    randomness or the wall clock, which is the right BEHAVIOUR and the wrong
    REPORT: a sequential chain labelled "random" is indistinguishable from a bug,
    and the caller who forgot to inject never learns they forgot. So the degrade
    is computed here, on the same conditions, and reported beside the plan.
    """
    if declared not in _fallback_strategies():
        return (
            _DEFAULT_FALLBACK_STRATEGY,
            f"'{declared}' is not a known fallback strategy",
        )
    if declared == "random" and rng is None:
        return (
            _DEFAULT_FALLBACK_STRATEGY,
            "no rng was injected, so the tail was not shuffled",
        )
    if declared == "cheapest_now" and _clock_parts(when) is None:
        return (
            _DEFAULT_FALLBACK_STRATEGY,
            "no clock was injected, so prices could not be compared",
        )
    return declared, ""


def _order_chain(
    chain: List[Dict[str, Any]],
    strategy: str,
    pin_primary: bool,
    rng: Optional[random.Random],
    when: Optional["datetime"],
) -> List[Dict[str, Any]]:
    """capabilities.order_chain, with the clock only when it accepts one."""
    if _ORDER_CHAIN_ACCEPTS_WHEN:
        return _caps.order_chain(
            chain, strategy=strategy, pin_primary=pin_primary, rng=rng, when=when
        )
    return _caps.order_chain(
        chain, strategy=strategy, pin_primary=pin_primary, rng=rng
    )


def _apply_time_cap(
    chain: List[Dict[str, Any]],
    cap: Optional[float],
    when: Optional["datetime"],
) -> Dict[str, Any]:
    """Run the price-ceiling stage, or return the no-op result.

    No clock, no declared cap, or no usable stage function means the chain passes
    through untouched with empty diagnostics. A stage that raises degrades the
    SAME way — the time layer is a cost control, so its failure mode must be
    "no cost control", never "no route" and never "no plan".

    A stage that returns an empty chain is overridden with the input: never
    emptying the chain is the invariant the cap's own bypass exists to hold, and
    it is cheap to hold it twice.
    """
    neutral = {"chain": list(chain), "capped": [], "bypassed": False}
    fn = getattr(_caps, "apply_time_cap", None) if _caps else None
    if cap is None or when is None or not callable(fn):
        return neutral
    try:
        result = fn(chain, cap, when=when)
    except (AttributeError, TypeError, ValueError, KeyError):
        return neutral
    if not isinstance(result, dict):
        return neutral
    capped_chain = result.get("chain")
    usable = isinstance(capped_chain, list) and bool(capped_chain)
    return {
        "chain": list(capped_chain) if usable else list(chain),
        "capped": [
            item for item in (result.get("capped") or []) if isinstance(item, dict)
        ],
        "bypassed": bool(result.get("bypassed", False)),
    }


def _apply_time_policy(
    chain: List[Dict[str, Any]],
    policy: Any,
    when: Optional["datetime"],
) -> Dict[str, Any]:
    """Run the avoid_peak / prefer stage, or return the no-op result.

    Same degrade contract as :func:`_apply_time_cap`. The returned chain must be a
    PERMUTATION of the input — this stage moves elos, it never removes one — so a
    result of a different length is discarded in favour of the input rather than
    trusted.
    """
    neutral = {"chain": list(chain), "demoted": [], "promoted": [], "peak_priced": []}
    fn = getattr(_caps, "apply_time_policy", None) if _caps else None
    if when is None or not isinstance(policy, dict) or not callable(fn):
        return neutral
    try:
        result = fn(chain, policy, when=when)
    except (AttributeError, TypeError, ValueError, KeyError):
        return neutral
    if not isinstance(result, dict):
        return neutral
    moved = result.get("chain")
    if not isinstance(moved, list) or len(moved) != len(chain):
        return neutral
    return {
        "chain": list(moved),
        "demoted": [m for m in (result.get("demoted") or []) if isinstance(m, str)],
        "promoted": [m for m in (result.get("promoted") or []) if isinstance(m, str)],
        # POSITION vs PRICE, carried separately on purpose. ``demoted`` names only
        # what this call actually moved later; ``peak_priced`` names every elo
        # ``avoid_peak`` matched that is inside a dearer window, whether or not
        # moving it changed anything. Dropping the second one here is how the
        # console ended up with no way to say "these are charging more" — the
        # very distinction the split exists to express.
        "peak_priced": [
            m for m in (result.get("peak_priced") or []) if isinstance(m, str)
        ],
    }


def _multipliers_for(
    chain: List[Dict[str, Any]],
    capped: List[Dict[str, Any]],
    when: Optional["datetime"],
) -> Dict[str, float]:
    """model -> the price multiplier this plan was made on, at ``when``.

    Empty without a clock: 1.0 for everything would be a claim about prices that
    nobody checked, and the console needs to be able to tell "the planner saw the
    base rate" from "the planner had no clock".

    Capped elos are included — they are not in the chain, and their multiplier is
    the number that explains why. Their own reported multiplier wins, because that
    is the value the cap decided on.
    """
    if _clock_parts(when) is None:
        return {}
    fn = getattr(_caps, "price_multiplier", None) if _caps else None
    if not callable(fn):
        return {}

    multipliers: Dict[str, float] = {}
    for item in capped or []:
        model = item.get("model") if isinstance(item, dict) else None
        value = _as_number(item.get("multiplier")) if isinstance(item, dict) else None
        if isinstance(model, str) and model and value is not None:
            multipliers.setdefault(model, value)
    for entry in chain or []:
        if not isinstance(entry, dict):
            continue
        model = entry.get("model")
        if not isinstance(model, str) or not model or model in multipliers:
            continue
        try:
            value = _as_number(fn(model, when, entry))
        except (AttributeError, TypeError, ValueError):
            continue
        if value is not None:
            multipliers[model] = value
    return multipliers


def _clock_parts(when: Any) -> Optional[Tuple[int, int]]:
    """(utc_hour, utc_weekday) for an injected clock, or None for "no clock".

    Mirrors ``capabilities._utc_parts`` deliberately: the plan's reported hour and
    the multipliers applied to it must come from the same reading of the same
    clock, or the trace explains an order that was decided at another hour.

    An AWARE datetime is converted to UTC, a NAIVE one is taken to be UTC already
    — both via ``utctimetuple()``, which is what lets this module answer the
    question WITHOUT importing datetime and therefore without being able to read
    a clock. Anything that cannot supply both an hour and a weekday (a bare
    ``date``, a ``time``, a string decoded from a trace) is no clock at all rather
    than an exception: a diagnostic must not break the request path.
    """
    if when is None:
        return None
    try:
        stamp = getattr(when, "utctimetuple", None)
        weekday_of = getattr(when, "weekday", None)
        if not callable(stamp) or not callable(weekday_of):
            return None
        if getattr(when, "hour", None) is None:
            # A stdlib ``date`` never gets this far — it has no ``utctimetuple``
            # at all — so what this catches is a DUCK-TYPED clock: an object a
            # caller assembled (a trace decoder, a sibling deployment's shim)
            # that answers the two calls above and still cannot name an hour.
            # No hour, no plan hour: "no clock", never hour 0.
            return None
        parts = stamp()
        hour = int(parts.tm_hour)
        weekday = int(parts.tm_wday)
    except (AttributeError, TypeError, ValueError, OSError, OverflowError):
        return None
    if not 0 <= hour <= _UTC_HOUR_MAX or not 0 <= weekday <= _UTC_WEEKDAY_MAX:
        return None
    return hour, weekday


def _clock_keys(when: Optional["datetime"]) -> Dict[str, Any]:
    """The plan's clock keys: the hour it was planned at, or "time-agnostic".

    ``utc_hour``/``utc_weekday`` are OMITTED when there is no clock rather than
    set to None. A JSON consumer reads ``Number(null)`` as ``0``, so a null hour
    renders as midnight — a plan that silently claims an hour it never saw is
    exactly what ``time_agnostic`` exists to prevent.
    """
    parts = _clock_parts(when)
    if parts is None:
        return {"time_agnostic": True}
    return {"time_agnostic": False, "utc_hour": parts[0], "utc_weekday": parts[1]}


def _as_number(value: Any) -> Optional[float]:
    """A real number as float, or None. Booleans are not numbers here.

    ``True`` is an int in Python, so a config saying ``max_multiplier: true``
    would otherwise become a cap of 1.0 — a value the operator never wrote.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _build_chain(output: Dict[str, Any]) -> List[Dict[str, Any]]:
    """[{model, provider, ...declared}] + output['fallback'] — new dicts only."""
    chain: List[Dict[str, Any]] = []
    model = output.get("model")
    if isinstance(model, str) and model:
        primary: Dict[str, Any] = {"model": model, "provider": output.get("provider")}
        declared = output.get("declared_capabilities")
        if isinstance(declared, dict):
            primary.update(declared)
        chain.append(primary)
    fallback = output.get("fallback")
    if isinstance(fallback, list):
        for hop in fallback:
            if isinstance(hop, dict) and hop.get("model"):
                chain.append(dict(hop))
    return chain


def _all_clauses_match(
    when: Dict[str, Any],
    features: Dict[str, Any],
    blocked_model: bool,
) -> bool:
    """Return True when ALL when clauses hold against features."""
    if not when:
        return False

    for field, condition in when.items():
        # A clause lint REFUSES must not raise through the request path. `when:
        # {has_code: true}` — the op map an operator forgets to write — reaches
        # `condition.items()` as a bool and raised AttributeError out of match(),
        # i.e. lint said "must be an op map" while the engine said 500. A
        # non-mapping condition names no operator, so it holds for nothing and the
        # row is dead: exactly what lint reports it as. Guessing an op instead
        # (reading the bare value as `eq`) is the one answer worse than both,
        # because it would route on a row the write gate calls invalid.
        if not isinstance(condition, dict):
            return False

        # Special case: blocked_model injected boolean (never in features)
        if field == "blocked_model":
            # Evaluate the AUTHOR'S operators, not a hardcoded eq. Re-reading the
            # value under condition["eq"] discarded whatever op was written and
            # defaulted the target to True, so `{ne: true}` - the natural "only
            # when NOT blocked" guard - evaluated as `eq true` and was False
            # exactly when the model was healthy. Such a rule is dead on the live
            # path (rules only run with blocked_model=False; a block returns
            # early at the veto), while _matching_clauses reported a chip for it,
            # so /explain claimed a clause matched that the engine had rejected.
            if not all(
                _eval_clause(op, blocked_model, target)
                for op, target in condition.items()
            ):
                return False
            continue

        if field not in features:
            return False

        feat_val = features[field]
        for op, target in condition.items():
            if not _eval_clause(op, feat_val, target):
                return False

    return True


def _eval_clause(op: str, actual: Any, target: Any) -> bool:
    """Evaluate a single (op, target) against an actual value."""
    try:
        if op == "eq":
            return actual == target
        elif op == "ne":
            return actual != target
        elif op == "in":
            if isinstance(target, list):
                return actual in target
            return actual == target
        elif op == "nin":
            if isinstance(target, list):
                return actual not in target
            return actual != target
        elif op == "gt":
            return float(actual) > float(target)
        elif op == "gte":
            return float(actual) >= float(target)
        elif op == "lt":
            return float(actual) < float(target)
        elif op == "lte":
            return float(actual) <= float(target)
        elif op == "contains":
            if isinstance(actual, list):
                return str(target).lower() in [str(a).lower() for a in actual]
            return str(target).lower() in str(actual).lower()
        elif op == "starts_with":
            return str(actual).lower().startswith(str(target).lower())
        elif op == "ends_with":
            return str(actual).lower().endswith(str(target).lower())
        elif op == "matches":
            return bool(re.search(str(target), str(actual)))
        return False
    except (TypeError, ValueError):
        return False


def _resolve_tiers(output: Dict[str, Any], tiers: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Resolve Tn aliases in output['model'] against Table 2.

    Besides model/provider/fallback, a resolved output carries the tier's
    routing knobs when the tier DECLARES them: 'fallback_strategy',
    'pin_primary', 'billing_mode', 'requirements' (the per-tier capability
    floor), 'time_cap' and 'time_policy' (the per-tier clock knobs) and
    'declared_capabilities' (capability keys declared on the tier itself, used as
    the primary hop's overrides). Absent keys are not materialised — a tier that
    declares none of them resolves byte-identically to the pre-capability engine
    — and the documented defaults (fallback_strategy "sequential", pin_primary
    True, no cap, no policy) are applied by plan_chain, which reads them through
    _strategy_of / _pin_primary_of / _time_cap_of.

    Returns a new dict — never mutates the input: the time knobs are COPIED, so a
    plan can never write back into the tier table it was resolved from.
    """
    result = dict(output)
    model = result.get("model")
    if isinstance(model, str) and model in tiers:
        tier = tiers[model]
        result["model"] = tier.get("model", model)
        if "provider" in tier:
            result["provider"] = tier["provider"]
        fallback = tier.get("fallback")
        if isinstance(fallback, list):
            result["fallback"] = [
                dict(target) for target in fallback if isinstance(target, dict)
            ]

        # Carried, not validated: lint() is the gate that rejects a bad strategy
        # or a non-boolean pin_primary. Resolution stays permissive (and
        # normalising) so a stale config still routes.
        if "fallback_strategy" in tier:
            result["fallback_strategy"] = _strategy_of(tier)
        if "pin_primary" in tier:
            result["pin_primary"] = _pin_primary_of(tier)

        billing_mode = tier.get("billing_mode")
        if isinstance(billing_mode, str):
            result["billing_mode"] = billing_mode

        requirements = tier.get("requirements")
        if isinstance(requirements, dict):
            result["requirements"] = {
                key: value
                for key, value in requirements.items()
                if key in _requirement_keys()
            }

        time_cap = tier.get("time_cap")
        if isinstance(time_cap, dict):
            result["time_cap"] = dict(time_cap)

        time_policy = tier.get("time_policy")
        if isinstance(time_policy, dict):
            result["time_policy"] = {
                key: list(value) if isinstance(value, list) else value
                for key, value in time_policy.items()
            }

        declared = _declared_capabilities(tier)
        if declared:
            result["declared_capabilities"] = declared
    return result


def _matching_clauses(
    when: Dict[str, Any],
    features: Dict[str, Any],
    blocked_model: bool = False,
) -> Dict[str, Any]:
    """Return the subset of when clauses that matched.

    blocked_model is not a feature: it is computed by the caller and injected, so a
    plain `field in features` test silently dropped it. The console renders this dict
    as the "because ..." chips, so dropping it meant a two-clause rule explained
    itself with one chip and a blocked_model-only rule gave no reason at all.
    _all_clauses_match evaluates the same operators against the same value, so
    a chip appears here exactly when that clause held there. They diverged twice:

      * the matcher hardcoded `eq` while this function honoured the author's op, so
        /explain showed a chip for a `{ne: true}` clause the engine had rejected;
      * and this function used ANY over the operators on a field while the matcher
        uses ALL. A single-operator condition cannot tell the two apart, which is
        why the agreement test did not — but `{utc_hour: {gte: 6, lt: 10}}`, the
        idiom router.example.yaml teaches, can: measured at hour 5 and hour 14,
        `_all_clauses_match` said False while this returned the clause as MATCHED.
        The console renders that as a "because utc_hour ≥ 6 and < 10" chip on a
        rule that did not fire.

    ALL, therefore, on both sides.
    """
    matched: Dict[str, Any] = {}
    for field, condition in when.items():
        # Same guard as _all_clauses_match, and it has to be the same VERDICT: a
        # clause the engine cannot evaluate must never come back as a chip that
        # explains a match. Skipped rather than returned on, because this function
        # reports per clause — and a rule with such a clause never matched anyway.
        if not isinstance(condition, dict):
            continue
        if field == "blocked_model":
            if all(_eval_clause(op, blocked_model, target)
                   for op, target in condition.items()):
                matched[field] = condition
            continue
        if field in features:
            if all(_eval_clause(op, features[field], target)
                   for op, target in condition.items()):
                matched[field] = condition
    return matched


def _cause_labeller() -> Optional[Callable[[Any, Dict[str, Any]], str]]:
    """The ONE rule-id → cause labeller: ``adapter._cause_from_rule``.

    Imported at CALL time, not at module scope, because ``router.adapter``
    imports this module at ITS module scope — the cycle would leave whichever of
    the two was imported second reading a half-built first. That is the whole
    reason this is a function and not another ``try: from router import ...``
    block at the top of the file. Nothing is memoised: a warm import is a
    sys.modules lookup, and a cached handle would be module STATE this module
    promises not to keep (and would freeze a transient failure forever).

    Fetching the labeller rather than just its ``_RULE_ID_CAUSES`` table is
    deliberate: the table AND the substring heuristic that backs it up are both
    drift surfaces, and this module had a copy of the heuristic that already
    lagged the original by two probes. One function means one answer, including
    when the two files are a version apart (this plugin is deployed by copy) — a
    version-skewed adapter that predates the table still agrees with itself.

    None means "no labeller reachable" — the module is absent, too old to export
    the name, or mid-import. Returned rather than raised: the caller's job is to
    label a decision, and a diagnostic must not raise through /explain.

    Relative first, absolute second, for the reason the module-scope registry
    imports resolve that way. Resolving only the absolute name made None the
    NORMAL answer under Hermes's ``hermes_plugins.<slug>`` shape rather than the
    exotic one this docstring describes, and that is the precise disagreement the
    delegation exists to prevent: every rule-keyed cause degraded to
    ``default_fallthrough`` on the surfaces that DISPLAY a decision while the
    adapter — which reaches this module fine from its own module scope — went on
    recording ``keyword_match`` / ``size_rule`` / ``hard_rule`` for the same
    match on the path that RUNS it.
    """
    try:
        from .adapter import _cause_from_rule
    except ImportError:  # pragma: no cover - flat layout, absent, or mid-import
        try:
            from router.adapter import _cause_from_rule
        except ImportError:  # pragma: no cover - adapter absent, or mid-import
            return None
    if not callable(_cause_from_rule):  # pragma: no cover - not a function
        return None
    return _cause_from_rule


def _determine_cause(rule_id: Any, output: Dict[str, Any]) -> str:
    """Map rule id + output to a closed-set cause label.

    This is the label the surface that DISPLAYS a decision shows — explain(),
    RouterService.explain, the sidecar's /explain, the dashboard. The path that
    RUNS the decision labels the same rule match with
    ``adapter._cause_from_rule``, so the rule-id axis is NOT decided here: it is
    delegated to that exact function (:func:`_cause_labeller`), whose
    ``_RULE_ID_CAUSES`` is the one table.

    It used to be decided here, by a private copy of the heuristic, and the copy
    drifted: measured on the shipped policy, the running path recorded
    `vision-required` as ``keyword_match`` while every /explain surface reported
    ``classifier`` for the same decision, and `huge-context-read`,
    `cross-file-or-protocol` and `standard-implementation` disagreed the same way
    — four of eight shipped rows labelled two different things depending on which
    surface an operator asked.

    Two labels stay local because they are keyed on the OUTPUT, not on the rule
    id, and each matches what the running path records for that output:

      * ``deny`` is the blocklist veto. Decided first, ahead of the delegation,
        so the veto label — the one that must never be wrong — does not depend on
        another module being importable. The labeller decides it identically.
      * ``action: classify`` means the pure core delegated the model choice to
        Stage 1, and ``classifier`` is exactly what ``adapter.route()`` records
        for such a route. Labelling it from the rule id here instead would put
        /explain at odds with the recorded cause of every `review-request` route.

    ``rule_id`` is whatever the policy author wrote and is NOT coerced here. YAML
    yields an int for `id: 7` and the classifier path passes None; the labeller
    owns that coercion (``.lower()`` on an int raised AttributeError, which
    service.explain does not catch — it catches ValueError — so /explain died
    uncaught inside the code whose only job is to explain a route).

    With no labeller reachable the id cannot be labelled and this returns
    ``default_fallthrough``, the closed-set member for "no rule-keyed cause". In
    that state nothing disagrees, because the module that records the other half
    of the pair is the one that failed to import. The cause set is CLOSED on
    purpose: ``decision_log.record()`` coerces an unknown cause to
    ``fail_safe_strong``, so inventing a cause string here would relabel healthy
    routes as fail-safe.
    """
    if output.get("deny"):
        return "blocklist_veto"
    if output.get("action") == "classify":
        return "classifier"
    labeller = _cause_labeller()
    if labeller is None:
        return "default_fallthrough"
    return labeller(rule_id, output)


def _is_shadowed(
    earlier_when: Dict[str, Any],
    later_when: Dict[str, Any],
) -> bool:
    """Return True when ``later_when`` can never fire because the earlier row will.

    First-match semantics: the later row is dead when EVERY feature vector that
    matches it also matches the earlier row. Two things have to hold.

      1. Every field the earlier row constrains is constrained by the later row
         too. Otherwise a vector matching the later row is free to violate the
         earlier row on a field the later row never mentions.
      2. On each of those fields the later row's condition is CONTAINED in the
         earlier row's: every value the later admits, the earlier admits as well
         (see :func:`_condition_contains`).

    Containment is decided per operator family, and the undecidable answer is
    False, never True. That asymmetry is deliberate. ``lint()`` is the write gate,
    so a FALSE shadow refuses a legitimate config and strands the operator outside
    the guarded path — the exact failure the gate exists to prevent. A MISSED
    shadow leaves a dead row that is visible in the file, visible in ``explain()``
    as a rule that never matches, and visible in the decision log as a rule id
    with zero hits. Hence: two disjoint context thresholds shadow nothing and must
    ship, while a genuinely contained interval is still reported.
    """
    # Unreachable from lint(), whose shadow loop isinstance-guards both `when`
    # mappings (a non-mapping `when` is already a reported error) before calling.
    # Kept for the direct callers in the test suite and for the conservative
    # contract this whole function states: undecidable is False, never True.
    if not isinstance(earlier_when, dict) or not isinstance(later_when, dict):
        return False  # pragma: no cover - lint() isinstance-guards both first
    if not earlier_when or not later_when:
        return False
    if not set(earlier_when).issubset(set(later_when)):
        return False
    return all(
        _condition_contains(earlier_when[field], later_when[field])
        for field in earlier_when
    )


def _condition_contains(earlier: Any, later: Any) -> bool:
    """Whether every value satisfying ``later`` also satisfies ``earlier``.

    Decidable families only — anything else is False:

      * identical conditions: trivially contained;
      * comparisons (gt/gte/lt/lte) with numeric bounds: interval containment, so
        ``{gt: 400000}`` contains ``{gt: 800000}`` (the later row is genuinely
        dead) while ``{gt: 800000}`` and ``{lt: 2000}`` contain nothing of each
        other (both must ship);
      * membership (eq/in): set containment;
      * exclusion (ne/nin): excluding LESS admits more, so the earlier row's
        excluded set must be a subset of the later's;
      * an earlier exclusion against a later membership: contained when none of
        the later row's values are excluded.

    Substring and regex operators (contains/starts_with/ends_with/matches) describe
    regions this module does not model — comparing them would mean deciding
    whether one regex's language contains another's — so only the identical case
    above ever decides them. An empty condition constrains nothing beyond the
    field's presence, which the later row also requires, so it contains anything.
    """
    if not isinstance(earlier, dict) or not isinstance(later, dict):
        return False
    if not earlier:
        return True
    if not later:
        return False
    if earlier == later:
        return True

    earlier_ops, later_ops = set(earlier), set(later)
    if earlier_ops <= _COMPARISON_OPS and later_ops <= _COMPARISON_OPS:
        return _interval_contains(earlier, later)
    if earlier_ops <= _INCLUDE_OPS and later_ops <= _INCLUDE_OPS:
        outer, inner = _include_set(earlier), _include_set(later)
        return outer is not None and inner is not None and inner <= outer
    if earlier_ops <= _EXCLUDE_OPS and later_ops <= _EXCLUDE_OPS:
        outer, inner = _exclude_set(earlier), _exclude_set(later)
        return outer is not None and inner is not None and outer <= inner
    if earlier_ops <= _EXCLUDE_OPS and later_ops <= _INCLUDE_OPS:
        excluded, admitted = _exclude_set(earlier), _include_set(later)
        return (
            excluded is not None
            and admitted is not None
            and not (excluded & admitted)
        )
    return False


def _interval_contains(earlier: Dict[str, Any], later: Dict[str, Any]) -> bool:
    """Interval containment for two comparison-only conditions on one field."""
    outer = _bounds_of(earlier)
    inner = _bounds_of(later)
    if outer is None or inner is None:
        return False
    return _lower_covers(outer[0], inner[0]) and _upper_covers(outer[1], inner[1])


def _bounds_of(condition: Dict[str, Any]) -> Optional[Tuple[Any, Any]]:
    """((lower, strict) | None, (upper, strict) | None), or None if undecidable.

    ``strict`` marks an EXCLUSIVE bound (gt/lt). Two bounds on the same side are
    reduced to the tighter one, so ``{gt: 10, gte: 20}`` is ``> 20``. A non-numeric
    bound makes the interval unknowable: ``gt`` coerces with ``float()`` at match
    time, so ``{gt: "200k"}`` is a row that never matches rather than a bound this
    function may invent a number for.
    """
    lower: Optional[Tuple[float, bool]] = None
    upper: Optional[Tuple[float, bool]] = None
    for op, raw in condition.items():
        value = _as_number(raw)
        if value is None:
            return None
        if op in ("gt", "gte"):
            lower = _tighter_lower(lower, (value, op == "gt"))
        elif op in ("lt", "lte"):
            upper = _tighter_upper(upper, (value, op == "lt"))
        else:
            # Unreachable from _condition_contains, the only caller: it reaches
            # _interval_contains only when BOTH conditions' op sets are subsets of
            # _COMPARISON_OPS, which is exactly the two arms above. Kept so the
            # function is total for a condition it is handed directly.
            return None  # pragma: no cover - caller gates on _COMPARISON_OPS
    return lower, upper


def _tighter_lower(
    current: Optional[Tuple[float, bool]],
    candidate: Tuple[float, bool],
) -> Tuple[float, bool]:
    """The lower bound that admits FEWER values: the larger, exclusive on a tie."""
    if current is None:
        return candidate
    if candidate[0] > current[0]:
        return candidate
    if candidate[0] == current[0] and candidate[1]:
        return candidate
    return current


def _tighter_upper(
    current: Optional[Tuple[float, bool]],
    candidate: Tuple[float, bool],
) -> Tuple[float, bool]:
    """The upper bound that admits FEWER values: the smaller, exclusive on a tie."""
    if current is None:
        return candidate
    if candidate[0] < current[0]:
        return candidate
    if candidate[0] == current[0] and candidate[1]:
        return candidate
    return current


def _lower_covers(
    outer: Optional[Tuple[float, bool]],
    inner: Optional[Tuple[float, bool]],
) -> bool:
    """Whether ``outer``'s lower bound admits everything ``inner``'s does."""
    if outer is None:
        return True  # unbounded below admits everything
    if inner is None:
        return False  # inner admits arbitrarily small values, outer does not
    if inner[0] != outer[0]:
        return inner[0] > outer[0]
    # Same value: outer must not be the stricter of the two.
    return not (outer[1] and not inner[1])


def _upper_covers(
    outer: Optional[Tuple[float, bool]],
    inner: Optional[Tuple[float, bool]],
) -> bool:
    """Whether ``outer``'s upper bound admits everything ``inner``'s does."""
    if outer is None:
        return True
    if inner is None:
        return False
    if inner[0] != outer[0]:
        return inner[0] < outer[0]
    return not (outer[1] and not inner[1])


def _include_set(condition: Dict[str, Any]) -> Optional[set]:
    """The values an eq/in condition ADMITS, or None when undecidable.

    Several include ops on one field are ANDed, so they intersect.
    """
    admitted: Optional[set] = None
    for _op, raw in condition.items():
        values = _hashable_set(raw)
        if values is None:
            return None
        admitted = values if admitted is None else admitted & values
    return admitted


def _exclude_set(condition: Dict[str, Any]) -> Optional[set]:
    """The values a ne/nin condition REJECTS, or None when undecidable."""
    rejected: set = set()
    for _op, raw in condition.items():
        values = _hashable_set(raw)
        if values is None:
            return None
        rejected |= values
    return rejected


def _hashable_set(raw: Any) -> Optional[set]:
    """A set of the operand's values, or None when they cannot form one.

    A bare (non-list) operand is one value: ``in``/``nin`` treat a scalar as an
    equality test, and this mirrors that so lint reasons about what matching
    actually does. Unhashable operands (a nested mapping or list) are undecidable
    rather than an exception.
    """
    try:
        return set(raw) if isinstance(raw, list) else {raw}
    except TypeError:
        return None
