"""Decision log — one greppable cause= line per turn.

Pure in-memory bookkeeping: no IO, no model calls. (The durable subclass adds
the only IO, and guards it.)

Closed cause set — the only valid strings:
  blocklist_veto, breaker_cooldown, keyword_match, size_rule,
  has_code_rule, hard_rule, classifier, session_pin, default_fallthrough,
  fail_safe_strong, profile_ignored, role_out_of_scope, selection_vetoed,
  unknown_cause

A cause outside the closed set is recorded AS ``unknown_cause`` — it used to
be coerced to ``fail_safe_strong``, which painted an inventing caller as the
router's WORST real outcome at the exact spot where 18 of 40 live decisions
already read ``fail_safe``. An unknown cause is a programming error upstream;
the trace should say it is unknown, not that the fail-safe fired.

Chain-plan persistence (additive, backward compatible):
  ``record(..., chain_plan=...)`` attaches the capability/fallback chain plan
  produced by ``rules.plan_chain`` so a trace can be replayed and audited —
  which elos survived the capability filter, which were rejected and why, and
  in what order they will be tried. Two properties are load-ballasting here:
    * OLD entries have no ``chain_plan`` key at all — the box has live traces in
      routes.jsonl written by the current code, and a reader that raises on them
      takes the console down. :func:`chain_plan_of` therefore returns an empty
      default for missing AND for corrupt values, and never raises.
    * routes.jsonl is size-bounded and rotated, so the entry is bounded:
      ``rejected`` is truncated to :data:`MAX_REJECTED_ENTRIES` and the number
      of dropped rows is reported as ``rejected_truncated``.

  The read-back whitelist covers the PHASE-2 plan too — the time layer's
  ``multipliers``/``capped``/``demoted``/``promoted``/``time_cap*`` keys, the
  strategy-degrade triple, ``pin_primary``, ``unsatisfiable``, the clock keys and
  the blocklist veto's ``blocked``/``blocklist_widened``/``blocklist_bypassed``.
  It has to: :func:`bound_chain_plan` persists whatever the planner produced, so
  a narrower whitelist silently dropped every one of those fields on the way back
  out and the console rendered a phase-1 plan for a phase-2 decision. Anything
  the planner adds and this module does not list is invisible on replay, which is
  a worse failure than a missing key because it looks like data.

Attempted head (additive, and the reason it exists):
  ``output["model"]`` is the DECLARED tier primary — the tier identity the rule
  or the classifier settled on. The model production actually attempts FIRST is
  the head of the planned chain, and after a capability filter, a shuffle or a
  blocklist veto the two differ. A reader of routes.jsonl had no way to tell: a
  vision decision was labelled ``glm-5.3`` while ``gpt-5.6-luna`` is what ran.
  :func:`record` therefore records the planned head alongside it as
  ``output.attempted_model`` / ``output.attempted_provider`` rather than
  redefining ``model``, because other consumers read ``model`` as the tier.

  Writer and reader share ONE definition of "head" — :func:`plan_head_of` — and
  the read side is :func:`attempted_head_of`. That is deliberate: the defect this
  key exists to fix was two surfaces disagreeing about which model ran, so it
  would be absurd to fix it with two implementations of which hop is first.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

# Closed set — the only valid cause values
VALID_CAUSES: set[str] = {
    "blocklist_veto",
    "breaker_cooldown",
    "keyword_match",
    "size_rule",
    "has_code_rule",
    "hard_rule",
    "classifier",
    "session_pin",
    "default_fallthrough",
    "fail_safe_strong",
    # HISTORICAL, no longer produced. Until 2026-08-26 the kanban dispatch path
    # refused a decision whose ``then.profile`` differed from the card's assignee
    # and dropped the model half with it — which is how 135 of 158 measured
    # decisions came to contribute nothing. Kept in the closed set because trace
    # files written before that change still carry it, and a screen that cannot
    # name what it reads is worse than one naming a retired value.
    "profile_ignored",
    # The rule's role half does not apply here: the caller already fixed the role
    # (a kanban card's assignee) and this path cannot change it, so only the model
    # half was applied. NOT a refusal — the decision ran, minus an axis that was
    # never this path's to move. A rule that should not fire for a role at all
    # says so on the input side, ``when: {assignee: {eq: <role>}}``.
    "role_out_of_scope",
    # The selection guard (cost/data-policy) refused the chosen rail AND the
    # fallback chain offered no clean replacement — a denial, not a substitution.
    "selection_vetoed",
    # A cause outside the closed set is recorded AS unknown (never masked as
    # the worst real outcome) so an operator can spot the inventing caller.
    "unknown_cause",
}

# ---------------------------------------------------------------------------
# Chain plan — bounded shape + defensive parse
# ---------------------------------------------------------------------------

# Hard cap on persisted ``rejected`` rows. routes.jsonl is size-bounded and
# rotated, so one pathological chain must not evict everybody else's traces.
MAX_REJECTED_ENTRIES = 8

# Executor outcome records are a separate append-only journal joined to a
# decision by durable_decision_log. ``attempts`` is OPTIONAL on a decision:
# absent means it predates executor instrumentation — a reader must say "não
# instrumentado", never invent "zero tentativas". An explicit [] is different:
# the current executor was present and walked no backend.
_ATTEMPT_OUTCOMES = frozenset({"served", "failed", "skipped"})


def attempts_of(entry: Any) -> Optional[List[Dict[str, Any]]]:
    """Validated attempt records, [] for an instrumented empty run, or None old.

    A bad individual journal row cannot hide the valid rows around it: skip only
    that row. ``served``/``failed``/``skipped`` have mutually exclusive error
    contracts, so consumers never infer a failure from a stray error payload.
    """
    if not isinstance(entry, dict) or "attempts" not in entry:
        return None
    raw = entry.get("attempts")
    if not isinstance(raw, list):
        return None
    valid: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        n = item.get("n")
        duration = item.get("duration_ms")
        started = item.get("started_at")
        outcome = item.get("outcome")
        if (
            not isinstance(n, int) or isinstance(n, bool) or n < 1
            or not isinstance(duration, int) or isinstance(duration, bool) or duration < 0
            or not isinstance(started, (int, float)) or isinstance(started, bool)
            or not isinstance(item.get("model"), str) or not isinstance(item.get("provider"), str)
            or outcome not in _ATTEMPT_OUTCOMES
        ):
            continue
        record = {
            "n": n, "model": item["model"], "provider": item["provider"],
            "started_at": float(started), "duration_ms": duration, "outcome": outcome,
        }
        error = item.get("error")
        if outcome == "failed":
            if not isinstance(error, dict) or not isinstance(error.get("code"), str) or not isinstance(error.get("message"), str):
                continue
            record["error"] = {"code": error["code"], "message": error["message"]}
        elif error is not None:
            continue
        valid.append(record)
    return sorted(valid, key=lambda record: record["n"])

# The persisted chain-plan keys: rules.plan_chain()'s return shape plus the
# truncation counter added here. Built by a factory, never a shared constant —
# the values are mutable and callers are free to mutate what they get back.
#
# ``utc_hour``/``utc_weekday``/``time_cap`` are listed as persisted keys but are
# NOT members of the empty default: plan_chain omits them rather than nulling
# them, because a JSON consumer reads ``Number(null)`` as 0, so a null hour
# renders as midnight and a null cap as a ceiling of 0x. Absence plus
# ``time_agnostic: True`` is unambiguous; a null is not.
CHAIN_PLAN_KEYS: frozenset = frozenset({
    # phase 1
    "chain", "requirements", "rejected", "unknown",
    "bypassed", "strategy", "independent_rails", "rejected_truncated",
    # phase 2 — capability/strategy diagnostics
    "unsatisfiable", "strategy_declared", "strategy_degraded",
    "strategy_degraded_reason", "pin_primary",
    # phase 2 — the time layer
    "time_agnostic", "time_cap_bypassed", "capped", "demoted", "promoted",
    "peak_priced",
    "multipliers", "time_cap", "utc_hour", "utc_weekday",
    # the blocklist veto (adapter._veto_blocked), which runs AFTER the planner
    # and edits the chain the planner produced. Absent unless the veto acted.
    "blocked", "blocklist_widened", "blocklist_bypassed",
})

# Keys omitted from the empty default on purpose. Two different reasons:
#   * the clock/cap trio, because a JSON consumer reads Number(null) as 0, so a
#     defaulted hour renders as midnight and a defaulted cap as a ceiling of 0x;
#   * the blocklist-veto trio, because the veto is a NO-OP on almost every turn
#     and defaulting its keys would rewrite the shape of every historical entry
#     and every clean trace. Absent means "the veto removed nothing", which for a
#     list and two booleans is unambiguous — undefined and [] / false read the
#     same way, unlike a null hour.
_OPTIONAL_CHAIN_PLAN_KEYS: frozenset = frozenset({
    "time_cap", "utc_hour", "utc_weekday",
    "blocked", "blocklist_widened", "blocklist_bypassed",
})


def empty_chain_plan() -> Dict[str, Any]:
    """A fresh, mutable empty chain plan — the default for OLD/corrupt entries.

    A MIRROR of ``rules._empty_chain_plan()`` (and of
    ``service._empty_chain_plan()``) plus this module's own
    ``rejected_truncated`` counter. One shape, because a console branches on
    these keys and cannot see which module produced the plan it was handed;
    ``time_agnostic: True`` in particular is what stops a consumer pricing a
    plan against the BROWSER's hour when no clock was ever involved.
    """
    return {
        "chain": [],
        "requirements": {},
        "rejected": [],
        "unknown": [],
        "bypassed": False,
        "unsatisfiable": [],
        "strategy": "sequential",
        "strategy_declared": "sequential",
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
        "rejected_truncated": 0,
    }


def bound_chain_plan(plan: Any) -> Optional[Dict[str, Any]]:
    """Return a bounded, JSON-safe copy of ``plan``, or None to skip it.

    None means "do not record a chain_plan key" — a non-mapping plan is a
    programming error upstream, and a trace entry is never worth raising over.
    ``rejected`` is truncated to :data:`MAX_REJECTED_ENTRIES`; the count of
    dropped rows lands in ``rejected_truncated`` (0 when nothing was dropped, so
    consumers never have to branch on the key's presence).
    """
    if not isinstance(plan, dict):
        return None

    bounded: Dict[str, Any] = dict(plan)

    rejected = plan.get("rejected")
    if not isinstance(rejected, list):
        # Corrupt/absent rejected list degrades to empty rather than raising.
        bounded["rejected"] = []
        bounded["rejected_truncated"] = 0
        return bounded

    dropped = max(0, len(rejected) - MAX_REJECTED_ENTRIES)
    bounded["rejected"] = list(rejected[:MAX_REJECTED_ENTRIES])
    bounded["rejected_truncated"] = dropped
    return bounded


# Read-back field groups. One table instead of one branch per key, so widening
# the plan is adding a name to a tuple and cannot half-happen: a key that is
# persisted but listed nowhere here is silently dropped on read-back, which is
# the phase-2 defect this table exists to make structurally impossible.
_PLAN_LIST_KEYS = (
    "chain", "rejected", "unknown", "unsatisfiable",
    "capped", "demoted", "promoted", "peak_priced", "blocked",
)
_PLAN_DICT_KEYS = ("requirements", "multipliers", "time_cap")
_PLAN_BOOL_KEYS = (
    "bypassed", "strategy_degraded", "time_agnostic", "time_cap_bypassed",
    "pin_primary", "blocklist_widened", "blocklist_bypassed",
)
# Non-empty strings: an empty strategy name is corrupt, not a choice.
_PLAN_NAME_KEYS = ("strategy", "strategy_declared")
# Free strings: "" is the legitimate "no reason, nothing degraded" value.
_PLAN_TEXT_KEYS = ("strategy_degraded_reason",)
_PLAN_COUNT_KEYS = ("independent_rails", "rejected_truncated")
#: Clock keys and their inclusive valid range — the addendum's, 0=Monday.
_PLAN_CLOCK_RANGES = {"utc_hour": (0, 23), "utc_weekday": (0, 6)}


def chain_plan_of(entry: Any) -> Dict[str, Any]:
    """Read the chain plan out of a recorded entry, defensively.

    Contract (all three branches are exercised by live data on the box):
      * entry written before this feature -> no ``chain_plan`` key -> empty default
      * ``chain_plan`` present but corrupt (string, list, null) -> empty default
      * ``chain_plan`` present and a mapping -> merged over the empty default,
        with each field type-checked individually so ONE bad field cannot
        discard the rest of the plan.

    Entries written by a PHASE-1 writer simply lack the phase-2 keys and read
    back with the documented defaults, which is why widening the whitelist is
    backward compatible: the tolerance is per field, not per entry version.

    ``utc_hour``/``utc_weekday``/``time_cap`` and the blocklist veto's
    ``blocked``/``blocklist_widened``/``blocklist_bypassed`` are the keys that
    stay ABSENT when the persisted plan has no usable value, rather than being
    defaulted — see :data:`_OPTIONAL_CHAIN_PLAN_KEYS` for the two different
    reasons. An out-of-range hour is treated as corrupt for the same reason: a
    consumer prices a plan by its hour, so a plausible-looking wrong hour is
    worse than no hour.

    Never raises.
    """
    plan = empty_chain_plan()
    if not isinstance(entry, dict):
        return plan
    raw = entry.get("chain_plan")
    if not isinstance(raw, dict):
        return plan  # missing (old entry) or corrupt value -> skipped

    for key in _PLAN_LIST_KEYS:
        value = raw.get(key)
        if isinstance(value, list):
            plan[key] = list(value)
    for key in _PLAN_DICT_KEYS:
        value = raw.get(key)
        if isinstance(value, dict):
            plan[key] = dict(value)
    for key in _PLAN_BOOL_KEYS:
        value = raw.get(key)
        if isinstance(value, bool):
            plan[key] = value
    for key in _PLAN_NAME_KEYS:
        value = raw.get(key)
        if isinstance(value, str) and value:
            plan[key] = value
    for key in _PLAN_TEXT_KEYS:
        value = raw.get(key)
        if isinstance(value, str):
            plan[key] = value
    for key in _PLAN_COUNT_KEYS:
        value = raw.get(key)
        # bool is an int subclass — a boolean here is corrupt, not a count.
        if isinstance(value, int) and not isinstance(value, bool):
            plan[key] = value
    for key, (low, high) in _PLAN_CLOCK_RANGES.items():
        value = raw.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and low <= value <= high:
            plan[key] = value
    return plan


def plan_head_of(plan: Any) -> Optional[Tuple[str, str]]:
    """(model, provider) of a chain plan's FIRST hop, or None when it has none.

    The one definition of "head", shared by the writer
    (:meth:`DecisionLog.record`, which persists it) and the reader
    (:func:`attempted_head_of`, which reads it back). Two implementations of
    "which hop runs first" is how a trace comes to disagree with the executor.

    None — not ``("", "")`` — for a plan that is absent, corrupt, or has an empty
    chain, so the writer can tell "no head to record" from "a head whose provider
    is blank" and record nothing rather than an empty attempted_model.
    """
    if not isinstance(plan, dict):
        return None
    chain = plan.get("chain")
    if not isinstance(chain, list):
        return None
    for hop in chain:
        if isinstance(hop, dict) and hop.get("model"):
            return str(hop["model"]), str(hop.get("provider") or "")
    return None


def attempted_head_of(entry: Any) -> Tuple[str, str]:
    """(model, provider) production attempted FIRST for a recorded decision.

    The single accessor for "what actually ran", so a surface never has to know
    that ``output.model`` is the declared TIER primary while the planned head is
    what the executor dispatches. Prefers the recorded ``attempted_model``
    (written by :meth:`DecisionLog.record` whenever a plan supplied a head) and
    falls back to ``output.model``, which is the honest answer for a decision
    with no plan — every entry written before this feature, and every path with
    nothing to attempt.

    The one case this does NOT describe: a caller that named ``model`` explicitly
    overrides the routing decision downstream, so the executor tries the caller's
    model before the plan. That is not a property of the recorded decision and
    the router never sees it as its own choice; this function answers "what the
    ROUTER chose to attempt first".

    Never raises: an entry of the wrong shape yields ``("", "")``.
    """
    if not isinstance(entry, dict):
        return "", ""
    out = entry.get("output")
    if not isinstance(out, dict):
        return "", ""
    model = out.get("attempted_model") or out.get("model") or ""
    if out.get("attempted_model"):
        provider = out.get("attempted_provider") or ""
    else:
        provider = out.get("provider") or ""
    return str(model), str(provider)


# Where a decision came from. Closed set; entries written before this field have
# no ``source`` key and read back as UNKNOWN_SOURCE (never rewritten).
VALID_SOURCES: Tuple[str, ...] = ("kanban", "chat", "delegate")
UNKNOWN_SOURCE = "unknown"


def source_of(entry: Any) -> str:
    """``source`` of a recorded entry; ``unknown`` for old, corrupt or foreign values."""
    value = entry.get("source") if isinstance(entry, dict) else None
    return value if value in VALID_SOURCES else UNKNOWN_SOURCE


class DecisionLog:
    """Append-only decision log for greppable cause= tracing.

    ``source`` (kanban | chat | delegate) stamps every entry this log records.
    None records no key, which keeps the historical shape.
    """

    def __init__(self, source: Optional[str] = None) -> None:
        self._entries: List[Dict[str, Any]] = []
        self._source = source if source in VALID_SOURCES else None

    def record(
        self,
        cause: str,
        output: Dict[str, Any],
        matched_rule_id: Optional[str] = None,
        task_preview: str = "",
        *,
        steps: Optional[List[Dict[str, Any]]] = None,
        chain_plan: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a routing decision.

        ``steps`` is an optional per-stage in/out trace (``[{stage, in, out,
        cause}, ...]``) for visual replay. It is purely additive: when omitted
        the recorded entry keeps its historical shape (no ``steps`` key), so
        existing consumers and persisted logs are unchanged.

        ``chain_plan`` is the ``rules.plan_chain`` result (eligible order,
        rejected+reasons, derived requirements, strategy, independent_rails),
        after the blocklist veto has removed what must not be attempted. Also
        purely additive and bounded: omitted -> no key; a non-mapping value is
        skipped rather than raised; ``rejected`` is truncated to
        :data:`MAX_REJECTED_ENTRIES` with a ``rejected_truncated`` count.

        THE ATTEMPTED HEAD. Whenever ``chain_plan`` supplies a head, it is
        recorded on the entry's own copy of ``output`` as ``attempted_model`` /
        ``attempted_provider`` — the pair the executor dispatches FIRST. This is
        the whole point of the key: ``output["model"]`` is the declared TIER
        primary, and after a capability filter, a time cap, a shuffle or a
        blocklist veto the two differ. Measured before this was populated: a
        vision turn persisted ``output.model == 'glm-5.3'`` — a model that cannot
        see and was never attempted — while ``chain_plan.chain[0]`` was
        ``gpt-5.6-luna``, which is what ran. ``model`` is deliberately NOT
        redefined, because other consumers read it as the tier identity.

        Written UNCONDITIONALLY when a head exists, not only when it differs from
        ``model``: a reader must not have to know whether the writer thought the
        difference was interesting, and "the key is present iff a plan named a
        head" is the only rule :func:`attempted_head_of` can rely on. It is
        recorded on a COPY, so the caller's decision dict is not mutated by
        having been logged.
        """
        if cause not in VALID_CAUSES:
            # NOT fail_safe_strong: a cause this module does not know is a
            # programming error upstream, and recording it as the router's
            # worst real outcome buries the error at exactly the spot an
            # operator counts outcomes. Unknown stays unknown.
            cause = "unknown_cause"

        recorded_output = dict(output)
        bounded = bound_chain_plan(chain_plan) if chain_plan is not None else None
        head = plan_head_of(bounded)
        if head is not None:
            recorded_output["attempted_model"] = head[0]
            # A blank provider is omitted rather than recorded as "": the pair is
            # read back through attempted_head_of, which already answers "" for a
            # missing provider, and a null-ish value in a persisted trace is the
            # shape this module works hardest to avoid.
            if head[1]:
                recorded_output["attempted_provider"] = head[1]

        entry: Dict[str, Any] = {
            "ts": time.time(),
            "cause": cause,
            "output": recorded_output,
            "rule_id": matched_rule_id,
            "task": task_preview[:120],
        }
        if self._source:
            entry["source"] = self._source
        if steps is not None:
            entry["steps"] = steps
        if bounded is not None:
            entry["chain_plan"] = bounded
        self._entries.append(entry)

    def tail(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return the last N entries."""
        return self._entries[-n:]

    def format_line(self, entry: Dict[str, Any]) -> str:
        """Format one entry as a greppable line.

        Entries WITHOUT a chain plan (every historical entry) format exactly as
        before — the chain segment is only appended when a plan is present.

        ``model=`` keeps its historical meaning, the DECLARED tier primary, because
        that is what every existing grep and dashboard column reads it as. When the
        head the executor actually dispatches is a different elo, the line names it
        too as ``attempted=model@provider`` inside the chain segment. Emitting it
        only on a difference is what makes it useful: a line with no ``attempted=``
        states that the declared model IS the one that ran, so the greppable
        surface can no longer say ``model=glm-5.3`` about a turn ``gpt-5.6-luna``
        served without also saying which.
        """
        ts = entry.get("ts", 0)
        cause = entry.get("cause", "?")
        rule = entry.get("rule_id") or "-"
        out = entry.get("output", {})
        profile = out.get("profile", "")
        model = out.get("model", "")
        task = entry.get("task", "")
        chain_seg = ""
        if isinstance(entry.get("chain_plan"), dict):
            plan = chain_plan_of(entry)
            rejected_total = len(plan["rejected"]) + plan["rejected_truncated"]
            chain_seg = (
                f"chain={len(plan['chain'])} rejected={rejected_total} "
                f"strategy={plan['strategy']} rails={plan['independent_rails']} "
            )
            attempted_model, attempted_provider = attempted_head_of(entry)
            if attempted_model and attempted_model != model:
                target = (
                    f"{attempted_model}@{attempted_provider}"
                    if attempted_provider else attempted_model
                )
                chain_seg += f"attempted={target} "
        return (
            f"cause={cause} rule={rule} profile={profile} "
            f"model={model} {chain_seg}task=\"{task}\""
        )

    def chain_plan(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Chain plan of ``entry``, or the empty default for old/corrupt ones."""
        return chain_plan_of(entry)

    def entries(self) -> List[Dict[str, Any]]:
        return list(self._entries)
