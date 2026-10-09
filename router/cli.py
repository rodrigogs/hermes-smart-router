"""CLI governance — router explain, chain, lint, blocklist, log.

v1: file-config + CLI governance. No webui panel.

Read-only except for what the subcommand explicitly does: every command here
loads router.yaml and prints. ``chain`` is the shell-side view of the capability
filter + fallback strategy, so the feature is verifiable on the box without the
webui; ``--json`` makes it scriptable.

This module is the EDGE, so it is the layer allowed to do IO: it reads
router.yaml and it reads the wall clock. ``signals``/``rules``/``capabilities``
stay pure — the clock is injected downwards as a ``datetime`` parameter exactly
like the ``random.Random`` used by the ``random`` fallback strategy, never read
inside them. ``--at`` overrides that clock so "why did the 04:00 cron route
there" is answerable from a shell at 14:00; ``--time-agnostic`` passes no clock
at all (time-dependent features are omitted and time-dependent ordering
degrades to sequential).

The SIZE is injected the same way, and for the same reason: ``chain
--prompt-text`` sizes the turn from the composed context + goal the child would
really receive, because ``est_input_tokens`` and the ``min_context`` derived from
it decide which elos survive the filter. Every other read surface takes that
parameter, and every one of them reports ``preview.sized_from``; this one now does
too, so a plan measured from the goal line is labelled rather than assumed.

Imports of the newer router modules are GUARDED: the CLI is the operator's tool
of last resort, so it must still start (and still lint) when ``capabilities.py``
or a newer ``rules`` helper is absent or mid-write. Every time-layer entry point
(``rules.plan_chain(..., when=)``, ``capabilities.price_multiplier``,
``capabilities.effective_price``) is therefore called through a guard that
degrades to "not shown" rather than raising.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import re
import sys
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .signals import extract
from . import rules as _rules
from .rules import explain as rules_explain, lint as rules_lint
from .blocklist import Blocklist
from .decision_log import DecisionLog, chain_plan_of
from .cache import Cache, SessionPin

try:
    from . import capabilities as _caps
except ImportError:  # capabilities.py absent or mid-write — degrade, never crash
    try:
        from router import capabilities as _caps  # flat-layout fallback
    except ImportError:
        _caps = None

# The top-level ``price_windows`` overlay merge (spec t_c90c5336). Guarded like
# ``_caps``: the CLI is the operator's last-resort tool and must still start —
# and still lint — beside a rules.py that predates the overlay, so the merge
# degrades to the identity (no overlay, no change) rather than a crash.
try:
    from .rules import with_global_price_windows
except ImportError:  # pragma: no cover - flat layout, or rules.py without the overlay
    try:
        from router.rules import with_global_price_windows
    except ImportError:  # pragma: no cover - rules.py predates the overlay
        def with_global_price_windows(config: Dict[str, Any]) -> Dict[str, Any]:
            return config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Seed used when no ``--seed`` is given, mirroring ``rules.explain``'s dry-run
#: preview seed. A bare ``chain`` call must stay reproducible across runs, so it
#: reuses the preview stream rather than reaching for real randomness.
_PREVIEW_SEED = getattr(_rules, "_PREVIEW_SEED", 0)

#: Keys on a chain entry that are routing/identity, never a capability or price
#: declaration. Everything else is handed to the registry as a per-elo override,
#: because ``declared`` WINS over the registry (see capabilities.py docstring).
_ROUTING_KEYS = frozenset(
    getattr(_rules, "_NON_CAPABILITY_KEYS", ())
    or {"model", "provider", "fallback", "fallback_strategy",
        "pin_primary", "requirements"}
) | {"reject_reason"}

#: Plan fields the time layer sets, printed ONLY when set: an absent or falsey
#: flag is noise in the common case and would bury the one that fired.
#: ``peak_priced`` sits next to ``demoted`` on purpose — they are the PRICE and
#: the POSITION reading of the same ``avoid_peak`` match, and an operator can
#: only tell them apart if they are printed side by side.
#: ``time_cap`` leads because it is the CEILING the other two cost readings are
#: about: ``plan_chain`` sets it only when the tier declares one (never as a
#: null, which a consumer would read as a ceiling of 0x), and printing which
#: rails a cap refused while withholding the number it refused them against left
#: this surface saying less than the console, which shows the cap beside them.
_TIME_FLAG_KEYS: Tuple[str, ...] = (
    "time_cap", "capped", "demoted", "peak_priced", "promoted",
    "time_cap_bypassed", "strategy_degraded",
)

# Price lookup outcomes. "unpriced" and "unavailable" are deliberately
# DISTINCT: the first means the vendor publishes no per-token dollar rate (a
# plan model — never $0, which would make it win every cost comparison), the
# second means this build of capabilities.py cannot answer yet.
_PRICE_OK = "priced"
_PRICE_UNPRICED = "unpriced"
_PRICE_UNAVAILABLE = "unavailable"

_BARE_HOUR_RE = re.compile(r"^(\d{1,2})$")
_HOUR_MINUTE_RE = re.compile(r"^(\d{1,2}):(\d{2})$")

#: Which text a plan was SIZED from. The same two words, spelled the same way,
#: that ``RouterService._preview_note`` and the sidecar/dashboard ``/explain``
#: responses report in ``preview.sized_from`` — one vocabulary across every read
#: surface, so "this plan measured the goal line" needs no translation table.
_SIZED_FROM_TASK = "task"
_SIZED_FROM_PROMPT = "prompt_text"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str = "router.yaml") -> Dict[str, Any]:
    """Load router.yaml."""
    p = Path(path)
    if not p.exists():
        print(f"router: config not found at {p.resolve()}", file=sys.stderr)
        sys.exit(1)
    with open(p) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# The measured text — the other input a preview must not invent
# ---------------------------------------------------------------------------

def resolve_prompt(args: argparse.Namespace) -> Tuple[str, str]:
    """Return ``(text_to_size_from, sized_from)`` for a plan.

    ``task`` is the GOAL line; ``--prompt-text`` is the full text the child would
    actually receive (context + goal), which is what production sizes a turn from.
    Signals are extracted from the returned text, because ``est_input_tokens`` and
    the ``min_context`` requirement derived from it decide the route: a 120k-char
    context is 33344 estimated tokens and a real floor, while the goal line alone
    measures 6 and derives no floor at all. Without the option this command
    silently answered a different question than production asked — the same defect
    ``prompt_text`` was added to ``adapter.route`` and ``RouterService.explain`` to
    close — and, being the only surface that did not report which text it measured,
    it answered it invisibly.

    ``prompt_text or task`` is the SAME falsy test the other two surfaces use, so
    ``--prompt-text ''`` and ``--prompt-text '  '`` are measured here exactly as
    production measures them. Being cleverer (stripping, or treating whitespace as
    absent) would make this plan disagree with the path it exists to reproduce, in
    the one direction that matters: char_len.

    The composition of context and goal is NOT re-implemented: the operator pastes
    the composed text, so the plugin's ``_compose_prompt`` stays its single
    definition.

    No length bound, unlike ``RouterService.explain``: that bound exists because an
    unbounded HTTP read path can be made to cost arbitrary CPU by a caller, and
    here the caller IS the operator, spending their own shell on their own box.
    Refusing a 2 MiB prompt would only send them back to measuring the goal line,
    which is the answer this option exists to stop them getting.

    Reads the parsed NAMESPACE rather than a bare string so another subcommand can
    adopt the option without a second copy of the falsy test that decides which
    text was measured.
    """
    prompt_text = getattr(args, "prompt_text", "") or ""
    if not prompt_text:
        return args.task, _SIZED_FROM_TASK
    return prompt_text, _SIZED_FROM_PROMPT


# ---------------------------------------------------------------------------
# The injected clock — parsed here, never read downstream
# ---------------------------------------------------------------------------

def resolve_when(args: argparse.Namespace) -> Tuple[Optional[datetime], str]:
    """Return ``(when, source)`` — the clock to inject and where it came from.

    ``source`` is one of ``now``, ``explicit`` or ``time-agnostic``, so the
    rendered plan says which clock produced it. That vocabulary is deliberately
    the one ``RouterService._evaluated_at`` reports and NOT the flag spelling
    (``--at``): the two surfaces answer the same question, and a console that had
    to translate this one field per surface would be carrying a table that only
    exists because two authors named the same concept twice.

    ``--at`` accepts, in this order:
      * a bare UTC hour, ``0``-``23``       -> today's UTC date at that hour
      * ``HH:MM`` (UTC)                     -> today's UTC date at that time
      * an ISO-8601 timestamp               -> ``2026-08-17T07:00:00Z``,
        ``...+00:00`` or naive (naive is read as UTC); anything with an offset
        is converted to UTC
      * ``now``                             -> the real current UTC time

    The bare-hour form deliberately inherits TODAY's weekday, because the zai
    peak window is weekday-only: an hour with no date could not answer "is this
    inside the Mon-Fri peak" at all. Pass a full timestamp to pick the day.

    Fail-closed: an unparseable value exits 2 rather than silently falling back
    to "now" — an audit tool that answers a different question than the one
    asked is worse than one that refuses.
    """
    if getattr(args, "time_agnostic", False):
        return None, "time-agnostic"
    raw = getattr(args, "at", None)
    if raw is None or raw == "":
        return _utc_now(), "now"
    try:
        return _parse_when(str(raw)), "explicit"
    except ValueError as exc:
        print(f"router: --at {raw!r}: {exc}", file=sys.stderr)
        sys.exit(2)


def _utc_now() -> datetime:
    """The real current UTC time. The ONLY clock read in the router."""
    return datetime.now(timezone.utc)


def _parse_when(raw: str) -> datetime:
    """Parse one ``--at`` value into an aware UTC datetime, or raise ValueError."""
    text = raw.strip()
    if not text:
        raise ValueError("empty value")
    if text.lower() == "now":
        return _utc_now()

    bare = _BARE_HOUR_RE.match(text)
    if bare:
        return _at_time_today(int(bare.group(1)), 0)
    hm = _HOUR_MINUTE_RE.match(text)
    if hm:
        return _at_time_today(int(hm.group(1)), int(hm.group(2)))

    iso = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(iso)
    except ValueError:
        raise ValueError(
            "expected a UTC hour 0-23, HH:MM, or an ISO-8601 timestamp"
        ) from None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _at_time_today(hour: int, minute: int) -> datetime:
    """Today's UTC date at ``hour:minute``, bounds-checked."""
    if not 0 <= hour <= 23:
        raise ValueError("UTC hour must be 0..23")
    if not 0 <= minute <= 59:
        raise ValueError("minute must be 0..59")
    return _utc_now().replace(hour=hour, minute=minute, second=0, microsecond=0)


def _utc_parts(when: Optional[datetime]) -> Optional[Tuple[int, int]]:
    """Return ``(utc_hour, utc_weekday)`` for ``when``, or None for "no clock".

    The same normalisation as ``capabilities._utc_parts``, ``rules._clock_parts``
    and ``adapter._clock_features``: an AWARE datetime is converted to UTC, a
    NAIVE one is assumed to be UTC already. Every caller here comes through
    :func:`resolve_when`, which already returns aware UTC, so this is belt and
    braces — but a feature vector that named a different hour than the price
    multipliers derived from the same clock is the one inconsistency the
    injected-clock design forbids, so the normalisation lives here rather than in
    the convention that callers happen to follow.
    """
    if when is None:
        return None
    if when.tzinfo is not None:
        when = when.astimezone(timezone.utc)
    return when.hour, when.weekday()


def _time_features(when: Optional[datetime]) -> Dict[str, Any]:
    """The two injected time features, or {} when no clock was supplied.

    Matches the spec exactly: with no clock the features are OMITTED, so a
    time-keyed ``when`` clause is inert (an absent feature never matches) rather
    than matching against a guessed hour.
    """
    parts = _utc_parts(when)
    if parts is None:
        return {}
    hour, weekday = parts
    return {"utc_hour": hour, "utc_weekday": weekday}


def _time_payload(when: Optional[datetime], source: str) -> Dict[str, Any]:
    """The machine-readable description of the injected clock.

    ``at_source`` uses the SAME vocabulary as ``RouterService`` (``now`` /
    ``explicit`` / ``time-agnostic``) rather than the flag spelling, so a console
    can render either surface without a per-field translation table.
    """
    parts = _utc_parts(when)
    hour, weekday = parts if parts is not None else (None, None)
    return {
        "at": when.isoformat() if when is not None else None,
        "at_source": source,
        "utc_hour": hour,
        "utc_weekday": weekday,
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_explain(args: argparse.Namespace) -> None:
    """Run explain() on a task and print the decision trace.

    The injected clock (``--at``/``--time-agnostic``, default now) adds
    ``utc_hour``/``utc_weekday`` to the feature vector, so a time-keyed rule is
    traceable here at any hour and not only at the hour you happen to be at.
    """
    task = args.task
    config = load_config(args.config)
    config = with_global_price_windows(config)

    when, at_source = resolve_when(args)
    features = extract(task)
    features.update(_time_features(when))
    blocklist = Blocklist(config)

    # Check if a model override is in the task. `would_block`, not `is_blocked`:
    # this command only EXPLAINS, and the mutating form burns the breaker's
    # single probe slot for an expired-OPEN rail — asking why a turn routed
    # somewhere must not take a rail out of rotation.
    requested_model = args.model or ""
    blocked = blocklist.would_block(requested_model, "")
    if blocked:
        print(json.dumps({
            "cause": "blocklist_veto",
            "output": {"deny": True},
            "fallback": blocklist.fallback_for(requested_model),
        }, indent=2))
        return

    result, _time_aware = _call_explain(task, features, blocked, config, when)
    result = dict(result)
    result.update(_time_payload(when, at_source))

    print(json.dumps(result, indent=2, default=str))


def cmd_chain(args: argparse.Namespace) -> None:
    """Print the capability/fallback chain plan for a task.

    Shows the injected clock, the derived requirements, the eligible order
    actually tried, the per-elo price multiplier and effective price at that
    clock, the rejected elos with their reason, the fallback strategy, the
    time-layer flags that fired (the declared ``time_cap`` and the ``capped``
    rails it refused, ``demoted``, ``peak_priced``, ``promoted``,
    ``time_cap_bypassed``, ``strategy_degraded``) and the number of
    independent upstream rails. A requirement nothing could ever meet is
    headlined from ``unsatisfiable`` rather than left to be inferred from a
    bypass plus a run of identical rejections. ``--json`` prints the same payload
    machine readable.

    ``--prompt-text`` is the composed context + goal the child would really
    receive; the signals are read from it and ``preview.sized_from`` /
    ``preview.prompt_chars`` name the text that produced the plan, the same way
    every other read surface reports it. Omitted, the goal line is measured — which
    is a legitimate answer, and now a DISCLOSED one instead of an assumption.

    ``--seed N`` plans with ``random.Random(N)`` instead of reusing explain's
    fixed-seed preview, so different seeds really do produce different
    ``random``-strategy orders and the same seed reproduces byte-identically —
    which is the whole point of the flag. Without ``--seed`` the preview path is
    kept, so a bare ``chain`` call stays stable across runs. ``plan_source``
    always names the path that produced the printed order.

    Never raises on a missing newer module: the plan degrades to the empty shape
    plus a ``plan_source`` of ``unavailable``, and an unanswerable price renders
    as ``n/a``.
    """
    config = load_config(args.config)
    config = with_global_price_windows(config)
    task = args.task
    when, at_source = resolve_when(args)
    # The task stays the GOAL line (it is what the classifier and the response
    # cache key on) while the signals are measured from the composed prompt — the
    # same split production makes.
    prompt, sized_from = resolve_prompt(args)
    features = extract(prompt)
    features.update(_time_features(when))

    requested_model = getattr(args, "model", "") or ""
    # Read-only, like `explain` above: a plan preview consumes no capacity and
    # must consume no probe slot.
    blocked = Blocklist(config).would_block(requested_model, "")

    result, explain_time_aware = _call_explain(task, features, blocked, config, when)
    output = result.get("output", {}) if isinstance(result, dict) else {}
    seed = getattr(args, "seed", None)
    plan, source, plan_time_aware = _chain_plan_for(
        result, output, features,
        seed=seed, when=when, explain_time_aware=explain_time_aware,
    )

    payload: Dict[str, Any] = {
        "task": task,
        "matched_rule_id": result.get("matched_rule_id"),
        "cause": result.get("cause"),
        "output": output,
        "seed": seed,
        "plan_source": source,
        "plan_time_aware": plan_time_aware,
        "chain_plan": plan,
        "pricing": _pricing_rows(plan, when),
        # Which text the plan was sized from, in the vocabulary every other
        # surface reports it in. A plan sized from the goal line looks exactly
        # like one sized from the real turn and answers a different question.
        "preview": {"sized_from": sized_from, "prompt_chars": len(prompt)},
    }
    payload.update(_time_payload(when, at_source))

    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, default=str))
        return
    _print_chain_plan(payload)


def cmd_lint(args: argparse.Namespace) -> None:
    """Validate router.yaml, fail-closed on errors, advisory on warnings.

    Hard errors are the fail-closed gate and exit 1. ``rules.lint_warnings`` is
    advisory only (a tier whose first two hops share an upstream is legal but
    gives no real redundancy) — it prints in its own clearly-labelled block and
    NEVER changes the exit code, so a warning can never wedge a deploy.
    """
    config = load_config(args.config)
    errors = rules_lint(config)
    warnings = _lint_warnings(config)
    if errors:
        print(f"router: {len(errors)} config error(s):")
        for e in errors:
            print(f"  - {e}")
        _print_warnings(warnings)
        sys.exit(1)
    print("router: config valid")
    _print_warnings(warnings)


def cmd_model_lint(args: argparse.Namespace) -> None:
    """Exit 1 listing every model/provider pair missing from the Hermes cache."""
    from . import model_lint
    from .paths import hermes_root

    root = hermes_root()
    problems = model_lint.run(
        Path(args.config),
        Path(args.profiles_dir) if args.profiles_dir else root / "profiles",
        Path(args.cache) if args.cache else root / "provider_models_cache.json",
    )
    if problems:
        print(f"router: {len(problems)} model(s) missing from provider cache:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("router: all models exist in provider cache")


def _print_warnings(warnings: List[str]) -> None:
    """Print the advisory block. Exit code is deliberately untouched."""
    if not warnings:
        return
    print(f"router: {len(warnings)} warning(s) (advisory, exit code unaffected):")
    for w in warnings:
        print(f"  ! {w}")


def _lint_warnings(config: Dict[str, Any]) -> List[str]:
    """``rules.lint_warnings(config)`` if that helper exists, else no warnings.

    Guarded twice over: the helper may not exist yet (older rules.py), and a
    defect inside it must not turn ``lint`` — the fail-closed gate every write
    runs through — into a traceback.
    """
    fn = getattr(_rules, "lint_warnings", None)
    if not callable(fn):
        return []
    try:
        warnings = fn(config)
    except Exception as exc:  # advisory path: degrade to a note, never raise
        return [f"lint_warnings unavailable: {exc}"]
    if not isinstance(warnings, list):
        return []
    return [str(w) for w in warnings]


def cmd_blocklist(args: argparse.Namespace) -> None:
    """Show banned models, breaker state, and fallback chain."""
    config = load_config(args.config)
    bl = Blocklist(config)

    print("Manual bans:")
    for ban in bl.manual_bans():
        print(f"  - model={ban['model']} provider={ban.get('provider', '*')} "
              f"reason={ban.get('reason', 'none')}")

    # Show breaker cooldowns if enabled
    if bl.breaker_enabled():
        status = bl.breaker_status()
        if status:
            print(f"\nAuto-breaker cooldowns:")
            for s in status:
                remaining = s.get("cooldown_remaining_s", 0)
                if remaining > 120:
                    rem_str = f"{remaining/60:.0f}m remaining"
                elif remaining > 1:
                    rem_str = f"{remaining:.0f}s remaining"
                else:
                    rem_str = "expiring now"
                print(f"  - model={s['model_key']} state={s['state']} "
                      f"cooldown={rem_str} backoff={s['backoff_seconds']:.0f}s "
                      f"last_failure={s.get('last_failure_kind', '-')}")
        else:
            print(f"\nAuto-breaker: enabled, no active cooldowns")

    print(f"\nFallback chain: {' → '.join(bl.fallback_chain())}")


def cmd_log(args: argparse.Namespace) -> None:
    """Tail the decision log."""
    n = args.tail or 20
    # In production, this reads from the session decision log.
    # For v1 CLI, we read from a file if --file is given.
    if args.file:
        try:
            with open(args.file) as f:
                lines = f.readlines()
            for line in lines[-n:]:
                print(line.rstrip())
        except FileNotFoundError:
            print(f"router: log file not found: {args.file}", file=sys.stderr)
            sys.exit(1)
    else:
        print("router: no log file specified (use --file)")

    if args.follow:
        print("router: --follow not yet implemented (v2)")


def cmd_alignment_report(args: argparse.Namespace) -> None:
    """Alignment precision/recall with Wilson intervals."""
    try:
        from . import alignment_eval as ev
    except ImportError:  # pragma: no cover - flat layout
        from router import alignment_eval as ev
    rep = ev.report_from_disk(
        Path(args.log) if args.log else None, Path(args.db) if args.db else None)
    print(json.dumps(rep, indent=2) if args.json else ev.render(rep))


# ---------------------------------------------------------------------------
# Chain plan — resolution (guarded) and rendering
# ---------------------------------------------------------------------------

def _accepts(fn: Any, name: str) -> bool:
    """True when ``fn`` takes a keyword argument called ``name``.

    Asked by signature rather than by catching TypeError, because a TypeError
    raised INSIDE a mid-write module would otherwise be misread as "does not
    take this argument" and silently drop the clock. Unintrospectable callables
    (C builtins, some mocks) answer False: the guarded call then omits the
    argument, which is the degraded-but-correct direction.
    """
    if not callable(fn):
        return False
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    for param in params.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
    param = params.get(name)
    return param is not None and param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY,
    )


def _call_explain(
    task: str,
    features: Dict[str, Any],
    blocked: bool,
    config: Dict[str, Any],
    when: Optional[datetime],
) -> Tuple[Dict[str, Any], bool]:
    """Run ``rules.explain``, injecting the clock only if it accepts one.

    Returns ``(result, time_aware)``. ``time_aware`` is False when a clock was
    supplied but this build of ``rules.explain`` has no ``when`` parameter — the
    preview plan is then time-blind, and the caller must NOT present it as the
    plan for the requested hour.
    """
    pass_when = when is not None and _accepts(rules_explain, "when")
    kwargs: Dict[str, Any] = {"when": when} if pass_when else {}
    result = rules_explain(
        task, features, blocked,
        config.get("rules", []),
        config.get("default", {}),
        config.get("tiers", {}),
        **kwargs,
    )
    if not isinstance(result, dict):
        return {}, False
    # No clock to reflect => the preview is trivially consistent with it.
    return result, pass_when or when is None


def _chain_plan_for(
    result: Dict[str, Any],
    output: Dict[str, Any],
    features: Dict[str, Any],
    *,
    seed: Optional[int] = None,
    when: Optional[datetime] = None,
    explain_time_aware: bool = True,
) -> Tuple[Dict[str, Any], str, bool]:
    """Return ``(plan, source, time_aware)`` for a decision, degrading in steps.

    Preference order — most authoritative first:
      1. ``explain()``'s own ``chain_plan`` key (already computed for this turn),
         used ONLY when no ``--seed`` was given and that preview saw the same
         clock. explain plans with a FIXED ``random.Random(0)``, so reusing it
         under ``--seed N`` would make the seed a no-op;
      2. ``rules.plan_chain(output, features, rng=..., when=...)``;
      3. locally composed from ``capabilities`` primitives (older rules.py);
      4. the empty plan with source ``unavailable`` — printing "no plan" beats
         a traceback in the operator's only shell-side tool.

    ``source`` names the path AND the rng that produced the order
    (``rules.plan_chain(seed=7)`` vs ``rules.plan_chain(preview)``), because an
    order you cannot attribute is indistinguishable from a bug. ``seed`` makes
    the ``random`` strategy reproducible: same task + same seed => same order.
    Without a seed the preview stream (``random.Random(0)``) is reused so a bare
    ``chain`` call stays stable across runs.

    ``time_aware`` reports whether the returned plan actually saw ``when``.
    """
    seeded = seed is not None
    rng = random.Random(seed if seeded else _PREVIEW_SEED)
    label = f"seed={seed}" if seeded else "preview"

    explained = result.get("chain_plan") if isinstance(result, dict) else None
    if not seeded and explain_time_aware and isinstance(explained, dict):
        return _normalize_plan(explained), "explain", True

    fn = getattr(_rules, "plan_chain", None)
    if callable(fn):
        plan, time_aware = _call_plan_chain(fn, output, features, rng, when)
        if plan is not None:
            return _normalize_plan(plan), f"rules.plan_chain({label})", time_aware

    plan, time_aware = _compose_plan(output, features, rng, when)
    if plan is not None:
        return _normalize_plan(plan), f"capabilities({label})", time_aware

    return _normalize_plan(None), "unavailable", when is None


def _call_plan_chain(
    fn: Any,
    output: Dict[str, Any],
    features: Dict[str, Any],
    rng: Optional[random.Random],
    when: Optional[datetime],
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Call ``rules.plan_chain`` tolerantly.

    Returns ``(plan, time_aware)``; ``(None, ...)`` when the planner is
    unusable. ``when`` is passed only when the planner declares it, so a
    pre-time-layer ``rules.py`` still plans (time-blind, and reported as such)
    instead of dying on an unexpected keyword.
    """
    kwargs: Dict[str, Any] = {}
    time_aware = when is None
    if _accepts(fn, "when"):
        kwargs["when"] = when
        time_aware = True
    try:
        plan = fn(output, features, rng=rng, **kwargs)
    except TypeError:
        try:
            plan = fn(output, features)
        except Exception:
            return None, False
        time_aware = when is None
    except Exception:
        return None, False
    return (plan if isinstance(plan, dict) else None), time_aware


def _compose_plan(
    output: Dict[str, Any],
    features: Dict[str, Any],
    rng: Optional[random.Random],
    when: Optional[datetime] = None,
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Build a plan straight from the capabilities primitives.

    Used only when ``rules`` has no ``plan_chain`` yet. Every call is inside one
    guard: this is a diagnostic path and must never be the thing that breaks.
    Returns ``(plan, time_aware)`` — the ordering primitive only sees the clock
    when this build of ``capabilities.order_chain`` declares a ``when``.
    """
    if _caps is None:
        return None, False
    chain: List[Dict[str, Any]] = []
    if output.get("model"):
        head = {"model": output.get("model")}
        if output.get("provider"):
            head["provider"] = output["provider"]
        chain.append(head)
    fallback = output.get("fallback")
    if isinstance(fallback, list):
        chain.extend(t for t in fallback if isinstance(t, dict))
    if not chain:
        return None, False

    tier_requirements = output.get("requirements")
    strategy = output.get("fallback_strategy") or "sequential"
    pin_primary = output.get("pin_primary", True)
    order_kwargs: Dict[str, Any] = {}
    time_aware = when is None
    if _accepts(getattr(_caps, "order_chain", None), "when"):
        order_kwargs["when"] = when
        time_aware = True
    try:
        requirements = _caps.derive_requirements(
            features,
            tier_requirements if isinstance(tier_requirements, dict) else None,
        )
        filtered = _caps.filter_chain(chain, requirements)
        eligible = _caps.order_chain(
            filtered.get("eligible", []), str(strategy),
            pin_primary=bool(pin_primary), rng=rng, **order_kwargs,
        )
        return {
            "chain": eligible,
            "requirements": requirements,
            "rejected": filtered.get("rejected", []),
            "unknown": filtered.get("unknown", []),
            "bypassed": bool(filtered.get("bypassed", False)),
            "strategy": str(strategy),
            "independent_rails": _caps.independent_rails(eligible),
        }, time_aware
    except Exception:
        return None, False


def _normalize_plan(plan: Any) -> Dict[str, Any]:
    """Guarantee every rendered key exists, keeping any extra keys the plan has.

    Reuses the decision-log parser so the CLI and a replayed trace agree on what
    a missing or corrupt plan means: the empty default, never an exception.
    """
    normalized = chain_plan_of({"chain_plan": plan})
    if isinstance(plan, dict):
        for key, value in plan.items():
            if key not in normalized:
                normalized[key] = value
    return normalized


# ---------------------------------------------------------------------------
# Time-relative pricing — what each elo costs at the injected clock
# ---------------------------------------------------------------------------

def _pricing_rows(
    plan: Dict[str, Any],
    when: Optional[datetime],
) -> List[Dict[str, Any]]:
    """Per-elo multiplier + effective price at ``when``, in chain order.

    The multiplier the PLANNER used wins when the plan carries one (``plan
    ['multipliers']``), because that is the number the ordering decision was
    actually made on; otherwise it is recomputed from the registry. A model with
    no published per-token price is reported as ``unpriced`` with null prices —
    NEVER 0.0, which would make a plan model look free and win every cost
    comparison.
    """
    rows: List[Dict[str, Any]] = []
    chain = plan.get("chain")
    # Both shapes are GUARANTEED by _normalize_plan (chain_plan_of defaults a
    # corrupt chain to [] and corrupt multipliers to {}), so neither guard can be
    # provoked through this command; they stay as the contract for a future
    # caller that hands over a raw plan.
    if not isinstance(chain, list):  # pragma: no cover - normalized upstream
        return rows
    declared_multipliers = plan.get("multipliers")
    if not isinstance(declared_multipliers, dict):  # pragma: no cover - as above
        declared_multipliers = {}

    for target in chain:
        if not isinstance(target, dict):
            continue
        model = target.get("model")
        model = model if isinstance(model, str) else str(model)
        multiplier = declared_multipliers.get(model)
        if not _is_number(multiplier):
            multiplier = _price_multiplier(model, target, when)
        status, price = _effective_price(model, target, when)
        rows.append({
            "model": model,
            "provider": target.get("provider"),
            "multiplier": float(multiplier) if _is_number(multiplier) else None,
            "pricing": status,
            "unpriced": status == _PRICE_UNPRICED,
            "price_in": price[0] if price is not None else None,
            "price_out": price[1] if price is not None else None,
        })
    return rows


def _is_number(value: Any) -> bool:
    """True for a real int/float. bool is an int subclass and is NOT a price."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _declared_of(target: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Per-elo registry overrides declared on a chain entry, or None."""
    declared = {
        key: value for key, value in target.items() if key not in _ROUTING_KEYS
    }
    return declared or None


def _price_multiplier(
    model: str,
    target: Dict[str, Any],
    when: Optional[datetime],
) -> Optional[float]:
    """``capabilities.price_multiplier`` if this build has it, else None."""
    fn = getattr(_caps, "price_multiplier", None) if _caps is not None else None
    if not callable(fn):
        return None
    try:
        value = fn(model, when=when, declared=_declared_of(target))
    except TypeError:
        try:
            value = fn(model, when)
        except Exception:
            return None
    except Exception:
        return None
    return float(value) if _is_number(value) else None


def _effective_price(
    model: str,
    target: Dict[str, Any],
    when: Optional[datetime],
) -> Tuple[str, Optional[Tuple[float, float]]]:
    """``(status, (price_in, price_out))`` at ``when``.

    ``status`` separates the two zero-like answers on purpose: ``unpriced``
    (vendor publishes no dollar rate — plan credits) from ``unavailable`` (this
    build of capabilities.py cannot answer). Neither is ever rendered as $0.
    """
    fn = getattr(_caps, "effective_price", None) if _caps is not None else None
    if not callable(fn):
        return _PRICE_UNAVAILABLE, None
    try:
        price = fn(model, when=when, declared=_declared_of(target))
    except TypeError:
        try:
            price = fn(model, when)
        except Exception:
            return _PRICE_UNAVAILABLE, None
    except Exception:
        return _PRICE_UNAVAILABLE, None
    if price is None:
        return _PRICE_UNPRICED, None
    if (
        isinstance(price, (tuple, list))
        and len(price) == 2
        and all(_is_number(p) for p in price)
    ):
        return _PRICE_OK, (float(price[0]), float(price[1]))
    return _PRICE_UNAVAILABLE, None


def _print_chain_plan(payload: Dict[str, Any]) -> None:
    """Human-readable rendering — the shell-side answer to 'why this elo?'."""
    plan = payload.get("chain_plan", {})
    print(f"task: {payload.get('task', '')}")
    print(f"matched_rule: {payload.get('matched_rule_id') or '-'} "
          f"cause={payload.get('cause') or '-'}")
    print(f"plan_source: {payload.get('plan_source', 'unavailable')}")
    print(_sized_from_line(payload))
    print(_at_line(payload))
    if payload.get("at") is not None and not payload.get("plan_time_aware", True):
        # An order computed without the clock must not be read as the order for
        # the requested hour, even though the prices below are for that hour.
        print("plan_time_aware: false (the planner has no clock parameter in "
              "this build — order is time-blind)")
    print(f"strategy: {plan.get('strategy', 'sequential')}")
    print(f"independent_rails: {plan.get('independent_rails', 0)}")
    print(f"bypassed: {str(bool(plan.get('bypassed', False))).lower()}")
    # Printed before the time flags and above the rejections it explains: this
    # is the headline of the pathological case, not a footnote to it.
    for line in _unsatisfiable_lines(plan):
        print(line)
    for line in _time_flag_lines(plan):
        print(line)

    requirements = plan.get("requirements") or {}
    print("requirements:")
    if requirements:
        for key in sorted(requirements):
            print(f"  {key} = {requirements[key]}")
    else:
        print("  (none derived)")

    print("eligible:")
    chain = plan.get("chain") or []
    if chain:
        for i, target in enumerate(chain, start=1):
            print(f"  {i}. {_target_label(target)}")
    else:
        print("  (empty)")

    _print_pricing(payload)

    print("rejected:")
    rejected = plan.get("rejected") or []
    if rejected:
        for target in rejected:
            reason = "unknown"
            if isinstance(target, dict):
                reason = target.get("reject_reason") or "unknown"
            print(f"  - {_target_label(target)} reject_reason={reason}")
    else:
        print("  (none)")
    truncated = plan.get("rejected_truncated", 0)
    if truncated:
        print(f"  ... {truncated} more rejected (truncated)")

    unknown = plan.get("unknown") or []
    if unknown:
        print(f"unknown_capabilities: {', '.join(str(u) for u in unknown)}")


def _unsatisfiable_lines(plan: Dict[str, Any]) -> List[str]:
    """The headline for a pathological request, or nothing at all.

    ``unsatisfiable`` names the requirement keys NO elo could ever meet. Without
    it rendered, this case reaches the operator as ``bypassed: true`` plus three
    identical ``context_too_small`` rejections and they have to work out for
    themselves that the requirement, not the roster, is the impossible half —
    the exact reconstruction the field was added to remove. So it is printed
    first among the diagnostics, it names the requirement, and it prints the
    ceiling that makes the requirement impossible, because "1.5M needed against
    a 1.05M largest window" is what tells an operator whether to split the task
    or add a rail.

    Never re-derived: ``capabilities.filter_chain`` owns "no model could ever
    meet this" and ``plan_chain`` carries it through unchanged, so a second
    opinion computed here could only ever be a second answer. The ceiling is a
    registry LOOKUP, and an absent one just drops that clause.
    """
    names = plan.get("unsatisfiable")
    if not isinstance(names, list) or not names:
        return []
    keys = [str(name) for name in names]
    lines = [f"unsatisfiable: {', '.join(keys)} "
             "(no registered elo could EVER meet this — the requirement is "
             "unmeetable, not these particular elos)"]

    requirements = plan.get("requirements")
    requirements = requirements if isinstance(requirements, dict) else {}
    ceiling = _context_ceiling()
    for key in keys:
        needed = requirements.get(key)
        if key == "min_context" and _is_number(needed) and ceiling is not None:
            lines.append(f"  {key}: {needed} needed, and the largest registered "
                         f"context window is {ceiling}")
        elif needed is not None:
            lines.append(f"  {key}: {needed} needed, and nothing registered "
                         f"offers it")
        else:
            lines.append(f"  {key}: nothing registered can satisfy it")
    lines.append("  → split the task into smaller reads, or add a rail that can "
                 "meet it; reordering this chain cannot help")
    return lines


def _context_ceiling() -> Optional[int]:
    """The largest context window any REGISTERED elo has, or None.

    Guarded like every other registry read here: an older or mid-write
    ``capabilities.py`` has no ``MAX_REGISTERED_CONTEXT``, and a headline that
    names the unmeetable requirement without its ceiling is still the half the
    operator cannot get from anywhere else.
    """
    value = (getattr(_caps, "MAX_REGISTERED_CONTEXT", None)
             if _caps is not None else None)
    if not _is_number(value) or value <= 0:
        return None
    return int(value)


def _sized_from_line(payload: Dict[str, Any]) -> str:
    """The size header — the requirements and rejections below are relative to it.

    Printed next to the clock line because the two are the same kind of fact: the
    inputs a preview would otherwise invent. A ``min_context`` requirement derived
    from a goal line is the most misleading output this command can produce, and
    the only thing that distinguishes it from the real one is this line.
    """
    preview = payload.get("preview")
    preview = preview if isinstance(preview, dict) else {}
    sized_from = preview.get("sized_from", _SIZED_FROM_TASK)
    chars = preview.get("prompt_chars", 0)
    if sized_from == _SIZED_FROM_PROMPT:
        return (f"sized_from: {sized_from} ({chars} chars — the composed context "
                "+ goal)")
    return (f"sized_from: {sized_from} ({chars} chars — the GOAL LINE only; pass "
            "--prompt-text to size from the composed prompt production sends)")


def _at_line(payload: Dict[str, Any]) -> str:
    """The clock header — every price and order below is relative to it."""
    at = payload.get("at")
    source = payload.get("at_source", "now")
    if at is None:
        return ("at: (time-agnostic — no clock injected; time-dependent features "
                f"omitted, time-dependent ordering degrades to sequential) "
                f"source={source}")
    return (f"at: {at} (utc_hour={payload.get('utc_hour')} "
            f"utc_weekday={payload.get('utc_weekday')}) source={source}")


def _print_pricing(payload: Dict[str, Any]) -> None:
    """Per-elo multiplier + effective price, aligned with the eligible order."""
    rows = payload.get("pricing")
    if not isinstance(rows, list) or not rows:
        return
    print("pricing:")
    for i, row in enumerate(rows, start=1):
        label = _target_label(row)
        print(f"  {i}. {label} {_multiplier_str(row.get('multiplier'))} "
              f"{_price_str(row)}")


def _multiplier_str(multiplier: Any) -> str:
    """``x2.0`` / ``x0.8`` / ``x1.25`` / ``x?`` when nothing can price the elo.

    A whole multiplier keeps one decimal on purpose: ``x2`` reads like a count,
    ``x2.0`` reads like a rate, and this line is scanned for "am I in a peak".
    """
    if not _is_number(multiplier):
        return "x?"
    value = float(multiplier)
    return f"x{value:.1f}" if value == int(value) else f"x{value:g}"


def _price_str(row: Dict[str, Any]) -> str:
    """The price fragment of one pricing row. Never renders an unpriced $0."""
    status = row.get("pricing")
    if status == _PRICE_OK:
        return (f"in=${row.get('price_in'):.4g}/1M "
                f"out=${row.get('price_out'):.4g}/1M")
    if status == _PRICE_UNPRICED:
        return "unpriced (no per-token $ rate published — plan/subscription credits)"
    return "price=n/a (this build of capabilities.py cannot price it)"


def _time_flag_lines(plan: Dict[str, Any]) -> List[str]:
    """Render the time-layer flags that fired, and only those.

    ``time_cap_bypassed`` is the one an operator must never miss: it means a
    cost cap was dropped to keep the chain non-empty (a cost control must not be
    able to cause an outage), so it is annotated rather than printed bare.

    ``strategy_degraded`` reads the reason the PLANNER computed
    (:func:`_strategy_degrade_note`) rather than asserting one of its own.

    ``demoted`` and ``peak_priced`` are annotated for a different reason: they
    are two readings of one ``avoid_peak`` match and printing both bare would
    leave the operator to guess which is which. ``demoted`` is a POSITION fact —
    what this plan actually moved later; ``peak_priced`` is a PRICE fact — every
    matched elo inside a dearer window, moved or not. On the shipped T3/T4 shape
    the matched rails are already the trailing hops, so nothing moves and only
    ``peak_priced`` is printed; that case gets an extra line saying so outright,
    because "the order looks untouched but two rails cost double" is otherwise
    read as a policy that failed to fire.
    """
    lines: List[str] = []
    for key in _TIME_FLAG_KEYS:
        if key not in plan:
            continue
        value = plan[key]
        if value is None or value is False or value == [] or value == {} or value == "":
            continue
        rendered = _flag_value(value)
        if key == "time_cap_bypassed":
            rendered += " (time_cap would have emptied the chain — cap dropped)"
        elif key == "strategy_degraded":
            rendered += _strategy_degrade_note(plan)
        elif key == "time_cap":
            rendered += (" (PRICE CEILING: a rail dearer than this at this hour "
                         "is refused — see capped)")
        elif key == "demoted":
            rendered += " (POSITION: time_policy moved these later in the chain)"
        elif key == "peak_priced":
            rendered += (" (PRICE: avoid_peak matched these inside a dearer "
                         "window at this hour — a statement about the bill, not "
                         "about the order)")
        lines.append(f"{key}: {rendered}")
        if key == "peak_priced" and not plan.get("demoted"):
            lines.append("  nothing moved: these were already the trailing hops, "
                         "so the order above is unchanged and correct — they cost "
                         "more at this hour and this chain cannot step around them")
    return lines


def _strategy_degrade_note(plan: Dict[str, Any]) -> str:
    """Why the declared strategy did not run — the PLANNER's sentence, not a guess.

    ``rules._effective_strategy`` already decides this and ``plan_chain`` carries
    it in ``strategy_degraded_reason`` ("'cheapest' is not a known fallback
    strategy", "no rng was injected, so the tail was not shuffled", "the
    capability registry is unavailable, so nothing was reordered"). This line used
    to annotate every degrade with one invented cause — "needed a clock/rng it did
    not get" — which is simply false for a misspelled strategy name or an absent
    registry, and sends the operator off to add ``--at``/``--seed`` for a typo.
    The console fixed the same defect by reading this field; the shell reads the
    same field, so the two surfaces cannot disagree about the cause.

    The DECLARED strategy is named because ``strategy:`` above prints the one that
    actually RAN: without it the block reports a degrade without ever saying which
    order was asked for.
    """
    declared = plan.get("strategy_declared") or "strategy"
    reason = plan.get("strategy_degraded_reason")
    if not isinstance(reason, str) or not reason:
        # A planner that reports the degrade without its cause: say that much and
        # stop. Filling the gap in is what made the old annotation wrong.
        return f" (declared {declared} did not run; this plan reports no reason)"
    return f" (declared {declared} did not run: {reason})"


def _flag_value(value: Any) -> str:
    """Flag rendering: bools lowercase, lists joined, anything else stringified."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ", ".join(_target_label(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}={value[k]}" for k in sorted(value, key=str))
    return str(value)


def _target_label(target: Any) -> str:
    """``model (provider)`` for a chain entry, tolerant of any junk value."""
    if not isinstance(target, dict):
        return str(target)
    model = target.get("model", "?")
    provider = target.get("provider")
    return f"{model} ({provider})" if provider else str(model)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _add_time_args(parser: argparse.ArgumentParser) -> None:
    """Add the injected-clock flags. Mutually exclusive: they are two answers
    to the same question, and accepting both would leave which one won implicit.
    """
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--at", default=None, metavar="TIME",
        help="Evaluate at this UTC time instead of now: a bare hour 0-23, "
             "HH:MM, or an ISO-8601 timestamp (2026-08-17T07:00:00Z; naive is "
             "read as UTC). Hour-only forms inherit today's UTC date, which "
             "matters for the weekday-only zai peak window",
    )
    group.add_argument(
        "--time-agnostic", action="store_true",
        help="Inject no clock at all: time features are omitted and "
             "time-dependent ordering degrades to sequential",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="router",
        description="Smart Router governance CLI",
    )
    parser.add_argument("--config", default="router.yaml",
                       help="Path to router.yaml (default: router.yaml)")

    sub = parser.add_subparsers(dest="command", required=True)

    # explain
    p_explain = sub.add_parser("explain", help="Trace routing decision for a task")
    p_explain.add_argument("task", help="Task description to classify")
    p_explain.add_argument("--model", default="", help="Requested model (for blocklist check)")
    _add_time_args(p_explain)
    p_explain.set_defaults(func=cmd_explain)

    # chain — capability filter + fallback strategy for one task
    p_chain = sub.add_parser(
        "chain",
        help="Show the capability/fallback chain plan for a task",
    )
    p_chain.add_argument("task", help="Task description to plan")
    p_chain.add_argument("--model", default="",
                         help="Requested model (for blocklist check)")
    p_chain.add_argument("--prompt-text", default="",
                         help="The composed prompt (context + goal) the child "
                              "would receive. Signals are measured from it, so "
                              "the plan is sized like production's. Omitted, the "
                              "task line is measured (and reported as such)")
    p_chain.add_argument("--seed", type=int, default=None,
                         help="RNG seed for the 'random' fallback strategy "
                              "(same seed => same order; different seeds => "
                              "different orders). Omit to reuse the fixed-seed "
                              "preview plan")
    _add_time_args(p_chain)
    p_chain.add_argument("--json", action="store_true",
                         help="Emit the plan as JSON")
    p_chain.set_defaults(func=cmd_chain)

    # lint
    p_lint = sub.add_parser("lint", help="Validate router.yaml")
    p_lint.set_defaults(func=cmd_lint)

    # model-lint
    p_ml = sub.add_parser(
        "model-lint", help="Check router.yaml/profile models against the Hermes cache"
    )
    p_ml.add_argument("--profiles-dir", help="Default: <hermes root>/profiles")
    p_ml.add_argument("--cache", help="Default: <hermes root>/provider_models_cache.json")
    p_ml.set_defaults(func=cmd_model_lint)

    # blocklist
    p_bl = sub.add_parser("blocklist", help="Show blocked models")
    p_bl.set_defaults(func=cmd_blocklist)

    # log
    p_log = sub.add_parser("log", help="Tail decision log")
    p_log.add_argument("--tail", type=int, default=20, help="Number of lines (default: 20)")
    p_log.add_argument("--file", help="Log file path")
    p_log.add_argument("--follow", "-f", action="store_true",
                       help="Follow (v2)")
    p_log.set_defaults(func=cmd_log)

    # alignment
    p_al = sub.add_parser("alignment", help="Session-alignment evaluation")
    al_sub = p_al.add_subparsers(dest="alignment_command", required=True)
    p_rep = al_sub.add_parser("report", help="Precision/recall (Wilson) of judged runs")
    p_rep.add_argument("--log", help="alignment.jsonl path")
    p_rep.add_argument("--db", help="kanban.db path (read-only snapshot)")
    p_rep.add_argument("--json", action="store_true", help="Emit JSON")
    p_rep.set_defaults(func=cmd_alignment_report)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
