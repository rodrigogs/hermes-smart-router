"""Unit tests for auto-breaker (router/breaker.py and router/blocklist.py)."""

import json

import pytest

from router.breaker import BreakerState, _Entry, _Event, FAILURE_WEIGHTS
from router.blocklist import Blocklist, _state_path, _state_dir


# ---------------------------------------------------------------------------
# State isolation — applies to every test in this module
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def hermetic_state(tmp_path, monkeypatch):
    """Point breaker state at a throwaway HERMES_HOME and return its path.

    The Blocklist tests build a real Blocklist with the auto-breaker enabled, so
    each one loads and persists breaker-state.json at whatever `_state_path()`
    resolves from HERMES_HOME — which defaults to the operator's own `~/.hermes`.
    This module pinned nothing and each of those tests opened by *deleting* that
    file: measured on a plain run, 26 write events against the live state (10
    unlink, 8 mkdir, 8 atomic replaces), leaving OPEN cooldowns for models that do
    not exist (`flaky@prov`) in the file that decides whether a billed rail gets
    retried.

    Autouse and module-wide rather than a call at the top of each test, so
    isolation is a property of the file: a test added later is covered without
    remembering to ask, and nothing has to delete anything to start clean. The pure
    state-machine tests never touch disk and do not need it, but pinning costs
    nothing and "no test here can resolve to the operator's home" is the rule worth
    holding unconditionally.

    The assert is the load-bearing part: it asks the module where it will write, so
    an isolation that stopped redirecting — renamed env var, a test that re-points
    HERMES_HOME itself — fails here instead of reporting green.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    path = _state_path()
    assert tmp_path.resolve() in path.resolve().parents, (
        f"breaker state escaped isolation: {path} is not under {tmp_path}"
    )
    return path


# ---------------------------------------------------------------------------
# BreakerState — pure state machine tests
# ---------------------------------------------------------------------------

BREAKER_CONFIG = {
    "enabled": True,
    "threshold": 5,
    "window_seconds": 600,
    "base_cooldown_seconds": 60,
    "max_cooldown_seconds": 900,
    "backoff_multiplier": 2.0,
}


class TestBreakerStateClosedToOpen:
    """CLOSED → OPEN transitions."""

    def test_trips_when_weight_exceeds_threshold(self):
        bs = BreakerState(BREAKER_CONFIG)
        # 2 TTFB stalls = weight 6 > threshold 5
        tripped = bs.record("gpt-5.6-sol@openai-codex", "ttfb_stall", 100.0)
        assert not tripped  # first event: weight 3 < 5
        tripped = bs.record("gpt-5.6-sol@openai-codex", "ttfb_stall", 110.0)
        assert tripped
        assert bs.is_blocked("gpt-5.6-sol@openai-codex", 120.0)

    def test_does_not_trip_below_threshold(self):
        bs = BreakerState(BREAKER_CONFIG)
        # 2 hard_timeouts = weight 2 < 5
        bs.record("model@prov", "hard_timeout", 100.0)
        bs.record("model@prov", "hard_timeout", 110.0)
        assert not bs.is_blocked("model@prov", 120.0)

    def test_sliding_window_prunes_old_events(self):
        bs = BreakerState(BREAKER_CONFIG)
        # Event at t=0, window=600s. At t=700, event pruned.
        bs.record("model@prov", "ttfb_stall", 0.0)  # weight 3
        bs.record("model@prov", "idle_stall", 10.0)  # weight 2, total 5 → trips
        # Verify blocked
        assert bs.is_blocked("model@prov", 50.0)
        # After cooldown + window expiry, old events gone
        # New event at t=700: old events are 700s old → pruned
        # Only the new event counts
        bs2 = BreakerState(BREAKER_CONFIG)
        bs2.record("model2@prov", "ttfb_stall", 0.0)
        # At t=700, this event is outside window
        tripped = bs2.record("model2@prov", "idle_stall", 700.0)
        # ttfb_stall at 0 is pruned; idle_stall at 700 = weight 2 < 5
        assert not tripped

    def test_weight_ttfb_stall(self):
        assert FAILURE_WEIGHTS["ttfb_stall"] == 3

    def test_weight_idle_stall(self):
        assert FAILURE_WEIGHTS["idle_stall"] == 2

    def test_weight_hard_timeout(self):
        assert FAILURE_WEIGHTS["hard_timeout"] == 1

    def test_weight_crash(self):
        assert FAILURE_WEIGHTS["crash"] == 1


class TestBreakerOpenHalfOpen:
    """OPEN → HALF_OPEN → CLOSED transitions."""

    def test_halp_open_after_cooldown(self):
        bs = BreakerState(BREAKER_CONFIG)
        # Trip with 2 TTFB stalls
        bs.record("model@prov", "ttfb_stall", 100.0)
        bs.record("model@prov", "ttfb_stall", 110.0)
        # Within cooldown (base = 60s, tripped at 110, cooldown until 170)
        assert bs.is_blocked("model@prov", 120.0)
        # After cooldown
        assert not bs.is_blocked("model@prov", 200.0)
        # Now in HALF_OPEN — not blocked

    def test_half_open_success_resets(self):
        bs = BreakerState(BREAKER_CONFIG)
        bs.record("model@prov", "ttfb_stall", 100.0)
        bs.record("model@prov", "ttfb_stall", 110.0)
        # Pass cooldown
        assert not bs.is_blocked("model@prov", 200.0)
        # Record success
        bs.record_success("model@prov", 210.0)
        # Verify not blocked
        assert not bs.is_blocked("model@prov", 220.0)

    def test_half_open_failure_retrips(self):
        bs = BreakerState(BREAKER_CONFIG)
        # Trip
        bs.record("model@prov", "ttfb_stall", 100.0)
        bs.record("model@prov", "ttfb_stall", 110.0)
        # Enter HALF_OPEN
        assert not bs.is_blocked("model@prov", 200.0)
        # Probe fails
        tripped = bs.record("model@prov", "idle_stall", 210.0)
        assert tripped
        # Back to OPEN with extended cooldown (120s now)
        assert bs.is_blocked("model@prov", 220.0)

    def test_half_open_allows_exactly_one_probe_until_outcome(self):
        """Cooldown expiry opens one probe slot, not a stampede of callers."""
        bs = BreakerState(BREAKER_CONFIG)
        bs.record("model@prov", "ttfb_stall", 100.0)
        bs.record("model@prov", "ttfb_stall", 110.0)

        assert bs.is_blocked("model@prov", 120.0)
        assert not bs.is_blocked("model@prov", 200.0)
        assert bs.is_blocked("model@prov", 201.0)

        bs.record_success("model@prov", 210.0)
        assert not bs.is_blocked("model@prov", 220.0)

    def test_exponential_backoff(self):
        bs = BreakerState(BREAKER_CONFIG)
        # First trip at t=110 (base 60s, until 170)
        bs.record("model@prov", "ttfb_stall", 100.0)
        bs.record("model@prov", "ttfb_stall", 110.0)
        assert bs.is_blocked("model@prov", 120.0)
        # Cooldown expires → HALF_OPEN
        assert not bs.is_blocked("model@prov", 200.0)
        # Second trip (HALF_OPEN failure): backoff = 60 * 2 = 120s, until 330
        bs.record("model@prov", "idle_stall", 210.0)
        assert bs.is_blocked("model@prov", 220.0)
        # Cooldown expires → HALF_OPEN
        assert not bs.is_blocked("model@prov", 400.0)
        # Third trip: backoff = 120 * 2 = 240s, until 650
        bs.record("model@prov", "hard_timeout", 410.0)
        assert bs.is_blocked("model@prov", 500.0)
        # Cooldown expires → HALF_OPEN
        assert not bs.is_blocked("model@prov", 700.0)
        # Fourth trip: backoff = 240 * 2 = 480s, until 1180
        bs.record("model@prov", "crash", 710.0)
        assert bs.is_blocked("model@prov", 800.0)

    def test_backoff_capped_at_max(self):
        config = {**BREAKER_CONFIG, "base_cooldown_seconds": 400}
        bs = BreakerState(config)
        # First trip: 400s, until 510
        bs.record("model@prov", "ttfb_stall", 100.0)
        bs.record("model@prov", "ttfb_stall", 110.0)
        assert bs.is_blocked("model@prov", 200.0)
        # Cooldown expires → HALF_OPEN
        assert not bs.is_blocked("model@prov", 600.0)
        # Second trip: backoff = 400 * 2 = 800s, until 1410
        bs.record("model@prov", "idle_stall", 610.0)
        assert bs.is_blocked("model@prov", 700.0)
        # Cooldown expires → HALF_OPEN
        assert not bs.is_blocked("model@prov", 1500.0)
        # Third trip: backoff = 800 * 2 = 1600 capped at 900, until 2400
        bs.record("model@prov", "crash", 1510.0)
        assert bs.is_blocked("model@prov", 1600.0)
        assert not bs.is_blocked("model@prov", 2500.0)  # 1510+900 = 2410 < 2500


class TestBreakerSerialization:
    """to_dict / from_dict round trip."""

    def test_round_trip_empty(self):
        bs = BreakerState(BREAKER_CONFIG)
        data = bs.to_dict()
        bs2 = BreakerState.from_dict(data, BREAKER_CONFIG)
        assert bs2.to_dict() == data

    def test_round_trip_with_entries(self):
        bs = BreakerState(BREAKER_CONFIG)
        bs.record("a@b", "ttfb_stall", 100.0)
        bs.record("a@b", "ttfb_stall", 110.0)
        bs.record("c@d", "idle_stall", 200.0)
        data = bs.to_dict()
        bs2 = BreakerState.from_dict(data, BREAKER_CONFIG)
        assert bs2.to_dict() == data

    def test_version_mismatch_returns_empty(self):
        bs = BreakerState.from_dict(
            {"version": 99, "entries": {}},
            BREAKER_CONFIG,
        )
        assert bs.to_dict() == {"version": 1, "entries": {}}

    def test_corrupt_json_returns_empty(self):
        bs = BreakerState.from_dict({"garbage": True}, BREAKER_CONFIG)
        assert bs.to_dict() == {"version": 1, "entries": {}}

    def test_none_returns_empty(self):
        bs = BreakerState.from_dict(None, BREAKER_CONFIG)  # type: ignore
        assert bs.to_dict() == {"version": 1, "entries": {}}


class TestBreakerStateTransitions:
    """Full state machine coverage."""

    def test_closed_to_open(self):
        bs = BreakerState(BREAKER_CONFIG)
        tripped = bs.record("k", "ttfb_stall", 100.0)
        assert not tripped
        tripped = bs.record("k", "ttfb_stall", 110.0)
        assert tripped
        assert bs.is_blocked("k", 120.0)

    def test_open_to_half_open(self):
        bs = BreakerState(BREAKER_CONFIG)
        bs.record("k", "ttfb_stall", 100.0)
        bs.record("k", "ttfb_stall", 110.0)
        assert bs.is_blocked("k", 120.0)
        assert not bs.is_blocked("k", 200.0)  # HALF_OPEN

    def test_half_open_to_closed(self):
        bs = BreakerState(BREAKER_CONFIG)
        bs.record("k", "ttfb_stall", 100.0)
        bs.record("k", "ttfb_stall", 110.0)
        assert not bs.is_blocked("k", 200.0)
        bs.record_success("k", 210.0)
        assert not bs.is_blocked("k", 220.0)

    def test_half_open_to_open(self):
        bs = BreakerState(BREAKER_CONFIG)
        bs.record("k", "ttfb_stall", 100.0)
        bs.record("k", "ttfb_stall", 110.0)
        assert not bs.is_blocked("k", 200.0)
        tripped = bs.record("k", "ttfb_stall", 210.0)
        assert tripped
        assert bs.is_blocked("k", 220.0)

    def test_success_in_closed_does_not_reset(self):
        """record_success in CLOSED has no effect (window governs expiry)."""
        bs = BreakerState(BREAKER_CONFIG)
        bs.record("k", "ttfb_stall", 100.0)
        bs.record_success("k", 110.0)
        # Still has 1 TTFB stall event (weight 3)
        tripped = bs.record("k", "idle_stall", 120.0)
        assert tripped  # 3 + 2 = 5 ≥ threshold


# ---------------------------------------------------------------------------
# Blocklist + Breaker integration tests
# ---------------------------------------------------------------------------

BLOCKLIST_CONFIG = {
    "blocklist": {
        "manual_ban": [
            {"model": "gpt-5.6-sol", "provider": "openai-codex",
             "reason": "accept-but-never-stream"},
        ],
        "fallback_chain": ["gpt-5.6-sol", "glm-5.2"],
        "auto_breaker": {
            "enabled": True,
            "threshold": 5,
            "window_seconds": 600,
            "base_cooldown_seconds": 60,
            "max_cooldown_seconds": 900,
            "backoff_multiplier": 2.0,
        },
    }
}


class TestBlocklistWithBreaker:
    """Blocklist integration with BreakerState."""

    def test_state_write_lands_under_the_temp_home(self, hermetic_state, tmp_path):
        """A real persist lands in the throwaway home, and only there.

        The fixture asserts where a write is *aimed*; this asserts where one
        *lands*, so the isolation cannot be vacuous — `record_failure` genuinely
        writes (even below the threshold), and dropping the HERMES_HOME pin puts
        this exact file back on top of the operator's cooldowns.
        """
        bl = Blocklist(BLOCKLIST_CONFIG)
        assert bl.record_failure("canary", "temp-prov", "ttfb_stall") is False
        assert hermetic_state.exists()
        state = json.loads(hermetic_state.read_text(encoding="utf-8"))
        assert "canary@temp-prov" in state["entries"]
        # The real write must remain inside pytest's isolated filesystem.  TMPDIR
        # may itself be under the operator's home, so comparing against Path.home()
        # incorrectly rejects an otherwise hermetic test run.
        assert tmp_path.resolve() in hermetic_state.resolve().parents

    def test_config_ban_still_fires(self):
        bl = Blocklist(BLOCKLIST_CONFIG)
        assert bl.is_blocked("gpt-5.6-sol", "openai-codex") is True

    def test_breaker_blocks_after_trip(self):
        bl = Blocklist(BLOCKLIST_CONFIG)
        model, provider = "some-flaky", "test-prov"
        tripped = bl.record_failure(model, provider, "ttfb_stall")
        assert not tripped
        tripped = bl.record_failure(model, provider, "ttfb_stall")
        assert tripped
        assert bl.is_blocked(model, provider) is True

    def test_expired_cooldown_allows_one_probe_across_fresh_blocklists(self, monkeypatch):
        """Fresh Blocklist instances must not all reopen the same expired breaker.

        ``delegate_profile`` constructs Blocklist per call. When an OPEN breaker
        expires, the first caller gets the HALF_OPEN probe and the consumed probe
        state must be saved before another caller reloads the file.
        """
        import router.blocklist as blocklist_mod

        model, provider = "flaky-probe", "prov"
        key = f"{model}@{provider}"
        clock = {"now": 100.0}
        monkeypatch.setattr(blocklist_mod.time, "time", lambda: clock["now"])
        bl = Blocklist(BLOCKLIST_CONFIG)
        bl.record_failure(model, provider, "ttfb_stall")
        clock["now"] = 110.0
        bl.record_failure(model, provider, "ttfb_stall")

        clock["now"] = 200.0
        assert Blocklist(BLOCKLIST_CONFIG).is_blocked(model, provider) is False
        assert Blocklist(BLOCKLIST_CONFIG).is_blocked(model, provider) is True

        data = json.loads(_state_path().read_text(encoding="utf-8"))
        assert data["entries"][key]["state"] == "HALF_OPEN"
        assert data["entries"][key]["probe_allowed"] is False

    def test_config_ban_fires_with_breaker_cooldown(self):
        bl = Blocklist(BLOCKLIST_CONFIG)
        assert bl.is_blocked("gpt-5.6-sol", "openai-codex") is True

    def test_breaker_disabled(self):
        config = {
            "blocklist": {
                "manual_ban": [],
                "fallback_chain": [],
                "auto_breaker": {"enabled": False},
            }
        }
        bl = Blocklist(config)
        assert not bl.breaker_enabled()
        assert bl.breaker_status() == []
        assert bl.record_failure("m", "p", "ttfb_stall") is False
        assert not bl.is_blocked("m", "p")

    def test_fallback_chain_unchanged(self):
        bl = Blocklist(BLOCKLIST_CONFIG)
        assert bl.fallback_for("gpt-5.6-sol") == "glm-5.2"
        assert bl.fallback_for("glm-5.2") is None

    def test_record_success_resets_breaker(self):
        bl = Blocklist(BLOCKLIST_CONFIG)
        model, provider = "flaky", "prov"
        bl.record_failure(model, provider, "ttfb_stall")
        bl.record_failure(model, provider, "ttfb_stall")
        assert bl.is_blocked(model, provider)
        bl.record_success(model, provider)

    def test_breaker_status(self):
        bl = Blocklist(BLOCKLIST_CONFIG)
        model, provider = "flaky2", "prov2"
        bl.record_failure(model, provider, "ttfb_stall")
        bl.record_failure(model, provider, "ttfb_stall")
        status = bl.breaker_status()
        # Exactly one: `>= 1` was only ever needed because the shared real state
        # could hold anything. A per-test HERMES_HOME means ours is the only entry,
        # so a leak from elsewhere now shows up here instead of being tolerated.
        assert len(status) == 1
        our_entry = [s for s in status if s["model_key"] == f"{model}@{provider}"]
        assert len(our_entry) == 1
        assert our_entry[0]["state"] == "OPEN"

    def test_last_failure_kind_is_set_by_record_not_by_hand(self):
        """record() must stamp last_failure_kind — the CLI reads it right after a trip."""
        bl = Blocklist(BLOCKLIST_CONFIG)
        model, provider = "flaky4", "prov4"
        bl.record_failure(model, provider, "ttfb_stall")
        bl.record_failure(model, provider, "idle_stall")
        status = bl.breaker_status()
        our_entry = [s for s in status if s["model_key"] == f"{model}@{provider}"][0]
        # The most recent failure to hit the entry, whatever tripped it. The
        # attribute is set by BreakerState.record — NOT pre-stamped here, which
        # is exactly the defect this guards: the only writer used to be
        # _Entry.from_dict, so a fresh in-process trip displayed `last_failure=-`.
        assert our_entry["last_failure_kind"] == "idle_stall"

    def test_breaker_state_serialization(self):
        bl = Blocklist(BLOCKLIST_CONFIG)
        model, provider = "flaky3", "prov3"
        bl.record_failure(model, provider, "ttfb_stall")
        bl.record_failure(model, provider, "ttfb_stall")
        state = bl.breaker_state_dict()
        assert "entries" in state
        assert f"{model}@{provider}" in state["entries"]
        assert state["entries"][f"{model}@{provider}"]["state"] == "OPEN"

    def test_fail_closed_no_state_file(self, hermetic_state):
        # A fresh HERMES_HOME per test is what makes the name true — assert it,
        # since the whole point is that a read path never creates the file.
        assert not hermetic_state.exists()
        bl = Blocklist(BLOCKLIST_CONFIG)
        assert bl.is_blocked("gpt-5.6-sol", "openai-codex") is True
        assert bl.is_blocked("claude-opus", "anthropic") is False
        assert not hermetic_state.exists()

    def test_blocked_model_not_blocked_wrong_provider(self):
        bl = Blocklist(BLOCKLIST_CONFIG)
        assert bl.is_blocked("gpt-5.6-sol", "anthropic") is False

    def test_state_dir_peels_profile_scoped_home(self, monkeypatch, tmp_path):
        """Writer (profiles/<name>) and sidecar (bare root) must share ONE state file.

        This is the ONE place the canonical literal is written on the breaker side —
        the trace side asserts AGREEMENT with it rather than repeating it (see
        test_the_two_state_paths_resolve_to_one_directory).
        """
        root = tmp_path
        canonical = root / "hermes-smart-router" / "state"
        # The delegate_profile plugin process runs with a profile-scoped HERMES_HOME...
        monkeypatch.setenv("HERMES_HOME", str(root / "profiles" / "trama-engineer"))
        assert _state_dir() == canonical
        # ...and the sidecar with a bare one — same canonical dir, or a rail's
        # cooldown would live in a file the other profile never reads.
        monkeypatch.setenv("HERMES_HOME", str(root))
        assert _state_dir() == canonical

    @pytest.mark.parametrize("home", [
        "bare", "profile-scoped", "unset",
    ])
    def test_the_two_state_paths_resolve_to_one_directory(
        self, home, monkeypatch, tmp_path,
    ):
        """The breaker file and the trace file must live in the SAME directory.

        They are resolved by two functions that each used to carry their own copy of
        the `profiles/<name>` peel and the directory name — and the copies DIVERGED:
        `d0802d6` added the peel to the trace path, `305a901` added it to the breaker
        path four weeks later, with a commit message describing the production bug
        the gap had caused in between (each profile read a different state file, so
        cooldowns never accumulated). The 2026-08-27 rename then had to touch both
        literals by hand.

        Asserted as AGREEMENT between the two producers over every HERMES_HOME shape,
        not against a literal in a second place — which is what let them drift.
        """
        import router.durable_decision_log as ddl

        # HERMES_ROUTE_TRACE_FILE is deliberately NOT part of this equality: it
        # overrides the trace file only, and breaker state must not follow it or a
        # redirected trace would orphan breaker-state.json somewhere unread.
        monkeypatch.delenv("HERMES_ROUTE_TRACE_FILE", raising=False)
        if home == "bare":
            monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        elif home == "profile-scoped":
            monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / "coder"))
        else:
            monkeypatch.delenv("HERMES_HOME", raising=False)

        assert _state_dir() == ddl.routes_path().parent
        assert _state_dir() == ddl.attempts_path().parent

    def test_the_trace_override_does_not_move_the_breaker_file(
        self, monkeypatch, tmp_path,
    ):
        """The one asymmetry, stated so nobody "fixes" it into symmetry."""
        import router.durable_decision_log as ddl

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        canonical = _state_dir()
        monkeypatch.setenv("HERMES_ROUTE_TRACE_FILE", str(tmp_path / "elsewhere.jsonl"))
        assert ddl.routes_path() == tmp_path / "elsewhere.jsonl"
        assert _state_dir() == canonical, (
            "the breaker file must not follow a trace-file override"
        )


# ---------------------------------------------------------------------------
# _Event / _Entry tests
# ---------------------------------------------------------------------------

class TestEvent:
    def test_to_dict_from_dict(self):
        ev = _Event("ttfb_stall", 100.0, 3)
        d = ev.to_dict()
        ev2 = _Event.from_dict(d)
        assert ev2 is not None
        assert ev2.kind == "ttfb_stall"
        assert ev2.ts == 100.0
        assert ev2.weight == 3

    def test_from_dict_invalid(self):
        assert _Event.from_dict({}) is None
        assert _Event.from_dict({"kind": ""}) is None
        # str(123) = "123" — valid kind, should create event
        ev = _Event.from_dict({"kind": 123})  # type: ignore
        assert ev is not None
        assert ev.kind == "123"


class TestEntry:
    def test_prune(self):
        entry = _Entry()
        entry.events = [
            _Event("a", 100.0, 1),
            _Event("b", 200.0, 1),
            _Event("c", 300.0, 1),
        ]
        entry.prune(350.0, 100.0)  # cutoff = 250
        assert len(entry.events) == 1  # only c survives
        assert entry.events[0].kind == "c"

    def test_total_weight(self):
        entry = _Entry()
        entry.events = [
            _Event("a", 100.0, 3),
            _Event("b", 200.0, 2),
        ]
        assert entry.total_weight() == 5

    def test_to_dict_from_dict_round_trip(self):
        entry = _Entry()
        entry.state = "OPEN"
        entry.events = [_Event("ttfb_stall", 100.0, 3)]
        entry.cooldown_until = 200.0
        entry.backoff_seconds = 60.0
        entry.last_failure_kind = "ttfb_stall"

        data = entry.to_dict()
        entry2 = _Entry.from_dict(data)
        assert entry2 is not None
        assert entry2.state == "OPEN"
        assert len(entry2.events) == 1
        assert entry2.events[0].kind == "ttfb_stall"
        assert entry2.cooldown_until == 200.0
        assert entry2.backoff_seconds == 60.0
        assert entry2.last_failure_kind == "ttfb_stall"

    def test_from_dict_invalid(self):
        assert _Entry.from_dict(None) is None  # type: ignore
        assert _Entry.from_dict("garbage") is None  # type: ignore


# ---------------------------------------------------------------------------
# Looking is not using — a status read must not consume the probe slot
# ---------------------------------------------------------------------------

class TestObservingDoesNotStrandARail:
    """``is_blocked`` is not a query, and three reporting surfaces used it as one.

    An expired OPEN entry transitions to HALF_OPEN on the first ``is_blocked`` and
    BURNS the single probe slot. HALF_OPEN is left only by a RECORDED success or
    failure — so when the caller was merely LOOKING (``router.cli blocklist``,
    ``RouterService.liveness``, ``blocked_entries`` itself), nothing was ever
    dispatched to the rail and it stayed excluded permanently, reporting
    ``cooldown_remaining_s: 0.0`` forever.
    """

    @staticmethod
    def _expired_open(cooldown_at=1000.0):
        """A breaker with one entry OPEN and its cooldown already in the past."""
        b = BreakerState({"enabled": True, "threshold": 2, "window_seconds": 600,
                          "base_cooldown_seconds": 60})
        b.record("rail", "ttfb_stall", cooldown_at)
        b.record("rail", "ttfb_stall", cooldown_at)
        assert b._entries["rail"].state == "OPEN", "the setup must actually trip it"
        return b

    def test_would_block_agrees_with_is_blocked_on_every_state(self):
        """The two must give the same ANSWER and differ only in their effect.

        Asserted per state against a fresh breaker each time, because
        ``is_blocked`` moves the state it is asked about — comparing them on one
        instance would only ever measure the first call.
        """
        later = 1000.0 + 10_000  # well past the cooldown

        # CLOSED, and an entry that does not exist at all.
        fresh = BreakerState({"enabled": True})
        assert fresh.would_block("nope", later) is fresh.is_blocked("nope", later)

        # OPEN, still cooling.
        assert (self._expired_open().would_block("rail", 1000.0)
                is self._expired_open().is_blocked("rail", 1000.0) is True)

        # OPEN, cooldown expired — both say "not blocked" (a probe is due).
        assert (self._expired_open().would_block("rail", later)
                is self._expired_open().is_blocked("rail", later) is False)

        # HALF_OPEN with the slot already consumed — both say blocked.
        consumed = self._expired_open()
        assert consumed.is_blocked("rail", later) is False   # consumes it
        assert consumed._entries["rail"].state == "HALF_OPEN"
        assert consumed.would_block("rail", later) is consumed.is_blocked(
            "rail", later) is True

    def test_would_block_leaves_the_state_untouched(self):
        b = self._expired_open()
        later = 1000.0 + 10_000
        before = b.to_dict()

        for _ in range(5):
            assert b.would_block("rail", later) is False

        assert b.to_dict() == before, "a query moved the state"
        # And the slot is still there for the call that actually dispatches.
        assert b.is_blocked("rail", later) is False
        assert b._entries["rail"].state == "HALF_OPEN"

    def test_listing_the_blocklist_does_not_burn_the_probe(self):
        """``blocked_entries`` is a DISPLAY. It used to consume every slot it drew."""
        b = self._expired_open()
        later = 1000.0 + 10_000

        rows = b.blocked_entries(later)
        assert [row["model_key"] for row in rows] == ["rail"]
        # Reported honestly as OPEN-with-nothing-left, not as the HALF_OPEN the
        # display itself used to cause.
        assert rows[0]["state"] == "OPEN"
        assert rows[0]["cooldown_remaining_s"] == 0.0
        assert b._entries["rail"].state == "OPEN", "drawing the row moved the state"

        # Drawing it a hundred times still leaves the rail probeable.
        for _ in range(100):
            b.blocked_entries(later)
        assert b.is_blocked("rail", later) is False, (
            "the probe slot was consumed by observation, so the rail could never "
            "be retried"
        )

    def test_the_blocklist_wrapper_is_read_only_on_disk_too(
        self, tmp_path, monkeypatch,
    ):
        """``Blocklist.would_block`` reads the freshest state but writes nothing."""
        config = {
            "blocklist": {
                "manual_ban": [{"model": "banned", "provider": "", "reason": "t"}],
                "fallback_chain": [],
                "auto_breaker": {"enabled": True, "threshold": 2,
                                 "window_seconds": 600,
                                 "base_cooldown_seconds": 0},
            }
        }
        bl = Blocklist(config)
        bl.record_failure("rail", "prov", "ttfb_stall")
        bl.record_failure("rail", "prov", "ttfb_stall")
        state = _state_path()
        assert state.exists()

        # A manual ban needs no state at all.
        assert bl.would_block("banned", "") is True

        before = state.read_bytes()
        mtime = state.stat().st_mtime_ns
        # base_cooldown_seconds=0, so the cooldown is already expired: this is
        # exactly the shape where the mutating form would transition and PERSIST.
        for _ in range(5):
            bl.would_block("rail", "prov")
        assert state.read_bytes() == before, "a read path rewrote the state file"
        assert state.stat().st_mtime_ns == mtime

        # The real decision path still gets its one probe.
        assert bl.is_blocked("rail", "prov") is False
        assert state.read_bytes() != before, "the consuming call must persist"

    def test_would_block_reports_a_cooldown_that_is_still_running(self):
        """The breaker half of the answer, on the shape that IS blocked.

        Distinct from the expired-cooldown case above: here ``would_block`` must
        say True, and must say it without touching the entry — a rail mid-cooldown
        has no probe slot to burn yet, and a status read must not create one.
        """
        config = {
            "blocklist": {
                "manual_ban": [],
                "fallback_chain": [],
                "auto_breaker": {"enabled": True, "threshold": 2,
                                 "window_seconds": 600,
                                 "base_cooldown_seconds": 3600},
            }
        }
        bl = Blocklist(config)
        bl.record_failure("rail", "prov", "ttfb_stall")
        bl.record_failure("rail", "prov", "ttfb_stall")

        before = _state_path().read_bytes()
        assert bl.would_block("rail", "prov") is True
        assert bl.would_block("rail", "prov") is True
        assert _state_path().read_bytes() == before

    def test_would_block_skips_the_breaker_entirely_when_it_is_disabled(self):
        """With the breaker off, only manual bans can answer — and no lock is taken."""
        config = {
            "blocklist": {
                "manual_ban": [{"model": "banned", "provider": "", "reason": "t"}],
                "fallback_chain": [],
                "auto_breaker": {"enabled": False},
            }
        }
        bl = Blocklist(config)
        assert bl.breaker_enabled() is False
        assert bl.would_block("banned", "") is True
        assert bl.would_block("anything-else", "prov") is False
        # No state file is created by a read path on a breaker-less config.
        assert not _state_path().exists()


class TestHalfOpenIsNotAbsorbing:
    """A granted probe that is never reported must not strand the rail forever.

    HALF_OPEN's only exits were `record_success` and `record_failure`. So a probe
    granted and then never reported — the process crashed, the attempt was
    abandoned, or (before 2026-09-01) a mere STATUS READ consumed the slot — left
    the rail blocked permanently, reporting `cooldown_remaining_s: 0.0`, which the
    CLI renders as "expiring now" for as long as anyone looks. There is no reset or
    unban command anywhere in this repo, so nothing could recover it.

    The grant now has a deadline of one `backoff_seconds` — the same interval the
    rail was just made to wait. Worst case is a DELAY of one backoff instead of
    permanent exclusion, and the anti-stampede property is unchanged INSIDE the
    window.
    """

    @staticmethod
    def _half_open_at(t):
        """A breaker whose entry is HALF_OPEN with its probe consumed at ``t``."""
        b = BreakerState({"enabled": True, "threshold": 2, "window_seconds": 600,
                          "base_cooldown_seconds": 60})
        b.record("rail", "ttfb_stall", t)
        b.record("rail", "ttfb_stall", t)
        assert b._entries["rail"].state == "OPEN"
        # Past the cooldown: this consumes the probe and stamps the deadline.
        assert b.is_blocked("rail", t + 10_000) is False
        entry = b._entries["rail"]
        assert entry.state == "HALF_OPEN"
        assert entry.probe_allowed is False
        return b, entry

    def test_the_grant_holds_for_one_backoff_then_re_arms(self):
        b, entry = self._half_open_at(1000.0)
        deadline = entry.half_open_until
        assert deadline == 10_000 + 1000.0 + entry.backoff_seconds

        # Inside the window: everyone else waits. That is the anti-stampede half.
        assert b.is_blocked("rail", deadline - 1) is True
        assert b.would_block("rail", deadline - 1) is True

        # At the deadline: a fresh probe, because the last one was never reported.
        assert b.is_blocked("rail", deadline) is False
        assert b._entries["rail"].half_open_until == deadline + entry.backoff_seconds

    def test_a_reported_outcome_still_ends_half_open_immediately(self):
        """The deadline is a backstop, not the normal exit."""
        b, _entry = self._half_open_at(1000.0)
        b.record_success("rail", 1000.0 + 10_001)
        assert b._entries["rail"].state == "CLOSED"
        assert b._entries["rail"].half_open_until == 0.0
        assert b.is_blocked("rail", 1000.0 + 10_002) is False

        b2, _e2 = self._half_open_at(2000.0)
        b2.record("rail", "ttfb_stall", 2000.0 + 10_001)
        assert b2._entries["rail"].state == "OPEN", "a failed probe re-opens"

    def test_would_block_agrees_across_the_new_state(self):
        """The two predicates must not disagree exactly when a rail is recovering."""
        for offset in (-1, 0, 1, 5_000):
            fresh_a, entry = self._half_open_at(3000.0)
            fresh_b, _ = self._half_open_at(3000.0)
            at = entry.half_open_until + offset
            assert fresh_a.would_block("rail", at) is fresh_b.is_blocked("rail", at), (
                f"disagreement at offset {offset}"
            )

    def test_an_entry_from_an_older_state_file_re_arms_rather_than_sticking(self):
        """`half_open_until` is additive, so old files load with 0.0.

        A missing deadline must not be read as an infinite one — an entry already
        stranded in HALF_OPEN on disk has to recover, which is the whole point.
        """
        old = {
            "version": 1,
            "entries": {"rail": {
                "state": "HALF_OPEN",
                "failure_events": [],
                "cooldown_until": 100.0,
                "backoff_seconds": 60.0,
                "last_failure_kind": "ttfb_stall",
                "probe_allowed": False,
                # no half_open_until key at all
            }},
        }
        b = BreakerState.from_dict(old, {"enabled": True})
        assert b._entries["rail"].half_open_until == 0.0
        assert b.would_block("rail", 1_000_000.0) is False
        assert b.is_blocked("rail", 1_000_000.0) is False, (
            "a stranded entry from an older version must recover"
        )

    def test_a_garbage_deadline_on_disk_degrades_to_re_arming(self):
        for bad in ("soon", None, [], {}):
            data = {
                "version": 1,
                "entries": {"rail": {
                    "state": "HALF_OPEN", "failure_events": [],
                    "cooldown_until": 1.0, "backoff_seconds": 60.0,
                    "last_failure_kind": "x", "probe_allowed": False,
                    "half_open_until": bad,
                }},
            }
            b = BreakerState.from_dict(data, {"enabled": True})
            assert b._entries["rail"].half_open_until == 0.0, bad

    def test_the_deadline_round_trips_through_persistence(self):
        b, entry = self._half_open_at(4000.0)
        restored = BreakerState.from_dict(b.to_dict(), {"enabled": True})
        assert restored._entries["rail"].half_open_until == entry.half_open_until
        # And the restored entry answers identically on both sides of the deadline.
        for at in (entry.half_open_until - 1, entry.half_open_until):
            assert restored.would_block("rail", at) is b.would_block("rail", at)

    def test_a_restored_entry_with_an_unconsumed_grant_reads_as_not_blocked(self):
        """A HALF_OPEN entry persisted with `probe_allowed: True` still owes a probe.

        `is_blocked` consumes it; `would_block` must report the same answer without
        consuming — the two have to agree here as everywhere else.
        """
        data = {
            "version": 1,
            "entries": {"rail": {
                "state": "HALF_OPEN", "failure_events": [],
                "cooldown_until": 1.0, "backoff_seconds": 60.0,
                "last_failure_kind": "ttfb_stall", "probe_allowed": True,
                "half_open_until": 1_000_000.0,
            }},
        }
        reading = BreakerState.from_dict(data, {"enabled": True})
        consuming = BreakerState.from_dict(data, {"enabled": True})
        # Not blocked: the grant is still owed, even though the deadline is far off.
        assert reading.would_block("rail", 500.0) is False
        assert consuming.is_blocked("rail", 500.0) is False
        # And only the consuming form spent it.
        assert reading._entries["rail"].probe_allowed is True
        assert consuming._entries["rail"].probe_allowed is False
        assert reading.would_block("rail", 500.0) is False
        assert consuming.would_block("rail", 500.0) is True
