"""Hermetic runtime-path tests for the delegate_profile plugin.

These tests exercise subprocess orchestration with fakes. They never invoke the
Hermes CLI or spawn an OS process; process-tree behaviour itself lives in
``test_delegate_profile.py``.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest import mock

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location("delegate_profile_runtime", REPO_ROOT / "__init__.py")
assert _spec is not None and _spec.loader is not None
_dp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dp)


def test_handler_returns_bad_args_when_router_cannot_resolve_profile(monkeypatch):
    monkeypatch.setattr(_dp, "_route_task", lambda *_args: None)
    handler = _dp._make_handler("parent", lambda _args: "inline")
    result = json.loads(handler({"goal": "task"}))
    assert result == {"error": "profile is required", "failure_kind": "bad_args"}


def test_explicit_auto_profile_is_a_sentinel_not_a_profile_name(monkeypatch):
    """profile="auto" asks the router to choose; it is never a real profile.

    Before the fix the sentinel survived a router decline, reached
    _profile_exists("auto"), and produced "Profile 'auto' does not exist. Create it
    with: hermes profile create auto" - instructing the operator to create a
    profile that would then shadow the sentinel. The decline must read as the same
    bad_args the empty case gives.
    """
    monkeypatch.setattr(_dp, "_route_task", lambda *_args: None)
    handler = _dp._make_handler("parent", lambda _args: "inline")
    result = json.loads(handler({"goal": "task", "profile": "auto"}))
    assert result == {"error": "profile is required", "failure_kind": "bad_args"}
    # And the router still gets to choose when it can.
    monkeypatch.setattr(_dp, "_route_task", lambda *_a: {"profile": "coder", "model": "m"})
    routed = json.loads(handler({"goal": "task", "profile": "auto"}))
    assert routed.get("error") != "profile is required"


def test_close_pipes_and_tail_bound_memory():
    class Pipe:
        def __init__(self, fail=False):
            self.fail = fail
            self.closed = False

        def close(self):
            self.closed = True
            if self.fail:
                raise OSError("already closed")

    class Proc:
        stdout = Pipe()
        stderr = Pipe(fail=True)

    proc = Proc()
    _dp._close_pipes(proc)
    assert proc.stdout.closed and proc.stderr.closed

    tail = _dp._Tail(cap=3)
    tail.append("abcd")
    tail.append("efgh")
    assert tail.text() == "fgh"


def test_kill_tree_windows_and_posix_lookup_error(monkeypatch):
    class Proc:
        pid = 99
        stdout = None
        stderr = None

        def poll(self):
            return None

        def wait(self, timeout):
            self.wait_timeout = timeout

    proc = Proc()
    called = []
    monkeypatch.setattr(_dp, "IS_WINDOWS", True)
    monkeypatch.setattr(_dp.subprocess, "run", lambda args, **kwargs: called.append((args, kwargs)))
    _dp._kill_tree(proc, None, 0.1)
    assert called[0][0][:3] == ["taskkill", "/F", "/T"]

    monkeypatch.setattr(_dp, "IS_WINDOWS", False)
    monkeypatch.setattr(_dp.os, "getpgid", lambda _pid: (_ for _ in ()).throw(ProcessLookupError()))
    _dp._kill_tree(Proc(), None, 0.1)


def test_get_pool_registers_once_and_pool_swallows_kill_error(monkeypatch):
    registrations = []
    monkeypatch.setattr(_dp, "_POOL", None)
    monkeypatch.setattr(_dp.atexit, "register", lambda fn: registrations.append(fn))
    pool = _dp._get_pool()
    assert pool is _dp._get_pool()
    assert registrations == [pool.kill_all]

    class Proc:
        pid = 1

    pool.register(Proc(), 1, {"profile": "child"})
    monkeypatch.setattr(_dp, "_kill_tree", lambda *_args: (_ for _ in ()).throw(RuntimeError("kill")))
    pool.kill_all()
    assert pool.snapshot() == []


def test_register_exposes_schema_and_hook(monkeypatch):
    class Ctx:
        def __init__(self):
            self.tools = []
            self.hooks = []

        def dispatch_tool(self, name, args):
            return f"{name}:{args['goal']}"

        def register_tool(self, **kwargs):
            self.tools.append(kwargs)

        def register_hook(self, *args, **kwargs):
            self.hooks.append((args, kwargs))

    monkeypatch.setattr(_dp, "_get_active_profile_name", lambda: "parent")
    ctx = Ctx()
    _dp.register(ctx)
    assert ctx.tools[0]["name"] == "delegate_profile"
    assert ctx.tools[0]["schema"]["parameters"]["required"] == ["prompt"]
    assert ctx.hooks


def test_register_schema_accepts_prompt_and_enumerates_profiles(monkeypatch):
    """The schema's canonical text field is `prompt`; `goal` remains an alias.

    profile carries an enum of known profile names (plus the implicit
    `default` and the `auto` router sentinel), never an empty list.
    """
    monkeypatch.setattr(_dp, "_get_active_profile_name", lambda: "parent")
    monkeypatch.setattr(_dp, "_list_known_profiles", lambda: ["coder", "reviewer"])

    class Ctx:
        def __init__(self):
            self.tools = []

        def register_tool(self, **kwargs):
            self.tools.append(kwargs)

        def register_hook(self, *_args, **_kwargs):
            pass

    ctx = Ctx()
    _dp.register(ctx)
    props = ctx.tools[0]["schema"]["parameters"]["properties"]
    assert "prompt" in props and "goal" in props
    assert props["profile"]["enum"] == ["auto", "coder", "default", "reviewer"]


def test_the_schema_tells_the_model_the_timeout_default_the_code_uses(monkeypatch):
    """A number the MODEL reads must not live only in prose.

    The description said "Default: 300 (5 min)" while ``_DEFAULT_TIMEOUT_S`` has
    been 600, so a caller sizing a task to fit "5 minutes" was budgeting against
    half the real ceiling. Nothing could catch it: the 300 appeared in no code path,
    only in a sentence. It is interpolated now, and this asserts the two agree
    rather than asserting either literal.
    """
    monkeypatch.setattr(_dp, "_get_active_profile_name", lambda: "parent")

    class Ctx:
        def __init__(self):
            self.tools = []

        def register_tool(self, **kwargs):
            self.tools.append(kwargs)

        def register_hook(self, *_args, **_kwargs):
            pass

    ctx = Ctx()
    _dp.register(ctx)
    described = ctx.tools[0]["schema"]["parameters"]["properties"]["timeout"][
        "description"
    ]
    assert f"Default: {_dp._DEFAULT_TIMEOUT_S}" in described, described
    # And the resolver really does hand back that number, so the sentence is not
    # merely self-consistent with a constant nothing uses.
    assert _dp._resolve_timeout(None) == _dp._DEFAULT_TIMEOUT_S


def test_the_capacity_error_names_the_key_the_reader_reads():
    """The operator-facing remedy must name a key that exists.

    ``at_capacity`` told whoever hit the concurrency cap to raise
    ``plugins.entries.delegate-profile.watchdog.max_concurrent`` while
    ``_watchdog_cfg`` had been reading ``hermes-smart-router`` since the 2026-08-27
    rename — so following the error message edited a key nothing reads, and the cap
    never moved. Both sides come from ``_CONFIG_NAMESPACE`` now.
    """
    assert _dp._CONFIG_NAMESPACE == "hermes-smart-router"
    assert _dp._WATCHDOG_CFG_PATH == (
        f"plugins.entries.{_dp._CONFIG_NAMESPACE}.watchdog"
    )

    captured = {}

    def _cfg_get(config, *path, default=None):
        captured["path"] = path
        return default

    fake = types.ModuleType("hermes_cli.config")
    fake.cfg_get = _cfg_get
    fake.load_config_readonly = lambda: {}
    parent = types.ModuleType("hermes_cli")
    parent.config = fake
    with mock.patch.dict(
        sys.modules, {"hermes_cli": parent, "hermes_cli.config": fake},
    ):
        _dp._watchdog_cfg()

    # The reader's actual path, and the message's advertised path, agree.
    assert captured["path"] == (
        "plugins", "entries", _dp._CONFIG_NAMESPACE, "watchdog",
    )
    assert ".".join(captured["path"][:-1] + ("watchdog",)) == _dp._WATCHDOG_CFG_PATH


def test_the_dispatching_classifier_default_is_the_documented_one():
    """One authority for the pair the classifier runs on.

    ``_make_classify_fn`` is the copy that DISPATCHES, and it defaulted to
    ``glm-5.2`` while ``router/classify.py`` said ``glm-5.3-flash``. glm-5.2 is
    registry-marked "plan auto-routes this id to glm-5.3": the call succeeds, the
    plan bills the substitute, and every trace names an id nobody ran. Reachable
    whenever ``classifier.model`` was absent from router.yaml.
    """
    from router.capabilities import MODEL_CAPABILITIES
    from router.classify import CLASSIFIER_DEFAULTS, Classifier

    # The plugin's defaults ARE the classifier module's, not a second copy.
    assert _dp._classifier_defaults() == CLASSIFIER_DEFAULTS
    # A copy, so no caller can edit the shared table.
    mutated = _dp._classifier_defaults()
    mutated["model"] = "tampered"
    assert CLASSIFIER_DEFAULTS["model"] != "tampered"
    # Both consumers land on the same pair for a config that names none.
    assert Classifier({}).model == CLASSIFIER_DEFAULTS["model"]
    assert Classifier({}).provider == CLASSIFIER_DEFAULTS["provider"]
    # And that model is not one the vendor silently substitutes.
    notes = str(MODEL_CAPABILITIES[CLASSIFIER_DEFAULTS["model"]].get("notes", ""))
    assert "auto-routes this id" not in notes, (
        f"the classifier default {CLASSIFIER_DEFAULTS['model']!r} is a vendor "
        f"alias: {notes}"
    )


def test_register_is_idempotent_per_context(monkeypatch):
    """A second register() with the SAME ctx is a no-op — no duplicate hooks."""
    monkeypatch.setattr(_dp, "_get_active_profile_name", lambda: "parent")

    class Ctx:
        def __init__(self):
            self.tools = []
            self.hooks = []

        def dispatch_tool(self, name, args):
            return f"{name}:{args['goal']}"

        def register_tool(self, **kwargs):
            self.tools.append(kwargs)

        def register_hook(self, *args, **kwargs):
            self.hooks.append((args, kwargs))

    ctx = Ctx()
    _dp.register(ctx)
    first_hooks = list(ctx.hooks)
    _dp.register(ctx)
    assert ctx.hooks == first_hooks, "hooks must not be duplicated on re-register"
    assert len(ctx.tools) == 1, "tool must not be re-registered"


def test_register_serves_fresh_contexts(monkeypatch):
    """A different ctx object registers fresh (host reload / test fake)."""
    monkeypatch.setattr(_dp, "_get_active_profile_name", lambda: "parent")

    class Ctx:
        def __init__(self):
            self.tools = []
            self.hooks = []

        def register_tool(self, **kwargs):
            self.tools.append(kwargs)

        def register_hook(self, *_args, **_kwargs):
            self.hooks.append(True)

    first, second = Ctx(), Ctx()
    _dp.register(first)
    _dp.register(second)
    assert second.tools and second.hooks, "fresh ctx must be registered"


def test_resolve_active_profile_and_profile_fallbacks(monkeypatch, tmp_path):
    import types

    profiles = types.ModuleType("hermes_cli.profiles")
    profiles.get_active_profile_name = lambda: "active"
    profiles.profile_exists = lambda name: name == "real"
    cli_package = types.ModuleType("hermes_cli")
    cli_package.profiles = profiles
    monkeypatch.setitem(sys.modules, "hermes_cli", cli_package)
    monkeypatch.setitem(sys.modules, "hermes_cli.profiles", profiles)
    assert _dp._get_active_profile_name() == "active"
    assert _dp._profile_exists("real")
    assert not _dp._profile_exists("missing")

    class Home:
        def __truediv__(self, _other):
            return tmp_path

    constants = types.ModuleType("hermes_constants")
    constants.get_hermes_home = lambda: Home()
    monkeypatch.setitem(sys.modules, "hermes_constants", constants)
    monkeypatch.delitem(sys.modules, "hermes_cli.profiles")
    monkeypatch.delitem(sys.modules, "hermes_cli")
    (tmp_path / "child").mkdir()
    assert _dp._profile_exists("child")
    assert not _dp._profile_exists("missing")


def test_resolve_hermes_bin_and_list_profiles_fallback(monkeypatch, tmp_path):
    expected = tmp_path / "hermes"
    monkeypatch.setattr(_dp.sys, "executable", str(tmp_path / "python"))
    expected.write_text("", encoding="utf-8")
    assert _dp._resolve_hermes_bin() == str(expected)

    import types
    constants = types.ModuleType("hermes_constants")
    constants.get_hermes_home = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "hermes_constants", constants)
    monkeypatch.setitem(sys.modules, "hermes_cli", types.ModuleType("hermes_cli"))
    (tmp_path / "profiles").mkdir()
    (tmp_path / "profiles" / "one").mkdir()
    assert _dp._list_known_profiles() == ["one"]


def test_spawn_windows_creation_flag(monkeypatch):
    captured = {}

    class Popen:
        def __init__(self, cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs

    monkeypatch.setattr(_dp, "IS_WINDOWS", True)
    monkeypatch.setattr(_dp.subprocess, "Popen", Popen)
    _dp._spawn(["cmd"], {"X": "1"})
    assert "creationflags" in captured["kwargs"]
    assert "start_new_session" not in captured["kwargs"]


def test_kill_tree_posix_escalates_and_tolerates_killpg_error(monkeypatch):
    class Proc:
        pid = 11
        stdout = None
        stderr = None
        calls = 0

        def poll(self):
            return None

        def wait(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise _dp.subprocess.TimeoutExpired("cmd", timeout)

    signals = []
    monkeypatch.setattr(_dp, "IS_WINDOWS", False)
    monkeypatch.setattr(_dp.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)))
    _dp._kill_tree(Proc(), 11, 0.1)
    assert [sig for _, sig in signals] == [_dp.signal.SIGTERM, _dp.signal.SIGKILL]

    monkeypatch.setattr(_dp.os, "killpg", lambda *_args: (_ for _ in ()).throw(OSError("gone")))
    _dp._kill_tree(Proc(), 11, 0.1)

    monkeypatch.setattr(_dp.os, "killpg", lambda *_args: (_ for _ in ()).throw(ProcessLookupError()))
    _dp._kill_tree(Proc(), 11, 0.1)

    class TimeoutProc:
        pid = 12
        stdout = None
        stderr = None

        def poll(self):
            return None

        def wait(self, timeout):
            raise _dp.subprocess.TimeoutExpired("cmd", timeout)

    monkeypatch.setattr(_dp.os, "killpg", lambda *_args: None)
    _dp._kill_tree(TimeoutProc(), 12, 0.1)


def test_breaker_outcome_no_model_and_error_is_nonblocking(monkeypatch):
    _dp._record_breaker_outcome("child", "", "crash")
    monkeypatch.setattr(_dp, "_load_router_config", lambda: (_ for _ in ()).throw(RuntimeError("bad")))
    _dp._record_breaker_outcome("child", "m", "crash")


def test_register_dispatches_same_profile_through_context(monkeypatch):
    class Ctx:
        def __init__(self):
            self.tool = None

        def dispatch_tool(self, name, args):
            return json.dumps({"name": name, "goal": args["goal"]})

        def register_tool(self, **kwargs):
            self.tool = kwargs

        def register_hook(self, *_args):
            pass

    monkeypatch.setattr(_dp, "_get_active_profile_name", lambda: "parent")
    monkeypatch.setattr(_dp, "_profile_exists", lambda _profile: True)
    ctx = Ctx()
    _dp.register(ctx)
    assert ctx.tool is not None
    result = json.loads(ctx.tool["handler"]({"goal": "task", "profile": "parent"}))
    assert result == {"name": "delegate_task", "goal": "task"}


def test_list_known_profiles_and_windows_kill_failures(monkeypatch, tmp_path):
    import types

    profiles = types.ModuleType("hermes_cli.profiles")
    profiles.list_profiles = lambda: []
    cli_package = types.ModuleType("hermes_cli")
    cli_package.profiles = profiles
    monkeypatch.setitem(sys.modules, "hermes_cli", cli_package)
    monkeypatch.setitem(sys.modules, "hermes_cli.profiles", profiles)
    assert _dp._list_known_profiles() == []

    monkeypatch.delitem(sys.modules, "hermes_cli.profiles")
    monkeypatch.delitem(sys.modules, "hermes_cli")
    constants = types.ModuleType("hermes_constants")
    constants.get_hermes_home = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "hermes_constants", constants)
    assert _dp._list_known_profiles() == []

    class Proc:
        pid = 42
        stdout = None
        stderr = None

        def poll(self):
            return None

        def kill(self):
            raise RuntimeError("already gone")

        def wait(self, _timeout):
            raise RuntimeError("already gone")

    monkeypatch.setattr(_dp, "IS_WINDOWS", True)
    monkeypatch.setattr(_dp.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("taskkill")))
    _dp._kill_tree(Proc(), None, 0.1)


def test_run_watched_tolerates_closed_pipes_and_late_reap(monkeypatch):
    class Pipe:
        def readline(self):
            raise ValueError("closed")

        def close(self):
            pass

    class Proc:
        stdout = Pipe()
        stderr = Pipe()
        returncode = 0

        def poll(self):
            return 0

        def wait(self, timeout):
            raise _dp.subprocess.TimeoutExpired("cmd", timeout)

    killed = []
    monkeypatch.setattr(_dp, "IS_WINDOWS", False)
    monkeypatch.setattr(_dp, "_kill_tree", lambda *_args: killed.append(True))
    result = _dp._run_watched(Proc(), 1, 0.1, 0.1, 0.1, 0.01)
    assert result[0] == "exited"
    assert killed == [True]

def test_cross_profile_survives_missing_pgid_and_breaker_scans_later_tier(monkeypatch):
    record_outcome = _dp._record_breaker_outcome
    handler, _pool = _cross_handler(monkeypatch, ("exited", 0, "done", ""))
    monkeypatch.setattr(_dp, "IS_WINDOWS", False)
    monkeypatch.setattr(_dp.os, "getpgid", lambda _pid: (_ for _ in ()).throw(OSError("gone")))
    assert json.loads(handler({"goal": "task", "profile": "child"}))["success"] is True

    calls = []

    class Blocklist:
        def __init__(self, _config):
            pass

        def record_success(self, *args):
            calls.append(args)

    import router.blocklist
    monkeypatch.setattr(_dp, "_record_breaker_outcome", record_outcome)
    monkeypatch.setattr(router.blocklist, "Blocklist", Blocklist)
    monkeypatch.setattr(
        _dp,
        "_load_router_config",
        lambda: {"tiers": {"T1": {"model": "other"}, "T2": {"model": "m", "provider": "later"}}},
    )
    _dp._record_breaker_outcome("child", "m", None)
    assert calls == [("m", "later")]

    monkeypatch.setattr(_dp, "_load_router_config", lambda: {"tiers": {}})
    _dp._record_breaker_outcome("child", "unknown", None)
    assert calls[-1] == ("unknown", "")


class FakeProcess:
    pid = 4321


class FakePool:
    def __init__(self, acquire_result: bool = True) -> None:
        self.acquire_result = acquire_result
        self.registered = []
        self.unregistered = []
        self.released = 0
        self.capacity = 4  # mirrors _Pool.capacity (used in at_capacity error text)

    def acquire(self, _wait: float) -> bool:
        return self.acquire_result

    def register(self, proc, pgid, meta) -> None:
        self.registered.append((proc, pgid, meta))

    def unregister(self, pid: int) -> None:
        self.unregistered.append(pid)

    def release(self) -> None:
        self.released += 1


def _cross_handler(monkeypatch, watched_result=("exited", 0, "result", ""), *, pool=None):
    """Build a cross-profile handler with all host/process seams faked."""
    test_pool = pool or FakePool()
    monkeypatch.setattr(_dp, "_profile_exists", lambda _profile: True)
    monkeypatch.setattr(_dp, "_resolve_hermes_bin", lambda: "hermes")
    monkeypatch.setattr(_dp, "_resolve_ladder", lambda _hard: (1.0, 2.0, 3.0, 0.1))
    monkeypatch.setattr(_dp, "_get_pool", lambda: test_pool)
    monkeypatch.setattr(_dp, "_spawn", lambda _cmd, _env: FakeProcess())
    monkeypatch.setattr(_dp.os, "getpgid", lambda _pid: 4321)
    monkeypatch.setattr(_dp, "_run_watched", lambda *_args: watched_result)
    monkeypatch.setattr(_dp, "_kill_tree", lambda *_args: None)
    monkeypatch.setattr(_dp, "_record_breaker_outcome", lambda *_args: None)
    return _dp._make_handler("parent", lambda _args: "inline"), test_pool


@pytest.mark.parametrize(
    ("watched_result", "expected_kind", "expected_fragment"),
    [
        (("hard_timeout", None, "", "diagnostic"), "hard_timeout", "Hard timeout"),
        (("ttfb_timeout", None, "partial", "diagnostic"), "ttfb_stall", "produced no output"),
        (("idle_timeout", None, "partial", "diagnostic"), "idle_stall", "went silent"),
        (("exited", -9, "", "diagnostic"), "crash_or_oom", "exited abnormally"),
        (("exited", 23, "", "diagnostic"), "nonzero_exit", "exited abnormally"),
    ],
)
def test_cross_profile_failure_envelopes(monkeypatch, watched_result, expected_kind, expected_fragment):
    handler, pool = _cross_handler(monkeypatch, watched_result)
    result = json.loads(handler({"goal": "task", "profile": "child", "model": "model-x"}))
    assert result["success"] is False
    assert result["failure_kind"] == expected_kind
    assert expected_fragment in result["error"]
    assert pool.released == 1
    assert pool.unregistered == [FakeProcess.pid]


def test_cross_profile_success_envelope_and_command(monkeypatch):
    captured = {}
    handler, pool = _cross_handler(monkeypatch, ("exited", 0, "done", ""))

    def spawn(cmd, env):
        captured["cmd"] = cmd
        captured["env"] = env
        return FakeProcess()

    monkeypatch.setattr(_dp, "_spawn", spawn)
    result = json.loads(handler({"goal": "task", "context": "background", "profile": "child", "model": "model-x"}))
    assert result["success"] is True
    assert result["result"] == "done"
    assert captured["cmd"] == ["hermes", "-p", "child", "chat", "-q", "Context: background\n\nTask: task", "-m", "model-x"]
    assert captured["env"]["HERMES_PROFILE"] == "child"
    assert captured["env"]["HERMES_DELEGATE_PROFILE_DISABLE"] == "1"
    assert pool.released == 1


def test_cross_profile_treats_quota_exhaustion_with_zero_exit_as_retryable(monkeypatch):
    output = """Initializing agent...
API call failed after 3 retries: HTTP 429 rate limited
Session: child-session
"""
    handler, _pool = _cross_handler(monkeypatch, ("exited", 0, output, ""))

    result = json.loads(
        handler({"goal": "task", "profile": "child", "model": "model-x"})
    )

    assert result["success"] is False
    assert result["failure_kind"] == "quota_exhausted"
    assert result["retryable"] is True
    assert "quota exhausted" in result["error"]
    assert "HTTP 429" in result["partial_output"]


def test_cross_profile_refuses_when_pool_is_at_capacity(monkeypatch):
    handler, pool = _cross_handler(monkeypatch, pool=FakePool(acquire_result=False))
    result = json.loads(handler({"goal": "task", "profile": "child", "timeout": 2}))
    assert result["failure_kind"] == "at_capacity"
    assert pool.released == 0


def test_cross_profile_queue_wait_from_config(monkeypatch):
    """queue_wait_seconds from config.yaml feeds the pool acquire wait (0 accepted)."""
    calls = []
    test_pool = FakePool()
    # Track the wait value passed to acquire.
    def _tracking_acquire(_wait: float) -> bool:
        calls.append(_wait)
        return True
    test_pool.acquire = _tracking_acquire
    handler, _ = _cross_handler(monkeypatch, pool=test_pool)
    # _cross_handler stubs _get_pool to return test_pool, but _watchdog_cfg is
    # the real one; stub it so queue_wait_seconds resolves from "config".
    monkeypatch.setattr(_dp, "_watchdog_cfg", lambda: {"queue_wait_seconds": 5}, raising=False)
    result = json.loads(handler({"goal": "task", "profile": "child"}))
    assert result["success"] is True
    assert calls == [5.0], f"acquire should see queue_wait from config, got {calls}"


def test_cross_profile_queue_wait_zero_means_up_to_hard_ceiling(monkeypatch):
    """queue_wait_seconds=0 must be honoured as 'wait up to the hard ceiling'."""
    calls = []
    test_pool = FakePool()
    def _tracking_acquire(_wait: float) -> bool:
        calls.append(_wait)
        return True
    test_pool.acquire = _tracking_acquire
    handler, _ = _cross_handler(monkeypatch, pool=test_pool)
    # _resolve_ladder is stubbed to (1.0, 2.0, 3.0, 0.1) in _cross_handler, so
    # hard=3.0: queue_wait=0 must pass hard (3.0) to acquire, not 0.
    monkeypatch.setattr(_dp, "_watchdog_cfg", lambda: {"queue_wait_seconds": 0}, raising=False)
    result = json.loads(handler({"goal": "task", "profile": "child"}))
    assert result["success"] is True
    assert calls == [3.0], f"acquire should see the hard ceiling when queue_wait=0, got {calls}"


def test_cross_profile_agent_error_on_zero_exit_with_failure_banner(monkeypatch):
    """A child that exits 0 but printed the CLI's post-retry error banner (and
    is NOT a quota exhaustion) is surfaced as a retryable agent_error, not a
    false success."""
    output = "Working…\nAPI call failed after 3 retries: provider returned garbage\n"
    handler, pool = _cross_handler(monkeypatch, ("exited", 0, output, ""))
    result = json.loads(handler({"goal": "task", "profile": "child", "model": "model-x"}))
    assert result["success"] is False
    assert result["failure_kind"] == "agent_error"
    assert result["retryable"] is True
    assert "exiting with code 0" in result["error"]


def test_cross_profile_preserves_existing_hermes_home(monkeypatch):
    """When HERMES_HOME is already set, the child inherits it unchanged (the
    resolve-and-inject branch is skipped)."""
    captured = {}
    handler, _pool = _cross_handler(monkeypatch, ("exited", 0, "done", ""))
    monkeypatch.setenv("HERMES_HOME", "/preset/hermes/home")
    monkeypatch.setattr(
        _dp, "_spawn",
        lambda _cmd, env: captured.setdefault("env", env) and FakeProcess() or FakeProcess(),
    )
    result = json.loads(handler({"goal": "task", "profile": "child"}))
    assert result["success"] is True
    assert captured["env"]["HERMES_HOME"] == "/preset/hermes/home"


@pytest.mark.parametrize(
    ("spawn_error", "expected_kind"),
    [(FileNotFoundError(), "binary_not_found"), (RuntimeError("boom"), "spawn_error")],
)
def test_cross_profile_spawn_errors_are_structured(monkeypatch, spawn_error, expected_kind):
    handler, pool = _cross_handler(monkeypatch)

    def fail_spawn(_cmd, _env):
        raise spawn_error

    monkeypatch.setattr(_dp, "_spawn", fail_spawn)
    result = json.loads(handler({"goal": "task", "profile": "child"}))
    assert result["failure_kind"] == expected_kind
    assert pool.released == 1


def test_handler_router_without_model_windows_group_and_hermes_home(monkeypatch, tmp_path):
    handler, pool = _cross_handler(monkeypatch, ("exited", 0, "done", ""))
    monkeypatch.setattr(_dp, "_route_task", lambda *_args: {"profile": "child"})
    routed = json.loads(handler({"goal": "task"}))
    assert routed["success"] is True
    assert pool.registered[0][1] == 4321

    handler, windows_pool = _cross_handler(monkeypatch, ("exited", 0, "done", ""))
    monkeypatch.setattr(_dp, "IS_WINDOWS", True)
    windows = json.loads(handler({"goal": "task", "profile": "child"}))
    assert windows["success"] is True
    assert windows_pool.registered[0][1] is None

    captured = {}
    handler, _pool = _cross_handler(monkeypatch, ("exited", 0, "done", ""))
    monkeypatch.setattr(_dp, "IS_WINDOWS", True)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    import types
    constants = types.ModuleType("hermes_constants")
    constants.get_hermes_home = lambda: tmp_path
    monkeypatch.setitem(sys.modules, "hermes_constants", constants)
    monkeypatch.setattr(_dp, "_spawn", lambda _cmd, env: captured.setdefault("env", env) and FakeProcess())
    result = json.loads(handler({"goal": "task", "profile": "child"}))
    assert result["success"] is True
    assert captured["env"]["HERMES_HOME"] == str(tmp_path)

    failed_home = {}
    handler, _pool = _cross_handler(monkeypatch, ("exited", 0, "done", ""))
    monkeypatch.setattr(_dp, "IS_WINDOWS", True)
    monkeypatch.delenv("HERMES_HOME", raising=False)
    constants.get_hermes_home = lambda: (_ for _ in ()).throw(RuntimeError("unavailable"))

    def spawn_without_home(_cmd, env):
        failed_home["env"] = env
        return FakeProcess()

    monkeypatch.setattr(_dp, "_spawn", spawn_without_home)
    result = json.loads(handler({"goal": "task", "profile": "child"}))
    assert result["success"] is True
    assert "HERMES_HOME" not in failed_home["env"]


def test_load_router_config_handles_missing_and_invalid_yaml(monkeypatch, tmp_path):
    monkeypatch.setattr(_dp, "__file__", str(tmp_path / "plugin.py"))
    assert _dp._load_router_config() == {}
    (tmp_path / "router.yaml").write_text("not: [valid", encoding="utf-8")
    assert _dp._load_router_config() == {}


def test_classifier_rejects_non_object_json(monkeypatch):
    monkeypatch.setattr(
        _dp,
        "_load_router_config",
        lambda: {"enabled": True, "classifier": {"model": "m", "provider": "p"}},
    )

    class Result:
        text = "[]"

    class LLM:
        def complete(self, **_kwargs):
            return Result()

    class Ctx:
        llm = LLM()

    fn = _dp._make_classify_fn(Ctx())
    assert fn is not None
    with pytest.raises(ValueError, match="JSON object"):
        fn("task", {})


def test_route_task_rejects_incomplete_and_handles_adapter_exception(monkeypatch):
    monkeypatch.setattr(_dp, "_load_router_config", lambda: {"enabled": True})
    import router.adapter

    monkeypatch.setattr(router.adapter, "route", lambda **_kwargs: {"model": "m"})
    assert _dp._route_task("task", "", None) is None

    monkeypatch.setattr(router.adapter, "route", lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    assert _dp._route_task("task", "", None) is None


def test_route_task_persists_trace_and_survives_durable_log_failure(monkeypatch, tmp_path):
    """The live routing hook passes a DurableDecisionLog so real decisions are
    persisted for replay — and a broken durable log never breaks routing."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(_dp, "_load_router_config", lambda: {
        "enabled": True,
        "default": {"profile": "coder", "model": "T1"},
        "tiers": {"T1": {"model": "m1", "provider": "p1"}, "T2": {}, "T3": {}, "T4": {}},
        "rules": [{"id": "any", "when": {}, "then": {"profile": "coder", "model": "T1"}}],
    })
    # A concrete route → the durable log writes routes.jsonl.
    result = _dp._route_task("do a thing", "", None)
    assert result is not None and result["profile"] == "coder"
    import router.durable_decision_log as ddl
    trace = ddl.routes_path()
    assert trace.exists()
    assert trace.read_text(encoding="utf-8").strip()  # at least one line

    # If DurableDecisionLog construction itself blows up, routing still returns
    # (the whole hook is best-effort under try/except).
    monkeypatch.setattr(
        ddl, "DurableDecisionLog",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("ctor boom")),
    )
    # Import path inside _route_task re-imports the module attribute, so patching
    # the class on the module is what the hook sees.
    result2 = _dp._route_task("do a thing", "", None)
    assert result2 is None or result2.get("profile") == "coder"


# ---------------------------------------------------------------------------
# The executor must attempt the router's PLANNED chain
# ---------------------------------------------------------------------------
#
# The declared order is what the executor used to rebuild, which left the
# capability filter and `fallback_strategy` with no effect on real traffic: a
# vision turn was sent to a tier primary that cannot see images while the
# console showed the filtered chain. These tests pin the plan as the authority.

_BLIND_PRIMARY = {"model": "glm-5.3", "provider": "zai"}
_DECLARED_FALLBACK = [
    {"model": "gpt-5.6-luna", "provider": "openai-codex"},
    {"model": "deepseek-v4-flash", "provider": "deepseek"},
]


def _spawn_recorder(monkeypatch):
    """Capture every spawned argv, returning the list."""
    cmds: list = []

    def spawn(cmd, _env):
        cmds.append(list(cmd))
        return FakeProcess()

    monkeypatch.setattr(_dp, "_spawn", spawn)
    return cmds


def _models_attempted(cmds):
    return [cmd[cmd.index("-m") + 1] for cmd in cmds if "-m" in cmd]


def _providers_attempted(cmds):
    return [cmd[cmd.index("--provider") + 1] for cmd in cmds if "--provider" in cmd]


def test_executor_attempts_the_planned_chain_not_the_declared_order(monkeypatch):
    """The plan drops the blind primary; production must not attempt it at all."""
    handler, _pool = _cross_handler(monkeypatch, ("exited", 0, "done", ""))
    cmds = _spawn_recorder(monkeypatch)
    monkeypatch.setattr(_dp, "_route_task", lambda *_args: {
        "profile": "child", **_BLIND_PRIMARY,
        "fallback": list(_DECLARED_FALLBACK),
        "chain": [{"model": "gpt-5.6-luna", "provider": "openai-codex"}],
    })

    result = json.loads(handler({"goal": "look at this screenshot"}))

    assert result["success"] is True
    assert result["model"] == "gpt-5.6-luna"
    assert _models_attempted(cmds) == ["gpt-5.6-luna"]
    assert _providers_attempted(cmds) == ["openai-codex"]


def test_executor_fails_over_along_the_planned_order(monkeypatch):
    """A retryable failure walks the PLAN's order, not the declared one."""
    handler, _pool = _cross_handler(monkeypatch)
    cmds = _spawn_recorder(monkeypatch)
    outcomes = iter([("exited", -9, "", "boom"), ("exited", 0, "done", "")])
    monkeypatch.setattr(_dp, "_run_watched",
                        lambda *_args: next(outcomes, ("exited", 0, "done", "")))
    monkeypatch.setattr(_dp, "_route_task", lambda *_args: {
        "profile": "child", **_BLIND_PRIMARY,
        "fallback": list(_DECLARED_FALLBACK),
        # A shuffled plan: neither hop is the declared primary, and the order is
        # the reverse of the declared tail.
        "chain": [{"model": "deepseek-v4-flash", "provider": "deepseek"},
                  {"model": "gpt-5.6-luna", "provider": "openai-codex"}],
    })

    result = json.loads(handler({"goal": "task"}))

    assert result["success"] is True
    assert _models_attempted(cmds) == ["deepseek-v4-flash", "gpt-5.6-luna"]
    assert [a["model"] for a in result["attempts"]] == ["deepseek-v4-flash",
                                                       "gpt-5.6-luna"]


def test_executor_uses_the_declared_order_when_there_is_no_plan(monkeypatch):
    """Back-compat: a decision without a chain still fails over as it always did."""
    handler, _pool = _cross_handler(monkeypatch)
    cmds = _spawn_recorder(monkeypatch)
    outcomes = iter([("exited", -9, "", "boom"), ("exited", 0, "done", "")])
    monkeypatch.setattr(_dp, "_run_watched",
                        lambda *_args: next(outcomes, ("exited", 0, "done", "")))
    monkeypatch.setattr(_dp, "_route_task", lambda *_args: {
        "profile": "child", **_BLIND_PRIMARY, "fallback": list(_DECLARED_FALLBACK),
    })

    result = json.loads(handler({"goal": "task"}))

    assert result["success"] is True
    assert _models_attempted(cmds) == ["glm-5.3", "gpt-5.6-luna"]


def test_an_explicitly_requested_model_stays_the_first_attempt(monkeypatch):
    """An explicit model overrides the routing decision; the plan is its tail."""
    handler, _pool = _cross_handler(monkeypatch)
    cmds = _spawn_recorder(monkeypatch)
    outcomes = iter([("exited", -9, "", "boom"), ("exited", 0, "done", "")])
    monkeypatch.setattr(_dp, "_run_watched",
                        lambda *_args: next(outcomes, ("exited", 0, "done", "")))
    monkeypatch.setattr(_dp, "_route_task", lambda *_args: {
        "profile": "child", **_BLIND_PRIMARY,
        "chain": [{"model": "gpt-5.6-luna", "provider": "openai-codex"}],
    })

    result = json.loads(handler({"goal": "task", "model": "operator-choice"}))

    assert result["success"] is True
    assert _models_attempted(cmds) == ["operator-choice", "gpt-5.6-luna"]


def test_a_target_is_never_attempted_twice(monkeypatch):
    """A repeat target is a wasted subprocess and a second breaker strike."""
    handler, _pool = _cross_handler(monkeypatch)
    cmds = _spawn_recorder(monkeypatch)
    # Retryable on purpose: without the dedupe the loop WOULD try again.
    monkeypatch.setattr(_dp, "_run_watched", lambda *_args: ("exited", -9, "", "boom"))
    monkeypatch.setattr(_dp, "_route_task", lambda *_args: {
        "profile": "child", "model": "gpt-5.6-luna", "provider": "openai-codex",
        "chain": [{"model": "gpt-5.6-luna", "provider": "openai-codex"},
                  {"model": "gpt-5.6-luna", "provider": "openai-codex"}],
    })

    json.loads(handler({"goal": "task", "model": "gpt-5.6-luna"}))

    assert _models_attempted(cmds) == ["gpt-5.6-luna"]


# ---------------------------------------------------------------------------
# The router sizes the turn from the prompt the child actually receives
# ---------------------------------------------------------------------------

def test_router_is_given_the_composed_prompt_not_the_goal_alone(monkeypatch):
    """est_input_tokens has to see the context; the child is sent context+goal."""
    seen: list = []
    handler, _pool = _cross_handler(monkeypatch, ("exited", 0, "done", ""))
    cmds = _spawn_recorder(monkeypatch)
    monkeypatch.setattr(
        _dp, "_route_task",
        lambda *args: seen.append(args) or {"profile": "child"},
    )

    handler({"goal": "fix the failing test", "context": "log line\n" * 2000})

    assert seen[0][0] == "fix the failing test", "the goal stays the goal"
    assert seen[0][3] == _dp._compose_prompt("fix the failing test",
                                             ("log line\n" * 2000).strip())
    # The routed text is byte-identical to the prompt the child was handed.
    assert seen[0][3] == cmds[0][cmds[0].index("-q") + 1]


def test_a_big_context_reaches_the_context_rule_end_to_end(monkeypatch, tmp_path):
    """The F6 defect, at the level it was reported: a trivial goal plus a huge
    context routed as if it were six tokens."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(_dp, "_load_router_config", lambda: {
        "enabled": True,
        "fail_safe": {"profile": "child", "model": "small-rail", "provider": "cheap"},
        "rules": [{"id": "huge-context-read",
                   "when": {"est_input_tokens": {"gt": 20000}},
                   "then": {"profile": "child", "model": "T3"}}],
        "default": {"action": "classify"},
        "tiers": {"T1": {"model": "tiny", "provider": "cheap"},
                  "T2": {"model": "small", "provider": "cheap"},
                  "T3": {"model": "gpt-5.6-terra", "provider": "openai-codex"},
                  "T4": {"model": "gpt-5.5", "provider": "openai-codex"}},
    })
    handler, _pool = _cross_handler(monkeypatch, ("exited", 0, "done", ""))
    cmds = _spawn_recorder(monkeypatch)

    handler({"goal": "fix the failing test",
             "context": "WARN retry scheduled for the nightly job\n" * 2000})
    assert _models_attempted(cmds) == ["gpt-5.6-terra"]

    # Same goal, no context: nothing in Table 1 matches and it falls to
    # fail_safe — which is exactly what the big-context turn used to do.
    cmds.clear()
    handler({"goal": "fix the failing test"})
    assert _models_attempted(cmds) == ["small-rail"]


def test_the_live_hook_persists_the_chain_plan_for_replay(monkeypatch, tmp_path):
    """routes.jsonl is the console replay panel's only data source.

    Before this, every live entry omitted chain_plan, so the panel showed its
    empty default forever and no operator could see what the filter dropped.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(_dp, "_load_router_config", lambda: {
        "enabled": True,
        "default": {"action": "classify"},
        "rules": [{"id": "vision-required", "when": {"needs_vision": {"eq": True}},
                   "then": {"profile": "coder", "model": "T2"}}],
        "tiers": {
            "T1": {"model": "glm-4.7", "provider": "zai"},
            "T2": {"model": "glm-5.3", "provider": "zai",
                   "fallback": [{"model": "gpt-5.6-luna", "provider": "openai-codex"},
                                {"model": "deepseek-v4-flash", "provider": "deepseek"}]},
            "T3": {"model": "gpt-5.6-terra", "provider": "openai-codex"},
            "T4": {"model": "gpt-5.5", "provider": "openai-codex"},
        },
    })

    routed = _dp._route_task("look at this screenshot of the ui", "", None)
    assert _dp._routed_targets(routed) == [("gpt-5.6-luna", "openai-codex")]

    import router.durable_decision_log as ddl
    entry = json.loads(
        [l for l in ddl.routes_path().read_text(encoding="utf-8").splitlines() if l.strip()][-1]
    )
    plan = entry["chain_plan"]
    assert [hop["model"] for hop in plan["chain"]] == ["gpt-5.6-luna"]
    assert {hop["reject_reason"] for hop in plan["rejected"]} == {"no_vision"}
    assert isinstance(entry["steps"][1]["in"]["seed"], int), \
        "the ordering seed must be replayable from the persisted trace"


def test_compose_prompt_is_the_single_definition_of_the_child_prompt():
    """Two copies of this composition is how the context signal went blind."""
    assert _dp._compose_prompt("goal", "ctx") == "Context: ctx\n\nTask: goal"
    assert _dp._compose_prompt("goal", "") == "goal"


def test_routed_targets_degrades_and_dedupes():
    """The executor's target derivation, on the shapes it really receives."""
    planned = {"model": "declared", "provider": "a",
               "fallback": [{"model": "hop", "provider": "b"}],
               "chain": [{"model": "planned", "provider": "c"}]}
    assert _dp._routed_targets(planned) == [("planned", "c")]
    assert _dp._routed_targets({"model": "declared", "provider": "a",
                                "fallback": [{"model": "hop", "provider": "b"}]}) == [
        ("declared", "a"), ("hop", "b")]
    assert _dp._routed_targets({"profile": "child"}) == []
    assert _dp._routed_targets({"model": "m", "chain": "not-a-list"}) == [("m", "")]
    assert _dp._routed_targets({"model": "m", "chain": [{"provider": "no-model"}]}) == [
        ("m", "")]
    assert _dp._routed_targets("not a mapping") == []
    assert _dp._dedupe_targets([("a", "x"), ("a", "x"), ("b", "y")]) == [
        ("a", "x"), ("b", "y")]


def test_record_breaker_outcome_dispatches_success_and_failure(monkeypatch):
    calls = []

    class Blocklist:
        def __init__(self, _config):
            pass

        def record_failure(self, *args):
            calls.append(("failure", args))

        def record_success(self, *args):
            calls.append(("success", args))

    import router.blocklist

    monkeypatch.setattr(router.blocklist, "Blocklist", Blocklist)
    monkeypatch.setattr(
        _dp,
        "_load_router_config",
        lambda: {"tiers": {"T1": {"model": "m", "provider": "p"}}},
    )
    _dp._record_breaker_outcome("child", "m", "ttfb_stall")
    _dp._record_breaker_outcome("child", "m", None)
    assert calls == [("failure", ("m", "p", "ttfb_stall")), ("success", ("m", "p"))]

    # The ATTEMPTED provider wins over the policy scan: the hop that ran is the
    # one the breaker must be keyed against, even when policy declares the same
    # model elsewhere under a different provider.
    calls.clear()
    _dp._record_breaker_outcome("child", "m", "quota_exhausted", "attempted")
    assert calls == [("failure", ("m", "attempted", "quota_exhausted"))]

    # And a hop that is ONLY ever a fallback (never a tier primary) is still
    # resolvable when a caller cannot name it — a primaries-only scan returned ""
    # for exactly the elos that only appear in a chain tail.
    calls.clear()
    monkeypatch.setattr(
        _dp,
        "_load_router_config",
        lambda: {"tiers": {"T2": {"model": "m", "provider": "p",
                                 "fallback": [{"model": "hop", "provider": "hp"}]}}},
    )
    _dp._record_breaker_outcome("child", "hop", "quota_exhausted")
    assert calls == [("failure", ("hop", "hp", "quota_exhausted"))]


# ---------------------------------------------------------------------------
# The breaker must bind under the key the routing path actually reads
# ---------------------------------------------------------------------------
#
# Blocklist keys breaker state as ``model@provider`` and routing asks
# ``is_blocked(model, provider)``; the console's liveness panel looks the same key
# up. The executor recorded outcomes with the attempted MODEL only, and the
# provider was re-derived by scanning tier PRIMARIES — so every fallback hop was
# recorded under a bare model key that neither surface reads. A quota-exhausted
# rail therefore kept being attempted while its breaker sat tripped in a cell
# nothing consulted: the recurring defect, with the DISPLAY (a tripped breaker in
# `hermes router blocklist`) disagreeing with the path that RUNS.

# gpt-5.6-luna is reachable on TWO rails: as T2's fallback hop on openai-codex
# (the one the executor attempts) and as T4's primary on openai-direct. No scan of
# the policy can tell which one just failed — only the attempt knows — so this
# shape is what forces the attempted provider to be passed through rather than
# re-derived, and it also pins that one rail's breaker never takes out its twin.
_TWO_RAIL_POLICY = {
    "enabled": True,
    "blocklist": {
        "auto_breaker": {
            "enabled": True,
            "threshold": 5,
            "window_seconds": 600,
            "base_cooldown_seconds": 60,
            "max_cooldown_seconds": 900,
            "backoff_multiplier": 2.0,
        }
    },
    "tiers": {
        "T1": {"model": "glm-4.7", "provider": "zai"},
        "T2": {"model": "glm-5.3", "provider": "zai",
               "fallback": [{"model": "gpt-5.6-luna", "provider": "openai-codex"}]},
        "T3": {"model": "mimo-v2.5", "provider": "xiaomi"},
        "T4": {"model": "gpt-5.6-luna", "provider": "openai-direct"},
    },
}

_QUOTA_OUTPUT = "API call failed after 3 retries: HTTP 429 rate limited\n"


def test_a_failing_fallback_hop_binds_the_breaker_every_surface_reads(
    monkeypatch, tmp_path
):
    """One failed hop, three surfaces, one key — the agreement IS the assertion.

    The executor records the outcome, the routing path asks
    ``is_blocked(model, provider)``, and ``RouterService.liveness`` renders the
    same ``model@provider``. Asserting only that the breaker "tripped" would have
    passed while the trip was filed under a key neither reads; asserting only the
    display would have passed too. So all three are read back for the hop that
    actually ran.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    real_record = _dp._record_breaker_outcome
    handler, _pool = _cross_handler(monkeypatch, ("exited", 0, _QUOTA_OUTPUT, ""))
    # _cross_handler stubs the breaker seam out; this test is about it.
    monkeypatch.setattr(_dp, "_record_breaker_outcome", real_record)
    monkeypatch.setattr(_dp, "_load_router_config",
                        lambda: copy.deepcopy(_TWO_RAIL_POLICY))
    monkeypatch.setattr(_dp, "_route_task", lambda *_args: {
        "profile": "child", **_BLIND_PRIMARY,
        # A vision turn: the blind primary is filtered out, so the ONLY eligible
        # target is the fallback hop — precisely the elo whose breaker never bound.
        "chain": [{"model": "gpt-5.6-luna", "provider": "openai-codex"}],
    })

    result = json.loads(handler({"goal": "look at this screenshot"}))
    assert result["model"] == "gpt-5.6-luna" and result["provider"] == "openai-codex"
    assert result["failure_kind"] == "quota_exhausted"

    from router.blocklist import Blocklist
    blocklist = Blocklist(copy.deepcopy(_TWO_RAIL_POLICY))

    # 1. The RUNNING path: what routing asks before attempting a target.
    assert blocklist.is_blocked("gpt-5.6-luna", "openai-codex") is True, (
        "the breaker must bind for the (model, provider) that actually failed"
    )
    # 2. The recorded key is the one both surfaces read — and it is the rail that
    #    failed, not the other rail serving the same elo, whose quota is intact.
    assert [entry["model_key"] for entry in blocklist.breaker_status()] == [
        "gpt-5.6-luna@openai-codex"
    ]
    assert blocklist.is_blocked("gpt-5.6-luna", "openai-direct") is False, (
        "one rail's breaker must not take out a healthy rail for the same elo"
    )
    # 3. The REPORTING surface: the console's liveness panel resolves the same key.
    policy_path = tmp_path / "router.yaml"
    policy_path.write_text(
        yaml.safe_dump(_TWO_RAIL_POLICY, sort_keys=False), encoding="utf-8"
    )
    from router.service import RouterService
    panel = {
        entry["model_key"]: entry
        for entry in RouterService(policy_path).liveness()["models"]
    }
    hop = panel["gpt-5.6-luna@openai-codex"]
    # The panel must land on the SAME breaker entry the routing path binds on.
    # Which non-alive label it picks (``degraded`` vs ``quota_exhausted``) is the
    # breaker module's own reporting, so the agreement asserted here is the key
    # and the OPEN state — not the wording.
    assert hop["breaker"].get("state") == "OPEN", (
        "the panel and the routing path must describe one breaker, not two"
    )
    assert hop["state"] != "alive"
    for healthy in ("glm-5.3@zai", "gpt-5.6-luna@openai-direct"):
        assert panel[healthy]["state"] == "alive", (
            "only the rail that actually ran may be marked"
        )


# ---------------------------------------------------------------------------
# The policy scan that fills in half of the breaker key
# ---------------------------------------------------------------------------
#
# ``_provider_of_declared_model`` is the LAST RESORT half of that key: it runs
# only when the caller could not name the rail it attempted. router.yaml is the
# operator's live file and is edited by hand, so the scan is written to walk PAST
# a half-edited row rather than raise or stop. Every guard below is asserted
# through the same pair the breaker itself turns on — the key
# ``_record_breaker_outcome`` WRITES and the key the routing path READS — because
# a scan that resolves the wrong rail files the outcome in a cell nothing
# consults, which is indistinguishable from not recording it at all.

# One tier table with every malformed shape the scan has to survive, and the real
# declaration for ``fallback-only-hop`` behind all of them. T3's fallback list is
# the interesting one: it holds a non-mapping entry, a hop with no model and a hop
# for a DIFFERENT model, so the scan has to run that list to its end and then
# carry on to the next tier instead of concluding "not declared".
_HALF_EDITED_POLICY = {
    "enabled": True,
    "blocklist": {"auto_breaker": {"enabled": True, "threshold": 5,
                                   "window_seconds": 600,
                                   "base_cooldown_seconds": 60,
                                   "max_cooldown_seconds": 900,
                                   "backoff_multiplier": 2.0}},
    "tiers": {
        # A tier that is not a mapping at all (a stray list item under `tiers:`).
        "T1": ["glm-4.7"],
        # A tier mid-edit: no model yet, and `fallback:` still a scalar.
        "T2": {"provider": "zai", "fallback": "gpt-5.6-luna"},
        # A well-formed tier that simply does not declare the elo being scanned.
        "T3": {"model": "glm-5.3", "provider": "zai", "fallback": [
            "deepseek-v4-flash",
            {"provider": "deepseek"},
            {"model": "deepseek-v4-flash", "provider": "deepseek"},
        ]},
        # ...and the row that does declare it, reachable only as a fallback hop.
        "T4": {"model": "gpt-5.5", "provider": "openai-codex", "fallback": [
            {"model": "fallback-only-hop", "provider": "xiaomi"},
        ]},
    },
}


def test_a_half_edited_tier_table_still_resolves_a_fallback_only_rail(
    monkeypatch, tmp_path
):
    """A malformed row must be walked past, not treated as the end of the scan.

    The elo that ran is declared on ONE rail, in the last tier, as a fallback
    hop — behind a non-mapping tier, a tier whose `fallback:` is a scalar, and a
    fallback list that names other models. A scan that stopped or raised at any of
    those would key this outcome under a bare model, and both assertions below
    read the pair that would then be empty.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(_dp, "_load_router_config",
                        lambda: copy.deepcopy(_HALF_EDITED_POLICY))

    _dp._record_breaker_outcome("child", "fallback-only-hop", "quota_exhausted")

    from router.blocklist import Blocklist
    blocklist = Blocklist(copy.deepcopy(_HALF_EDITED_POLICY))
    # The RUNNING path and the DISPLAY, for the one outcome just recorded.
    assert blocklist.is_blocked("fallback-only-hop", "xiaomi") is True, (
        "the scan lost the rail, so the breaker bound on a key routing never asks"
    )
    assert [entry["model_key"] for entry in blocklist.breaker_status()] == [
        "fallback-only-hop@xiaomi"
    ]
    # Non-vacuity: the bare-model key is the one a stopped scan would have
    # written, and it sees nothing — so the assertion above is about the rail.
    assert blocklist.is_blocked("fallback-only-hop", "") is False


def test_a_policy_with_no_tier_table_records_coarsely_rather_than_not_at_all(
    monkeypatch, tmp_path
):
    """``tiers:`` written as a LIST leaves no rail to derive, and that is not fatal.

    The docstring's stated last resort: "" (a breaker keyed by bare model) rather
    than a dropped outcome, since a breaker that records nothing is strictly worse
    than one keyed coarsely. Asserted as the same write/read pair — the coarse key
    is the one both sides then use, so the record is still legible.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    policy = copy.deepcopy(_HALF_EDITED_POLICY)
    policy["tiers"] = ["T1", "T2", "T3", "T4"]
    monkeypatch.setattr(_dp, "_load_router_config", lambda: copy.deepcopy(policy))

    assert _dp._provider_of_declared_model("glm-5.3", policy) == ""
    _dp._record_breaker_outcome("child", "glm-5.3", "quota_exhausted")

    from router.blocklist import Blocklist
    blocklist = Blocklist(copy.deepcopy(policy))
    assert blocklist.is_blocked("glm-5.3", "") is True, (
        "the outcome was dropped, not recorded coarsely"
    )
    assert [entry["model_key"] for entry in blocklist.breaker_status()] == ["glm-5.3"]


def test_a_router_yaml_whose_top_level_is_a_list_derives_no_rail(tmp_path):
    """The whole config, not just ``tiers``, can arrive as a non-mapping.

    ``_load_router_config`` ends in ``yaml.safe_load(...) or {}``, and a list is
    truthy — so a policy file mis-indented one level to the left reaches the scan
    as a list and never becomes ``{}``. Provoked with such a file rather than by
    handing the function a literal, because the ``or {}`` is exactly the step that
    looks like it already ruled this out.

    Only the scan is asserted: ``Blocklist(config)`` cannot be built from a list
    at all, so this shape has no breaker to record into either way.
    """
    corrupt = tmp_path / "router.yaml"
    corrupt.write_text("- model: glm-5.3\n  provider: zai\n", encoding="utf-8")
    config = yaml.safe_load(corrupt.read_text(encoding="utf-8")) or {}
    assert isinstance(config, list), "the premise: a list survives the `or {}`"

    assert _dp._provider_of_declared_model("glm-5.3", config) == ""


def test_a_caller_named_model_never_inherits_the_routers_rail(monkeypatch):
    """(model, provider) is ONE decision, and this seam split it.

    `routed_provider` was assigned unconditionally from the routed decision, while
    the PAIR was only completed `if not model` — so when the caller named a model,
    the router's rail survived and was spawned beside it. Reproduced argv:

        -m operator-choice --provider zai

    which is the exact shape `adapter.py` documents in three comment blocks after
    measuring `gpt-5.6-terra @ zai`: a pair that names no real rail, whose spawn
    fails with an opaque provider error, and — because `nonzero_exit` is not
    retryable — takes the cross-rail fallback down with it. It also records a
    breaker strike under a `model@provider` key naming a rail that does not exist,
    which then blocks the healthy pair nobody ran.
    """
    handler, _pool = _cross_handler(monkeypatch, ("exited", 0, "done", ""))
    cmds = _spawn_recorder(monkeypatch)
    # The router chose a T1-shaped decision on zai; the caller wants another model.
    monkeypatch.setattr(_dp, "_route_task", lambda *_args: {
        "profile": "child", "model": "glm-5.3-flash", "provider": "zai",
        "fallback": [{"model": "gpt-5.6-luna", "provider": "openai-codex"}],
    })

    handler({"prompt": "do the thing", "model": "operator-choice"})

    assert _models_attempted(cmds)[0] == "operator-choice", (
        "the caller's model must still be honoured"
    )
    assert "zai" not in _providers_attempted(cmds), (
        f"the caller's model inherited the router's rail: {cmds[0]}"
    )


def test_a_caller_named_model_that_is_a_plan_hop_takes_that_hops_rail(monkeypatch):
    """The rail comes from the hop that names the SAME model, or from nowhere.

    Not `(model, "")` unconditionally: when the caller names an elo the router also
    planned, that hop already pairs it with the rail it would be dispatched on, and
    dropping it would both lose a real answer and break the dedupe that keeps a
    target from being attempted twice.
    """
    handler, _pool = _cross_handler(monkeypatch, ("exited", 0, "done", ""))
    cmds = _spawn_recorder(monkeypatch)
    monkeypatch.setattr(_dp, "_route_task", lambda *_args: {
        "profile": "child", "model": "glm-5.3-flash", "provider": "zai",
        "fallback": [{"model": "gpt-5.6-luna", "provider": "openai-codex"}],
    })

    handler({"prompt": "do the thing", "model": "gpt-5.6-luna"})

    assert _models_attempted(cmds)[0] == "gpt-5.6-luna"
    assert _providers_attempted(cmds)[0] == "openai-codex", (
        "the rail must be the one that hop declared, not the router's primary rail"
    )


# ---------------------------------------------------------------------------
# A routing REFUSAL is not a caller mistake
# ---------------------------------------------------------------------------

class TestARoutingDeclineNamesItsOwnCause:
    """Seven declines used to arrive as one `bad_args`, with harmful advice.

    `_route_task` returns None for the recursion guard, a disabled router, a
    malformed policy, a blocklist/breaker veto, a pending `action: classify`, an
    unresolved profile and any exception. The handler collapsed all seven into
    `{"error": "profile is required", "failure_kind": "bad_args"}` with no
    `retryable`.

    `bad_args` describes a CALLER mistake, which a veto is not — and the
    remediation told the caller to name a profile, which skips routing and
    therefore skips the blocklist entirely. The advice for "fixing" a refusal was
    the one action that circumvents it. Measured on real traffic: 22 of 252
    decisions (~9%) were blocklist vetoes with no profile.
    """

    @staticmethod
    def _handler(monkeypatch, config):
        monkeypatch.setattr(_dp, "_load_router_config", lambda: config)
        monkeypatch.setattr(_dp, "_profile_exists", lambda _p: True)
        return _dp._make_handler("parent", lambda _args: "inline")

    @staticmethod
    def _shipped():
        return yaml.safe_load(
            (REPO_ROOT / "router.example.yaml").read_text(encoding="utf-8")
        )

    def test_a_blocklist_veto_is_not_reported_as_bad_args(
        self, monkeypatch, tmp_path,
    ):
        """The real path, not a stubbed `_route_task`: ban everything and route."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("HERMES_ROUTE_TRACE_FILE", str(tmp_path / "r.jsonl"))
        config = self._shipped()
        # A caller-named model that is banned: Stage 0 vetoes before anything else.
        config["blocklist"]["manual_ban"] = [
            {"model": "glm-5.3-flash", "provider": "", "reason": "test"}
        ]
        handler = self._handler(monkeypatch, config)

        answer = json.loads(handler({"prompt": "rename a variable",
                                     "model": "glm-5.3-flash"}))

        assert answer["failure_kind"] == "routing_denied", answer
        assert answer["failure_kind"] != "bad_args"
        assert answer["success"] is False
        assert answer["retryable"] is False
        assert answer["blocked_model"] == "glm-5.3-flash"
        assert answer["cause"] == "blocklist_veto"
        # The router's own sentence, verbatim.
        assert "blocked" in answer["error"]
        # And the advice must NOT be "name a profile", which would skip the veto.
        assert "skips this refusal" in answer["hint"]

    def test_a_disabled_router_still_says_name_a_profile_because_that_is_true(
        self, monkeypatch,
    ):
        """The one decline where the generic message was right — it just never said why."""
        handler = self._handler(monkeypatch, {"enabled": False})
        answer = json.loads(handler({"prompt": "rename a variable"}))
        assert answer["failure_kind"] == "bad_args"
        assert "disabled" in answer["error"]
        assert answer["retryable"] is False

    def test_the_recursion_guard_names_itself(self, monkeypatch):
        handler = self._handler(monkeypatch, self._shipped())
        monkeypatch.setattr(_dp._router_guard, "active", True, raising=False)
        answer = json.loads(handler({"prompt": "rename a variable"}))
        assert answer["failure_kind"] == "routing_recursion"
        assert "re-enter" in answer["error"]

    def test_a_router_exception_is_retryable_and_says_so(self, monkeypatch):
        handler = self._handler(monkeypatch, self._shipped())

        def _boom(*_args, **_kwargs):
            raise RuntimeError("registry on fire")

        # Break the router from the inside, past the seam the fakes replace.
        import router.adapter as adapter_mod
        monkeypatch.setattr(adapter_mod, "route", _boom)
        answer = json.loads(handler({"prompt": "rename a variable"}))
        assert answer["failure_kind"] == "routing_error"
        assert answer["retryable"] is True
        assert "registry on fire" in answer["error"]

    def test_a_replaced_seam_still_gets_the_generic_answer(self, monkeypatch):
        """Four test files patch `_route_task` as `lambda *_args: None`.

        The decline rides on the thread-local precisely so that shape keeps
        working — a stand-in sets nothing, and the handler falls back.
        """
        monkeypatch.setattr(_dp, "_route_task", lambda *_args: None)
        handler = _dp._make_handler("parent", lambda _args: "inline")
        assert json.loads(handler({"goal": "task"})) == {
            "error": "profile is required", "failure_kind": "bad_args",
        }


class TestAMalformedPolicyRootDoesNotBreakLoading:
    """`or {}` catches an empty file but not a root that is a list or a scalar.

    `[a, b]` and `off` are both legal YAML, and every caller then calls `.get` on
    the result. Both sibling loaders in the router package already guarded for
    this; this one did not, and the consequence was not a degraded route — it was
    the plugin failing to LOAD, because `_make_classify_fn` is called from
    `register()` and the AttributeError propagated out of registration BEFORE
    `_REGISTERED_CTX[ctx_id]` was set, so a retry re-raised identically.
    """

    @pytest.mark.parametrize("root", [
        ["not", "a", "mapping"], "off", 7, 1.5, True,
    ])
    def test_a_non_mapping_root_yields_an_empty_policy(self, root, tmp_path):
        assert _dp._as_policy_mapping(root, tmp_path / "router.yaml") == {}

    def test_an_empty_or_mapping_root_is_passed_through(self, tmp_path):
        assert _dp._as_policy_mapping(None, tmp_path / "r.yaml") == {}
        policy = {"enabled": True}
        assert _dp._as_policy_mapping(policy, tmp_path / "r.yaml") is policy

    def test_register_survives_a_list_rooted_policy(self, monkeypatch, tmp_path):
        """The worst of the three: the plugin did not load at all."""
        monkeypatch.setattr(
            _dp, "_load_router_config",
            lambda: _dp._as_policy_mapping(["nope"], tmp_path / "router.yaml"),
        )
        monkeypatch.setattr(_dp, "_get_active_profile_name", lambda: "parent")

        class Ctx:
            def __init__(self):
                self.tools = []
                self.hooks = []

            def register_tool(self, **kwargs):
                self.tools.append(kwargs)

            def register_hook(self, *args, **kwargs):
                self.hooks.append(args)

        ctx = Ctx()
        _dp.register(ctx)          # used to raise AttributeError
        assert ctx.tools and ctx.hooks, "the tool and hooks must still register"

    def test_the_kanban_hook_keeps_its_returns_none_contract(
        self, monkeypatch, tmp_path,
    ):
        monkeypatch.setattr(
            _dp, "_load_router_config",
            lambda: _dp._as_policy_mapping("off", tmp_path / "router.yaml"),
        )
        assert _dp._on_pre_kanban_dispatch(
            task_id="t", profile_name="p", board="b", assignee="coder", run_id=1,
        ) is None

    def test_the_warning_is_emitted_once_per_message(self, caplog, tmp_path):
        """A per-decision reload must not flood the log."""
        _dp._WARNED.clear()
        caplog.set_level("WARNING")
        for _ in range(5):
            _dp._as_policy_mapping(["nope"], tmp_path / "router.yaml")
        naming = [r for r in caplog.records if "top level" in r.getMessage()]
        assert len(naming) == 1, [r.getMessage() for r in naming]
        assert "routing is DISABLED" in naming[0].getMessage()


class TestTheAntiRecursionGuaranteeIsEnforced:
    """`HERMES_DELEGATE_PROFILE_DISABLE=1` was written and read by nothing.

    `_attempt` puts it in every child env, and the README (twice), PRODUCT.md and
    the architecture map all assert the guarantee as fact — while nothing in the
    repo read it back. It was a comment, not a control.

    It is not cosmetic: a nested `_spawn` creates its OWN session and pgid, so a
    depth-2 tree escapes both the outer `killpg` and the atexit registry — the
    orphaned-grandchildren failure mode this whole plugin exists to prevent.
    """

    class _Ctx:
        def __init__(self):
            self.tools = []
            self.hooks = []

        def register_tool(self, **kwargs):
            self.tools.append(kwargs["name"])

        def register_hook(self, name, _fn):
            self.hooks.append(name)

    def test_a_delegated_child_is_not_offered_the_tool(self, monkeypatch):
        monkeypatch.setenv("HERMES_DELEGATE_PROFILE_DISABLE", "1")
        monkeypatch.setattr(_dp, "_get_active_profile_name", lambda: "child")
        monkeypatch.setattr(_dp, "_REGISTERED_CTX", {})
        ctx = self._Ctx()
        _dp.register(ctx)
        assert ctx.tools == [], "a delegated child must not be able to spawn another"

    def test_the_hooks_stay_registered_in_that_child(self, monkeypatch):
        """Deliberate, and the reason matters.

        The child is still a Hermes agent that may dispatch kanban cards. Gating
        `register_hook` too would silently turn off shadow/live kanban routing in
        every delegated process — losing the measurement for exactly the traffic a
        subagent generates.
        """
        monkeypatch.setenv("HERMES_DELEGATE_PROFILE_DISABLE", "1")
        monkeypatch.setattr(_dp, "_get_active_profile_name", lambda: "child")
        monkeypatch.setattr(_dp, "_REGISTERED_CTX", {})
        ctx = self._Ctx()
        _dp.register(ctx)
        assert ctx.hooks == [
            "post_tool_call", "pre_api_request", "post_api_request", "pre_tool_call", "pre_llm_call",
            "pre_kanban_dispatch",
            "kanban_task_completed",
            "kanban_task_blocked", "on_kanban_worker_exited", "on_kanban_dispatch_tick",
        ]

    def test_an_ordinary_process_still_gets_the_tool(self, monkeypatch):
        monkeypatch.delenv("HERMES_DELEGATE_PROFILE_DISABLE", raising=False)
        monkeypatch.setattr(_dp, "_get_active_profile_name", lambda: "parent")
        monkeypatch.setattr(_dp, "_REGISTERED_CTX", {})
        ctx = self._Ctx()
        _dp.register(ctx)
        assert ctx.tools == ["delegate_profile"]
        assert ctx.hooks == [
            "post_tool_call", "pre_api_request", "post_api_request", "pre_tool_call", "pre_llm_call",
            "pre_kanban_dispatch",
            "kanban_task_completed",
            "kanban_task_blocked", "on_kanban_worker_exited", "on_kanban_dispatch_tick",
        ]

    @pytest.mark.parametrize("value", ["0", "", "true", "yes", "2"])
    def test_only_the_exact_documented_value_withholds_it(self, value, monkeypatch):
        """`== "1"` exactly, which is what `_attempt` writes and the README states.

        A truthy-string reading would make `HERMES_DELEGATE_PROFILE_DISABLE=0`
        disable the tool — a setting that looks like "off" turning it off.
        """
        monkeypatch.setenv("HERMES_DELEGATE_PROFILE_DISABLE", value)
        monkeypatch.setattr(_dp, "_get_active_profile_name", lambda: "parent")
        monkeypatch.setattr(_dp, "_REGISTERED_CTX", {})
        ctx = self._Ctx()
        _dp.register(ctx)
        assert ctx.tools == ["delegate_profile"], value

    def test_the_value_the_child_receives_is_the_value_that_withholds(
        self, monkeypatch,
    ):
        """The two halves must agree, and this asserts them against each other.

        `_attempt` writes the variable; `register` reads it. A test that pinned
        either literal alone would let the pair drift — which is how this stayed
        vacuous for as long as it did.
        """
        written = {}

        def fake_spawn(cmd, env):
            written.update(env)
            raise FileNotFoundError("stop here; the env is what is under test")

        monkeypatch.setattr(_dp, "_spawn", fake_spawn)
        monkeypatch.setattr(_dp, "_profile_exists", lambda _p: True)
        monkeypatch.setattr(_dp, "_resolve_hermes_bin", lambda: "hermes")
        monkeypatch.setattr(_dp, "_get_pool", lambda: FakePool())
        handler = _dp._make_handler("parent", lambda _a: "inline")
        handler({"prompt": "task", "profile": "child"})

        assert "HERMES_DELEGATE_PROFILE_DISABLE" in written
        monkeypatch.setenv(
            "HERMES_DELEGATE_PROFILE_DISABLE",
            written["HERMES_DELEGATE_PROFILE_DISABLE"],
        )
        monkeypatch.setattr(_dp, "_get_active_profile_name", lambda: "child")
        monkeypatch.setattr(_dp, "_REGISTERED_CTX", {})
        ctx = self._Ctx()
        _dp.register(ctx)
        assert ctx.tools == [], (
            "the value _attempt writes must be the value register withholds on"
        )


# ── `smart-router` as a selectable model: the plugin-side adapter ─────────────

def test_register_offers_the_pseudo_model_through_llm_request_middleware(monkeypatch):
    """MIDDLEWARE, not a hook — because no hook can change a chat turn's model.

    `VALID_HOOKS` is a closed set with nothing for it, and `pre_llm_call` fires every
    turn but its return value is read for `{"context": str}` only, so a returned
    `{"model": ...}` is discarded. `llm_request` middleware replaces the outgoing
    provider kwargs wholesale, and needs no change to hermes-agent.
    """
    class Ctx:
        def __init__(self):
            self.middleware = []

        def register_tool(self, **kwargs):
            pass

        def register_hook(self, *args, **kwargs):
            pass

        def register_middleware(self, kind, callback):
            self.middleware.append((kind, callback))

    monkeypatch.setattr(_dp, "_get_active_profile_name", lambda: "parent")
    ctx = Ctx()
    _dp.register(ctx)

    kinds = [kind for kind, _cb in ctx.middleware]
    assert kinds == ["llm_request"], kinds
    assert ctx.middleware[0][1] is _dp._on_llm_request


def test_a_host_without_register_middleware_still_registers_everything_else(monkeypatch):
    """The plugin is deployed BY COPY onto hosts whose hermes-agent version varies.

    `register_middleware` is newer than `register_hook`. Raising on a host that lacks it
    would take the tool, both hooks, the console and the sidecar down for a feature
    nobody on that host selected.
    """
    class OldCtx:
        def __init__(self):
            self.tools = []
            self.hooks = []

        def register_tool(self, **kwargs):
            self.tools.append(kwargs)

        def register_hook(self, *args, **kwargs):
            self.hooks.append(args)

    monkeypatch.setattr(_dp, "_get_active_profile_name", lambda: "parent")
    ctx = OldCtx()
    _dp.register(ctx)  # must not raise

    assert ctx.tools and ctx.tools[0]["name"] == "delegate_profile"
    assert [a[0] for a in ctx.hooks] == [
        "post_tool_call", "pre_api_request", "post_api_request", "pre_tool_call", "pre_llm_call",
            "pre_kanban_dispatch",
            "kanban_task_completed",
        "kanban_task_blocked", "on_kanban_worker_exited", "on_kanban_dispatch_tick",
    ]


def test_the_middleware_leaves_a_normal_model_untouched(monkeypatch):
    """The common case pays one dict lookup and returns None."""
    calls = []
    monkeypatch.setattr(_dp, "_route_task", lambda *a, **k: calls.append(a) or None)
    result = _dp._on_llm_request(
        request={"model": "us.anthropic.claude-opus-5", "messages": []},
        provider="bedrock", turn_id="t1",
    )
    assert result is None
    assert calls == [], "and never consults the router"


def test_the_middleware_rewrites_the_sentinel_to_the_routers_choice(monkeypatch):
    monkeypatch.setattr(
        _dp, "_load_router_config",
        lambda: {"fail_safe": {"model": "last-resort", "provider": "bedrock"}},
    )
    monkeypatch.setattr(_dp, "_make_classify_fn", lambda _ctx: None)
    monkeypatch.setattr(
        _dp, "_route_task",
        lambda goal, requested, classify, prompt_text="", source="": {
            "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0", "provider": "bedrock"},
    )
    _dp._PSEUDO_DECISIONS.clear()

    result = _dp._on_llm_request(
        request={"model": "smart-router",
                 "messages": [{"role": "user", "content": "rename a variable"}]},
        provider="bedrock", turn_id="turn-a",
    )
    assert result is not None
    assert result["request"]["model"] == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert result["request"]["messages"][0]["content"] == "rename a variable"


def test_the_middleware_never_raises_no_matter_what_the_router_does(monkeypatch):
    """A middleware that throws is swallowed by the caller and the ORIGINAL kwargs go
    out — which for a sentinel model means a fake id on the wire. So this guarding is
    not politeness; it is what stops a routing bug becoming a provider error."""
    def boom(*_a, **_k):
        raise RuntimeError("router exploded")

    monkeypatch.setattr(_dp, "_load_router_config", boom)
    assert _dp._on_llm_request(request={"model": "smart-router", "messages": []},
                               provider="bedrock", turn_id="t") is None


def test_the_decision_cache_is_bounded(monkeypatch):
    """A gateway process is long-lived and this dict grows once per prompt forever."""
    monkeypatch.setattr(_dp, "_load_router_config", lambda: {})
    monkeypatch.setattr(_dp, "_make_classify_fn", lambda _ctx: None)
    monkeypatch.setattr(
        _dp, "_route_task",
        lambda *a, **k: {"model": "m", "provider": "bedrock"},
    )
    _dp._PSEUDO_DECISIONS.clear()
    for i in range(_dp._PSEUDO_DECISIONS_MAX + 40):
        _dp._on_llm_request(
            request={"model": "smart-router",
                     "messages": [{"role": "user", "content": "hi"}]},
            provider="bedrock", turn_id=f"turn-{i}",
        )
    assert len(_dp._PSEUDO_DECISIONS) <= _dp._PSEUDO_DECISIONS_MAX
    # Oldest evicted, newest kept — a turn's calls are consecutive, so only recent
    # entries are ever read.
    assert "turn-0" not in _dp._PSEUDO_DECISIONS
    assert f"turn-{_dp._PSEUDO_DECISIONS_MAX + 39}" in _dp._PSEUDO_DECISIONS


def test_a_selected_pseudo_model_that_cannot_be_applied_says_so(monkeypatch):
    """The one case where the sentinel goes out unchanged, and it must be LOUD.

    With no usable last resort, this refuses to rewrite — inventing a model would be the
    plugin choosing routing policy. The request then carries `smart-router` to a provider
    that will answer model-not-found, so the log line is the only thing standing between
    that and a mystery.
    """
    monkeypatch.setattr(_dp, "_load_router_config", lambda: {})  # no fail_safe at all
    monkeypatch.setattr(_dp, "_make_classify_fn", lambda _ctx: None)
    monkeypatch.setattr(_dp, "_route_task", lambda *a, **k: None)  # declines
    _dp._PSEUDO_DECISIONS.clear()
    _dp._WARNED_ONCE.clear() if hasattr(_dp, "_WARNED_ONCE") else None
    warned = []
    monkeypatch.setattr(_dp, "_warn_once", lambda msg: warned.append(msg))

    result = _dp._on_llm_request(
        request={"model": "smart-router",
                 "messages": [{"role": "user", "content": "do a thing"}]},
        provider="bedrock", turn_id="t-unapplied",
    )
    assert result is None, "nothing rewritten"
    assert warned, "and the operator is told, because the fake id is now on the wire"
    assert "smart-router" in warned[0] and "not applied" in warned[0]


def test_each_trace_writer_stamps_its_source(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_ROUTE_TRACE_FILE", str(tmp_path / "routes.jsonl"))
    kanban = _dp._KanbanShadowLog(task_id="t1", run_id=1)
    kanban.record("classifier", {"model": "m"})
    assert kanban.entries()[0]["source"] == "kanban"
    import inspect
    params = inspect.signature(_dp._route_task).parameters
    assert params["source"].default == "delegate"
    assert 'source="chat"' in inspect.getsource(_dp._on_llm_request)
