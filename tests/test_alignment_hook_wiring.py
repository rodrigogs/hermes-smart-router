"""F7: the plugin-level hook callbacks that front router.alignment_runtime."""

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location("alignment_wiring_plugin", REPO_ROOT / "__init__.py")
dp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dp)

from router import alignment_runtime as rt  # noqa: E402


def _cfg(monkeypatch, alignment):
    monkeypatch.setattr(dp, "_load_router_config", lambda: {"alignment": alignment})


def test_post_hook_forwards_only_when_enabled(monkeypatch):
    seen = []
    monkeypatch.setattr(rt, "observe_post_api_request", lambda ctx, cfg, **kw: seen.append(kw))
    hook = dp._make_post_api_request(object())
    _cfg(monkeypatch, {"enabled": True})
    hook(session_id="s", api_call_count=3)
    _cfg(monkeypatch, {"enabled": False})
    hook(session_id="s", api_call_count=4)
    _cfg(monkeypatch, "junk")
    hook(session_id="s", api_call_count=5)
    assert seen == [{"session_id": "s", "api_call_count": 3}]


def test_pre_hook_stashes_history_only_when_enabled(monkeypatch):
    rt._HISTORY.clear()
    _cfg(monkeypatch, {"enabled": False})
    dp._on_pre_api_request(session_id="s", conversation_history=[1])
    assert rt._HISTORY == {}
    _cfg(monkeypatch, {"enabled": True})
    dp._on_pre_api_request(session_id="s", conversation_history=[1])
    assert rt._HISTORY == {"s": [1]}
    rt._HISTORY.clear()


def test_hooks_never_raise(monkeypatch):
    def boom():
        raise RuntimeError("x")

    monkeypatch.setattr(dp, "_load_router_config", boom)
    dp._on_pre_api_request(session_id="s")
    dp._make_post_api_request(None)(session_id="s")


def test_liveness_tick_direct_failure_is_swallowed(monkeypatch):
    from router import gateway_liveness

    monkeypatch.setattr(gateway_liveness, "on_dispatch_tick", lambda **kw: 1 / 0)
    dp._on_kanban_dispatch_tick(board="b")


def test_alignment_actions_direct_success_and_failure(monkeypatch):
    from router import alignment_actions

    _cfg(monkeypatch, {"enabled": True})
    monkeypatch.setattr(alignment_actions, "pre_tool_call", lambda cfg, **kw: {"tool": "ok"})
    monkeypatch.setattr(alignment_actions, "pre_llm_call", lambda cfg, **kw: {"llm": "ok"})
    assert dp._on_alignment_pre_tool_call(session_id="s") == {"tool": "ok"}
    assert dp._on_alignment_pre_llm_call(session_id="s") == {"llm": "ok"}
    _cfg(monkeypatch, {"enabled": False})
    assert dp._on_alignment_pre_tool_call(session_id="s") is None
    assert dp._on_alignment_pre_llm_call(session_id="s") is None
    _cfg(monkeypatch, {"enabled": True})
    monkeypatch.setattr(dp, "_alignment_actions", lambda: 1 / 0)
    assert dp._on_alignment_pre_tool_call(session_id="s") is None
    assert dp._on_alignment_pre_llm_call(session_id="s") is None


def _package_plugin(monkeypatch):
    parent = "hermes_plugins"
    name = f"{parent}.alignment_wiring_package_plugin"
    package = types.ModuleType(parent)
    package.__path__ = []
    monkeypatch.setitem(sys.modules, parent, package)
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "__init__.py", submodule_search_locations=[str(REPO_ROOT)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_package_relative_hook_imports(monkeypatch):
    prefix = "hermes_plugins"
    original_modules = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == prefix or name.startswith(f"{prefix}.")
    }
    try:
        pkg = _package_plugin(monkeypatch)
        _cfg(monkeypatch, {"enabled": True})
        monkeypatch.setattr(pkg, "_load_router_config", lambda: {"alignment": {"enabled": True}})
        seen = []
        liveness = importlib.import_module(f"{pkg.__name__}.router.gateway_liveness")
        actions = importlib.import_module(f"{pkg.__name__}.router.alignment_actions")
        runtime = importlib.import_module(f"{pkg.__name__}.router.alignment_runtime")
        monkeypatch.setattr(liveness, "on_dispatch_tick", lambda **kw: seen.append(("tick", kw)))
        monkeypatch.setattr(actions, "pre_tool_call", lambda cfg, **kw: {"tool": "package"})
        monkeypatch.setattr(actions, "pre_llm_call", lambda cfg, **kw: {"llm": "package"})
        monkeypatch.setattr(runtime, "stash_history", lambda session_id, history: seen.append((session_id, history)))
        pkg._on_kanban_dispatch_tick(board="b")
        pkg._on_pre_api_request(session_id="s", conversation_history=[1])
        assert pkg._on_alignment_pre_tool_call(session_id="s") == {"tool": "package"}
        assert pkg._on_alignment_pre_llm_call(session_id="s") == {"llm": "package"}
        assert seen == [("tick", {"board": "b"}), ("s", [1])]
    finally:
        for name in tuple(sys.modules):
            if name == prefix or name.startswith(f"{prefix}."):
                del sys.modules[name]
        sys.modules.update(original_modules)
