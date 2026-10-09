"""F7: the plugin-level hook callbacks that front router.alignment_runtime."""

import importlib.util
import sys
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
