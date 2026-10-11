"""Registry context/vision must equal the Hermes runtime, or be a flagged override."""
import sys
import types
from pathlib import Path

import pytest

from router import runtime_caps
from router.capabilities import MODEL_CAPABILITIES, registry_diagnostics
from router.runtime_caps import (
    DERIVED_FIELDS,
    load_runtime_lookup,
    reconcile,
    runtime_capabilities,
)

RUNTIME = "/home/rodrigo/hermes-runtime"


@pytest.fixture(scope="module")
def real_lookup():
    if not __import__("os").path.isdir(RUNTIME):
        pytest.skip("Hermes runtime not present")

    # The deploy gate runs from the hermes-agent development venv, which can
    # have already imported a different ``agent`` package.  This integration
    # check must inspect the served runtime at RUNTIME instead.
    previous_agent_modules = {
        name: module
        for name, module in tuple(sys.modules.items())
        if name == "agent" or name.startswith("agent.")
    }
    for name in previous_agent_modules:
        del sys.modules[name]
    sys.path.insert(0, RUNTIME)
    try:
        lookup = load_runtime_lookup()
        runtime_module = sys.modules.get("agent.models_dev")
        assert runtime_module is not None
        module_file = getattr(runtime_module, "__file__", None)
        assert isinstance(module_file, str)
        assert Path(module_file).resolve().is_relative_to(Path(RUNTIME).resolve())
        runtime_snapshot = (
            {
                (entry["provider"], model): lookup(entry["provider"], model)
                for model, entry in MODEL_CAPABILITIES.items()
            }
            if lookup is not None
            else None
        )
    finally:
        sys.path.remove(RUNTIME)
        for name in tuple(sys.modules):
            if name == "agent" or name.startswith("agent."):
                del sys.modules[name]
        sys.modules.update(previous_agent_modules)

    if runtime_snapshot is None:
        pytest.skip("agent.models_dev not importable")
    return lambda provider, model: runtime_snapshot.get((provider, model))


def test_every_model_equals_runtime_or_is_flagged_override(real_lookup):
    report = reconcile(real_lookup)
    assert report, "runtime knew no registered model"
    drift = {
        (m, f): MODEL_CAPABILITIES[m][f]
        for m, fields in report.items()
        for f, status in fields.items()
        if status == "drift"
    }
    assert drift == {}


def test_overrides_are_real_divergences(real_lookup):
    """A runtime_override flag on a field that now matches is stale."""
    for model, fields in reconcile(real_lookup).items():
        flagged = set(MODEL_CAPABILITIES[model].get("runtime_override", ()))
        stale = {f for f in flagged if fields[f] == "match"}
        assert stale == set(), model


def test_runtime_override_values_are_derived_fields_and_registry_is_clean():
    for model, entry in MODEL_CAPABILITIES.items():
        assert set(entry.get("runtime_override", ())) <= set(DERIVED_FIELDS), model
    assert registry_diagnostics() == []


def _fake_module(monkeypatch, getter):
    mod = types.ModuleType("fake_models_dev")
    mod.get_model_capabilities = getter
    monkeypatch.setitem(sys.modules, "fake_models_dev", mod)
    return "fake_models_dev"


def test_load_lookup_missing_runtime_is_none():
    assert load_runtime_lookup("no_such_runtime_module_xyz") is None


def test_load_lookup_module_without_getter_is_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "bare_mod", types.ModuleType("bare_mod"))
    assert load_runtime_lookup("bare_mod") is None


def test_load_lookup_adapts_runtime_object(monkeypatch):
    caps = types.SimpleNamespace(context_window=123, supports_vision=True)
    name = _fake_module(monkeypatch, lambda p, m: caps if m == "glm-5.3" else None)
    lookup = load_runtime_lookup(name)
    assert lookup("zai", "glm-5.3") == {"context_window": 123, "vision": True}
    assert lookup("zai", "other") is None


def test_runtime_capabilities_paths(monkeypatch):
    fake = lambda p, m: {"context_window": 5, "vision": False}  # noqa: E731
    assert runtime_capabilities("glm-5.3", fake) == {"context_window": 5, "vision": False}
    assert runtime_capabilities("not-registered", fake) is None
    monkeypatch.setattr(runtime_caps, "load_runtime_lookup", lambda: None)
    assert runtime_capabilities("glm-5.3") is None
    assert reconcile() == {}


def test_reconcile_classifies_match_override_drift(monkeypatch):
    monkeypatch.setattr(
        runtime_caps,
        "MODEL_CAPABILITIES",
        {
            "a": {"provider": "p", "context_window": 1, "vision": True},
            "b": {"provider": "p", "context_window": 1, "vision": True,
                  "runtime_override": ["context_window"]},
            "c": {"provider": "p", "context_window": 1, "vision": True},
            "d": {"provider": "p", "context_window": 1, "vision": True},
        },
    )
    runtime = {
        "a": {"context_window": 1, "vision": True},
        "b": {"context_window": 2, "vision": True},
        "c": {"context_window": 1, "vision": False},
    }
    report = reconcile(lambda p, m: runtime.get(m))
    assert report == {
        "a": {"context_window": "match", "vision": "match"},
        "b": {"context_window": "override", "vision": "match"},
        "c": {"context_window": "match", "vision": "drift"},
    }
