"""Derive context window and vision from the Hermes runtime, flag manual overrides.

The registry in ``capabilities.py`` used to be the only source for ``context_window`` and
``vision``; the runtime (``agent.models_dev.get_model_capabilities``, which feeds
``agent.model_metadata``) already knows both. This module reads the runtime and reconciles
it with the registry: an entry either MATCHES the runtime, or carries ``runtime_override``
naming the fields it deliberately overrides. Anything else is drift.

The runtime is optional (this plugin is deployed by copy and also runs in a sidecar), so
a missing runtime or a model the runtime does not know yields ``None`` — never a guess.
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, List, Optional

try:
    from .capabilities import MODEL_CAPABILITIES
except ImportError:  # pragma: no cover - flat layout used by the test harness
    from router.capabilities import MODEL_CAPABILITIES

DERIVED_FIELDS = ("context_window", "vision")

Lookup = Callable[[str, str], Optional[Dict[str, Any]]]


def load_runtime_lookup(module: str = "agent.models_dev") -> Optional[Lookup]:
    """Return ``lookup(provider, model) -> {context_window, vision} | None``, or None
    when the runtime module cannot be imported."""
    try:
        mod = importlib.import_module(module)
        getter = mod.get_model_capabilities
    except (ImportError, AttributeError):
        return None

    def lookup(provider: str, model: str) -> Optional[Dict[str, Any]]:
        caps = getter(provider, model)
        if caps is None:
            return None
        return {
            "context_window": caps.context_window,
            "vision": caps.supports_vision,
        }

    return lookup


def runtime_capabilities(
    model: str, lookup: Optional[Lookup] = None
) -> Optional[Dict[str, Any]]:
    """Runtime-derived context/vision for a registered ``model``, or None."""
    entry = MODEL_CAPABILITIES.get(model)
    lookup = lookup if lookup is not None else load_runtime_lookup()
    if entry is None or lookup is None:
        return None
    return lookup(entry["provider"], model)


def reconcile(lookup: Optional[Lookup] = None) -> Dict[str, Dict[str, str]]:
    """Per registered model, per derived field: ``match`` | ``override`` | ``drift``.

    Models the runtime does not know (or no runtime at all) are omitted.
    """
    lookup = lookup if lookup is not None else load_runtime_lookup()
    report: Dict[str, Dict[str, str]] = {}
    if lookup is None:
        return report
    for model, entry in MODEL_CAPABILITIES.items():
        derived = lookup(entry["provider"], model)
        if derived is None:
            continue
        flagged: List[str] = list(entry.get("runtime_override", ()))
        report[model] = {
            field: (
                "match"
                if entry.get(field) == derived.get(field)
                else "override" if field in flagged else "drift"
            )
            for field in DERIVED_FIELDS
        }
    return report
