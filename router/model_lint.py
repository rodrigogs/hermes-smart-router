"""Model-existence lint: every model/provider pair named in router.yaml and in
the profile configs must exist in Hermes' ``provider_models_cache.json``.

Pure functions plus one thin ``run`` that reads files. Each finding names the
file, the key path and the model, so a typo is fixable without a search.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import yaml

#: router.yaml subtrees that name models without routing to them.
_SKIP_KEYS = {"blocklist"}

Pair = Tuple[str, Optional[str], Optional[str]]  # (key path, provider, model)


def load_cache(path: Path) -> Dict[str, Set[str]]:
    """``{provider: {model ids}}`` from the cache file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        prov: {str(m) for m in (entry or {}).get("models", [])}
        for prov, entry in raw.items()
    }


def _walk(node: Any, key: str) -> Iterator[Pair]:
    if isinstance(node, dict):
        model, provider = node.get("model"), node.get("provider")
        if isinstance(model, str) and isinstance(provider, str):
            yield key, provider, model
        for k, v in node.items():
            if k not in _SKIP_KEYS:
                yield from _walk(v, f"{key}.{k}" if key else str(k))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{key}[{i}]")


def router_pairs(config: Dict[str, Any]) -> List[Pair]:
    """Every dict carrying both ``model`` and ``provider`` strings."""
    return list(_walk(config, ""))


def profile_pairs(config: Dict[str, Any]) -> List[Pair]:
    """``model.default`` and every ``fallback_providers`` entry of a profile.

    A bare-string fallback names only a provider (the model comes from
    ``fallback_model``), so it is checked for provider existence alone.
    """
    out: List[Pair] = []
    model = config.get("model")
    model = model if isinstance(model, dict) else {}
    if isinstance(model.get("default"), str):
        out.append(("model.default", model.get("provider"), model["default"]))
    for base, entries in (
        ("model.fallback_providers", model.get("fallback_providers")),
        ("fallback_providers", config.get("fallback_providers")),
    ):
        for i, e in enumerate(entries or []):
            key = f"{base}[{i}]"
            if isinstance(e, dict):
                out.append((key, e.get("provider"), e.get("model")))
            else:
                out.append((key, str(e), None))
    return out


def check(
    file: str, pairs: List[Pair], cache: Dict[str, Set[str]]
) -> List[str]:
    """One message per pair whose provider or model is absent from the cache."""
    problems: List[str] = []
    for key, provider, model in pairs:
        if provider not in cache:
            problems.append(
                f"{file}: {key}: provider {provider!r} not in cache (model {model!r})"
            )
        elif model is not None and model not in cache[provider]:
            problems.append(
                f"{file}: {key}: model {model!r} not in cache for provider {provider!r}"
            )
    return problems


def _load_yaml(path: Path) -> Dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def run(router_yaml: Path, profiles_dir: Path, cache_path: Path) -> List[str]:
    """All divergences across router.yaml and ``profiles_dir/*/config.yaml``."""
    cache = load_cache(cache_path)
    problems = check(str(router_yaml), router_pairs(_load_yaml(router_yaml)), cache)
    for cfg in sorted(Path(profiles_dir).glob("*/config.yaml")):
        problems += check(str(cfg), profile_pairs(_load_yaml(cfg)), cache)
    return problems
