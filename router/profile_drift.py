"""Profile <-> router drift detector.

Compares every profile's ``model.default`` and ``fallback_providers`` with the
pairs ``router.yaml`` routes on, and with Hermes' ``provider_models_cache.json``.
Deterministic: sorted, no timestamps, so identical drift gives identical text
(what a cron ``monitor`` needs to stay quiet). Reuses ``model_lint`` for parsing.

A profile model is a *floor*, so it need not be a tier primary, but it must be a
(provider, model) pair the router knows; a fallback must be a known pair, or a
known provider when it is a bare name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from . import model_lint


def _profiles(root: Path, profiles_dir: Path) -> List[Tuple[str, Path]]:
    found = [(p.parent.name, p) for p in sorted(Path(profiles_dir).glob("*/config.yaml"))]
    top = root / "config.yaml"
    if top.is_file():
        found.insert(0, ("(root)", top))
    return found


def drift(router_yaml: Path, profiles_dir: Path, cache_path: Path, root: Path) -> List[str]:
    """Sorted, de-duplicated drift lines; empty list means no drift."""
    pairs = model_lint.router_pairs(model_lint._load_yaml(Path(router_yaml)))
    known: Set[Tuple[str, str]] = {(p, m) for _, p, m in pairs}
    providers: Set[str] = {p for p, _ in known}
    cache = model_lint.load_cache(Path(cache_path))
    out: Set[str] = set()
    for name, cfg_path in _profiles(Path(root), Path(profiles_dir)):
        cfg: Dict[str, Any] = model_lint._load_yaml(cfg_path)
        for key, provider, model in model_lint.profile_pairs(cfg):
            where = f"{name}: {key}"
            if model is None:
                if provider not in providers:
                    out.add(f"{where}: provider {provider!r} not used by router.yaml")
            elif (provider, model) not in known:
                out.add(f"{where}: {provider}/{model} not in router.yaml")
        out.update(f"{name}: {p.split(': ', 1)[1]}" for p in model_lint.check(str(cfg_path), model_lint.profile_pairs(cfg), cache))
    return sorted(out)
