import json

import yaml

from router import profile_drift

CACHE = {"zai": {"models": ["glm-5.3", "x"]}, "copilot": {"models": ["c1"]}}


def _setup(tmp_path, root_cfg=None):
    (tmp_path / "cache.json").write_text(json.dumps(CACHE))
    (tmp_path / "router.yaml").write_text(
        yaml.safe_dump({"tiers": {"T1": {"model": "glm-5.3", "provider": "zai"}}})
    )
    p = tmp_path / "profiles" / "p1"
    p.mkdir(parents=True)
    if root_cfg:
        (tmp_path / "config.yaml").write_text(yaml.safe_dump(root_cfg))
    return p


def _run(tmp_path):
    return profile_drift.drift(
        tmp_path / "router.yaml", tmp_path / "profiles", tmp_path / "cache.json", tmp_path
    )


def test_clean_is_empty(tmp_path):
    p = _setup(tmp_path)
    (p / "config.yaml").write_text(yaml.safe_dump(
        {"model": {"default": "glm-5.3", "provider": "zai", "fallback_providers": ["zai"]}}))
    assert _run(tmp_path) == []


def test_drift_reported_sorted_and_deterministic(tmp_path):
    p = _setup(tmp_path, {"model": {"default": "gone", "provider": "zai"}})
    (p / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "c1", "provider": "copilot", "fallback_providers": ["nope"]},
        "fallback_providers": [{"provider": "zai", "model": "x"}],
    }))
    out = _run(tmp_path)
    assert out == sorted(out) and out == _run(tmp_path)
    assert "(root): model.default: zai/gone not in router.yaml" in out
    assert "(root): model.default: model 'gone' not in cache for provider 'zai'" in out
    assert "p1: model.default: copilot/c1 not in router.yaml" in out
    assert "p1: model.fallback_providers[0]: provider 'nope' not used by router.yaml" in out
    assert "p1: fallback_providers[0]: zai/x not in router.yaml" in out
