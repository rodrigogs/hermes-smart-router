"""Classifier provider must sit outside the tier-primary provider set (card t_a7baaba6)."""

import json
from pathlib import Path

import yaml

from router import model_lint
from router.rules import lint, lint_warnings

ROOT = Path(__file__).resolve().parents[2]
PROPOSED = ROOT / "docs" / "router.proposed.yaml"
SHIPPED = ROOT / "router.example.yaml"
CACHE = Path.home() / ".hermes" / "provider_models_cache.json"


def _load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _primary_providers(cfg):
    return {t["provider"] for t in cfg["tiers"].values()}


def _classifier_warnings(cfg):
    return [w for w in lint_warnings(cfg) if "failure domain" in w]


def test_shared_provider_is_flagged():
    cfg = _load(SHIPPED)
    cfg["classifier"]["provider"] = "zai"
    cfg["classifier"]["chain"][0]["provider"] = "zai"
    out = _classifier_warnings(cfg)
    assert len(out) == 2
    assert "classifier: provider 'zai'" in out[0]
    assert "classifier.chain[0]: provider 'zai'" in out[1]


def test_separate_provider_is_clean():
    cfg = _load(SHIPPED)
    cfg["classifier"]["provider"] = "copilot"
    cfg["classifier"]["chain"][0]["provider"] = "copilot"
    assert _classifier_warnings(cfg) == []


def test_no_classifier_block_is_silent():
    cfg = _load(SHIPPED)
    del cfg["classifier"]
    assert _classifier_warnings(cfg) == []


def test_proposed_classifier_outside_primary_providers():
    cfg = _load(PROPOSED)
    assert cfg["classifier"]["provider"] not in _primary_providers(cfg)
    assert cfg["classifier"]["chain"][0]["provider"] not in _primary_providers(cfg)
    assert (cfg["classifier"]["model"], cfg["classifier"]["provider"]) == (
        cfg["classifier"]["chain"][0]["model"],
        cfg["classifier"]["chain"][0]["provider"],
    )
    assert lint(cfg) == []
    assert _classifier_warnings(cfg) == []


def test_proposed_models_exist_in_cache(tmp_path):
    cache = tmp_path / "cache.json"
    cache.write_text(
        json.dumps(
            {
                "copilot": {"models": ["claude-haiku-5.5", "gpt-5.4-mini"]},
                "deepseek": {"models": ["deepseek-v4-flash"]},
            }
        )
    )
    cfg = _load(PROPOSED)
    pairs = [
        (f"classifier.chain[{i}]", h["provider"], h["model"])
        for i, h in enumerate(cfg["classifier"]["chain"])
    ]
    assert model_lint.check(str(PROPOSED), pairs, model_lint.load_cache(cache)) == []
