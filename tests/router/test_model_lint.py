import json

import pytest
import yaml

from router import cli, model_lint

CACHE = {
    "zai": {"models": ["glm-5.3"]},
    "copilot": {"models": ["claude-sonnet-5.5"]},
    "bare": {},
}


@pytest.fixture
def env(tmp_path):
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps(CACHE))
    router = tmp_path / "router.yaml"
    profiles = tmp_path / "profiles"
    (profiles / "p1").mkdir(parents=True)
    (profiles / "p1" / "config.yaml").write_text(
        yaml.safe_dump({"model": {"default": "glm-5.3", "provider": "zai"}})
    )
    router.write_text(yaml.safe_dump({"tiers": {"T1": {"model": "glm-5.3", "provider": "zai"}}}))
    return router, profiles, cache


def test_clean_files_pass(env):
    assert model_lint.run(*env) == []


def test_router_bad_model_names_file_key_model(env):
    router, profiles, cache = env
    router.write_text(
        yaml.safe_dump(
            {
                "tiers": {"T1": {"model": "glm-9", "provider": "zai", "fallback": [
                    {"model": "x", "provider": "nope"}]}},
                "blocklist": {"manual_ban": [{"model": "ghost", "provider": "zai"}]},
                "rules": [{"model": "T1"}],
            }
        )
    )
    out = model_lint.run(router, profiles, cache)
    assert len(out) == 2
    assert f"{router}: tiers.T1: model 'glm-9' not in cache for provider 'zai'" in out[0]
    assert "tiers.T1.fallback[0]: provider 'nope' not in cache" in out[1]


def test_profile_pairs_and_fallbacks(env):
    router, profiles, cache = env
    cfg = profiles / "p1" / "config.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "model": {"default": "bad", "provider": "zai", "fallback_providers": ["zai", "gone"]},
                "fallback_providers": [
                    {"provider": "copilot", "model": "claude-sonnet-5.5"},
                    {"provider": "copilot", "model": "missing"},
                ],
            }
        )
    )
    out = model_lint.run(router, profiles, cache)
    assert len(out) == 3
    assert "model.default: model 'bad'" in out[0] and str(cfg) in out[0]
    assert "model.fallback_providers[1]: provider 'gone'" in out[1]
    assert "fallback_providers[1]: model 'missing'" in out[2]


def test_profile_without_model_section_and_non_dict_yaml(env):
    router, profiles, cache = env
    (profiles / "p1" / "config.yaml").write_text("- just\n- a list\n")
    router.write_text("")
    assert model_lint.run(router, profiles, cache) == []


def test_cli_exit_codes(env, capsys):
    router, profiles, cache = env
    argv = ["--config", str(router), "model-lint", "--profiles-dir", str(profiles),
            "--cache", str(cache)]
    cli.main(argv)
    assert "all models exist" in capsys.readouterr().out
    router.write_text(yaml.safe_dump({"a": {"model": "nope", "provider": "zai"}}))
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 1
    assert "model 'nope'" in capsys.readouterr().out


def test_cli_defaults_to_hermes_root(env, monkeypatch, tmp_path, capsys):
    router, profiles, cache = env
    root = tmp_path / "root"
    root.mkdir()
    (root / "profiles").symlink_to(profiles)
    (root / "provider_models_cache.json").write_text(cache.read_text())
    monkeypatch.setenv("HERMES_HOME", str(root))
    cli.main(["--config", str(router), "model-lint"])
    assert "all models exist" in capsys.readouterr().out
