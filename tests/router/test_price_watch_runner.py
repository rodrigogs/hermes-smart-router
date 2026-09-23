"""Authority-gate tests for the executable pricing watcher."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
import types

import router.price_watch_runner as runner
from router.price_watch import WatchResult
from router.price_watch_runner import policy_providers, registered_providers


def test_registration_requires_a_real_authority_and_policy_reference() -> None:
    auth = {"deepseek": {"access_token": "oauth"}}
    environment = {"Z_AI_API_KEY": "present"}
    config = {"model": {"provider": "xiaomi", "base_url": "https://api.xiaomi.example"}}
    api_key_env_vars = {
        "deepseek": ("DEEPSEEK_API_KEY",),
        "zai": ("Z_AI_API_KEY",),
        "xiaomi": ("XIAOMI_API_KEY",),
    }
    policy = {
        "tiers": {
            "T1": {"model": "deepseek-v4-pro", "provider": "deepseek"},
            "T2": {"model": "glm-5.3", "provider": "zai"},
        }
    }

    assert registered_providers(auth, environment, config, api_key_env_vars) == {
        "deepseek", "zai", "xiaomi"
    }
    assert policy_providers(policy) == {"deepseek", "zai"}


def test_registered_providers_reads_credential_pool_and_ignores_empty_authorities() -> None:
    assert registered_providers(
        {"credential_pool": {"openai-codex": [{"kind": "oauth"}], "empty": []}},
        {},
        {"profiles": [{"provider": "local", "base_url": "http://localhost"}]},
        {},
    ) == {"openai-codex", "local"}


def test_authority_readers_ignore_malformed_values_and_accept_one_env_name() -> None:
    assert registered_providers({"bad": []}, {"ONE": "present"}, {}, {"keyed": "ONE"}) == {"keyed"}
    assert policy_providers({"tiers": [None, {"provider": "deepseek"}], "ignored": "value"}) == {"deepseek"}
    assert list(runner._strings(None)) == []
    assert list(runner._strings(["ONE", None])) == ["ONE"]


def test_dotenv_names_never_returns_values(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("export DEEPSEEK_API_KEY=not-for-output\nINVALID-NAME=x\n", encoding="utf-8")

    assert runner._dotenv_names(env) == {"DEEPSEEK_API_KEY"}
    assert runner._dotenv_names(tmp_path / "missing") == set()


def test_daily_runner_uses_all_three_authorities_without_network(monkeypatch, tmp_path: Path) -> None:
    policy = tmp_path / "router.yaml"
    policy.write_text("tiers:\n  T1:\n    provider: deepseek\n", encoding="utf-8")
    home = tmp_path / "hermes"
    home.mkdir()
    cards: list[dict[str, str]] = []

    monkeypatch.setattr(
        runner,
        "_hermes_inputs",
        lambda _home: ({"credential_pool": {"deepseek": [{}]}}, {}, {}, {}),
    )
    monkeypatch.setattr(runner, "DEFAULT_ADAPTERS", (runner.DEFAULT_ADAPTERS[0],))

    result = runner.run_daily(
        policy_path=policy,
        state_path=tmp_path / "state.json",
        hermes_home=home,
        fetch=lambda _url: "<p>Peak hours are 01:00 - 04:00 UTC.</p>",
        create_card=cards.append,
        now=lambda: datetime(2026, 8, 26, tzinfo=timezone.utc),
    )

    assert result.checked == ["deepseek"]
    assert cards == []


def test_daily_runner_fails_closed_when_policy_is_invalid(monkeypatch, tmp_path: Path) -> None:
    policy = tmp_path / "router.yaml"
    policy.write_text("tiers: [", encoding="utf-8")
    monkeypatch.setattr(runner, "_hermes_inputs", lambda _home: ({}, {}, {}, {}))

    result = runner.run_daily(
        policy_path=policy,
        state_path=tmp_path / "state.json",
        hermes_home=tmp_path,
        fetch=lambda _url: (_ for _ in ()).throw(AssertionError("must not fetch")),
    )

    assert result == WatchResult(checked=[], confirmed=[], changed=[], failed=[])


def test_daily_runner_fails_closed_when_policy_is_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "_hermes_inputs", lambda _home: ({}, {}, {}, {}))

    result = runner.run_daily(
        policy_path=tmp_path / "missing.yaml",
        state_path=tmp_path / "state.json",
        hermes_home=tmp_path,
        fetch=lambda _url: (_ for _ in ()).throw(AssertionError("must not fetch")),
    )

    assert result == WatchResult(checked=[], confirmed=[], changed=[], failed=[])


def test_daily_runner_rejects_a_scalar_policy(monkeypatch, tmp_path: Path) -> None:
    policy = tmp_path / "router.yaml"
    policy.write_text("true\n", encoding="utf-8")
    monkeypatch.setattr(runner, "_hermes_inputs", lambda _home: ({}, {}, {}, {}))

    assert runner.run_daily(
        policy_path=policy,
        state_path=tmp_path / "state.json",
        hermes_home=tmp_path,
        fetch=lambda _url: (_ for _ in ()).throw(AssertionError("must not fetch")),
    ) == WatchResult(checked=[], confirmed=[], changed=[], failed=[])


def test_hermes_inputs_uses_names_not_dotenv_values(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / "hermes"
    home.mkdir()
    (home / ".env").write_text("Z_AI_API_KEY=secret-value\n", encoding="utf-8")
    auth_module = types.SimpleNamespace(
        PROVIDER_REGISTRY={"zai": types.SimpleNamespace(api_key_env_vars=("Z_AI_API_KEY",))},
        _load_auth_store=lambda _path: {"credential_pool": {"zai": [{}]}},
    )
    config_module = types.SimpleNamespace(load_config_readonly=lambda: {"model": {"provider": "zai"}})
    monkeypatch.setitem(sys.modules, "hermes_cli.auth", auth_module)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", config_module)

    auth, config, environment, env_vars = runner._hermes_inputs(home)

    assert auth == {"credential_pool": {"zai": [{}]}}
    assert config == {"model": {"provider": "zai"}}
    assert environment["Z_AI_API_KEY"] == "present"
    assert "secret-value" not in environment
    assert env_vars == {"zai": ("Z_AI_API_KEY",)}


def test_create_card_and_main_are_thin_edges(monkeypatch, capsys) -> None:
    calls: list[object] = []

    class Connection:
        def close(self) -> None:
            calls.append("closed")

    kanban_db = types.SimpleNamespace(
        create_task=lambda conn, **kwargs: calls.append((conn, kwargs)),
    )
    kanban_db_connect = types.SimpleNamespace(
        connect=lambda board: calls.append(board) or Connection(),
    )
    # CI installs no hermes_cli at all.  The production edge imports the real
    # connection factory directly, rather than through the expired compat path.
    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__path__ = []  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_db", kanban_db)
    monkeypatch.setitem(sys.modules, "hermes_cli.kanban_db_connect", kanban_db_connect)
    runner._create_card({"title": "review", "body": "evidence"})
    assert calls[0] == "capability-router"
    assert calls[-1] == "closed"

    monkeypatch.setattr(
        runner,
        "run_daily",
        lambda **_kwargs: WatchResult(checked=["deepseek"], confirmed=[], changed=[], failed=[]),
    )
    assert runner.main(["--state", "/tmp/state.json"]) == 0
    assert '"checked": ["deepseek"]' in capsys.readouterr().out


def test_fetch_uses_a_browser_user_agent_without_live_network(monkeypatch) -> None:
    import urllib.request

    seen: dict[str, object] = {}

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return "texto ✓".encode("utf-8")

    def fake_urlopen(request: object, timeout: int) -> Response:
        assert isinstance(request, urllib.request.Request)
        seen["url"] = request.full_url
        seen["agent"] = request.get_header("User-agent")
        seen["timeout"] = timeout
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert runner._fetch("https://supplier.example/pricing") == "texto ✓"
    assert seen == {
        "url": "https://supplier.example/pricing",
        "agent": "Mozilla/5.0 (pricing provenance watcher)",
        "timeout": 30,
    }


def test_xiaomi_anchors_target_the_rule_phrase_not_the_title() -> None:
    # These strings are the contract with the supplier pages: the token-plan
    # anchor must be the off-peak clause (非高峰期), never "Token Plan", which
    # resolves to the og:title meta tag; the pay-as-you-go anchor must be the
    # metered-billing sentence, never "按量付费", which never appears on the
    # page ("按量计费" does, and its first hit is a nav label). Changing them is
    # a deliberate operator act, so the pin breaks loudly instead of silently
    # watching a title again.
    by_key = {adapter.key: adapter for adapter in runner.DEFAULT_ADAPTERS}
    assert by_key["xiaomi-token-plan"].anchor == "非高峰期"
    assert by_key["xiaomi-pay-as-you-go"].anchor == "按实际 Token 用量消耗账户余额"


def test_zai_is_watched_for_model_coverage_as_well_as_for_its_window() -> None:
    """A window is not the only fact the config leans on: coverage is the other.

    On 2026-08-26 the vendor dropped glm-4.7 and glm-5-turbo from the plan and
    began auto-routing both ids to glm-5.3-flash. The peak-hours clause — the only
    zai clause watched until then — did not change one character, so the watcher
    reported no change while four names in the shipped policy became aliases for a
    model nobody had chosen. Two clauses, two anchors, one page.
    """
    zai = [adapter for adapter in runner.DEFAULT_ADAPTERS if adapter.provider == "zai"]
    anchors = {adapter.anchor for adapter in zai}
    assert "Singapore Standard Time" in anchors, "the peak window is still watched"
    assert "will automatically be routed to" in anchors, (
        "the substitution sentence is what says which id actually runs"
    )
    # Two adapters on one provider need distinct keys or one overwrites the other's
    # recorded literal — the same shape the two xiaomi pages already have.
    keys = [adapter.key for adapter in zai]
    assert len(set(keys)) == len(keys), keys
    assert "zai-plan-model-coverage" in keys


def test_the_zai_coverage_anchor_resolves_to_the_substitution_clause() -> None:
    """The anchor is checked against the page's own wording, not just declared.

    An anchor that matches nothing raises at fetch time and lands in `failed`; one
    that matches the page shell raises too. Both are silent for as long as nobody
    runs the cron, so the resolution is exercised here against a captured line.
    """
    page = (
        "### Supported Models\n"
        "* All plans support **GLM-5.3**, GLM-5-Flash.\n"
        "* Requests for GLM-5.2/GLM-5.1 will be automatically routed to GLM-5.3, "
        "requests for GLM-5-Turbo/GLM-4.7 will automatically be routed to "
        "GLM-5.3-Flash.\n"
        "#### Usage Credit Allowance\n"
    )
    adapter = next(
        a for a in runner.DEFAULT_ADAPTERS if a.key == "zai-plan-model-coverage"
    )
    literal = adapter.extract(page)
    # The clause itself, with both arrows in it: the id on the RIGHT of each is
    # what runs, and that is the half a config edit has to follow.
    assert "GLM-5-Turbo/GLM-4.7 will automatically be routed to GLM-5.3-Flash" in literal
    assert "GLM-5.2/GLM-5.1 will be automatically routed to GLM-5.3" in literal
    # And it is the clause, not the bullet above it that only lists names.
    assert "All plans support" not in literal
