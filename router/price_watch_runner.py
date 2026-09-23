"""Executable edge for the isolated pricing-page watcher.

The router imports no network module.  This small runner owns supplier fetches,
the three-authority registration gate, state location and Kanban card creation;
it is intended for a daily Hermes cron job, never a routing decision.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Set

import yaml

from .capabilities import MODEL_CAPABILITIES
from .price_watch import ProviderAdapter, WatchResult, run_watch


DEFAULT_ADAPTERS = (
    ProviderAdapter("deepseek", "https://api-docs.deepseek.com/quick_start/pricing", "Peak hours"),
    ProviderAdapter("zai", "https://docs.z.ai/devpack/overview.md", "Singapore Standard Time"),
    ProviderAdapter(
        "zai",
        "https://docs.z.ai/devpack/overview.md",
        # WHICH MODELS THE PLAN COVERS is a second load-bearing clause on the same
        # page, and the window anchor above cannot see it. On 2026-08-26 the vendor
        # dropped glm-4.7 and glm-5-turbo from the plan and began auto-routing both
        # ids to glm-5.3-flash. The window clause did not change one character, so
        # this watcher reported "confirmed" while four names in the shipped policy
        # became vendor aliases — the request succeeds, another model answers, and
        # every trace reports the id that did not run.
        # The anchor is the substitution sentence itself: "Requests for
        # GLM-5.2/GLM-5.1 will be automatically routed to GLM-5.3, requests for
        # GLM-5-Turbo/GLM-4.7 will automatically be routed to GLM-5.3-Flash."
        # Its own words are what must be diffed, because the id on the RIGHT of
        # each arrow is what actually runs. Note the vendor writes "GLM-5-Flash" in
        # the supported-models bullet one line above and "GLM-5.3-Flash" here and
        # in the credit table; the callable id is glm-5.3-flash.
        "will automatically be routed to",
        key="zai-plan-model-coverage",
    ),
    ProviderAdapter(
        "xiaomi",
        "https://mimo.mi.com/docs/zh-CN/price/pay-as-you-go",
        # A página é servida no servidor (a medição de 2026-08-26 que "precisou de
        # browser" foi antes de o site mudar); a âncora antiga "按量付费" nunca
        # casava porque a página diz "按量计费", e a primeira ocorrência de
        # "按量计费" é um rótulo do menu. Esta frase única é a sentença que carrega
        # a regra "pay-as-you-go fatura pelo uso real, sem dimensão de tempo" — a
        # ausência de janela aqui é o que justifica não modelar o 0.8x no registry.
        "按实际 Token 用量消耗账户余额",
        key="xiaomi-pay-as-you-go",
    ),
    ProviderAdapter(
        "xiaomi",
        "https://mimo.mi.com/docs/zh-CN/quick-start/faq/token-plan",
        # A âncora antiga "Token Plan" casava primeiro no <meta property="og:title">
        # e o detector dizia confirmed vigiando o TÍTULO, não a regra. 非高峰期
        # aparece primeiro na linha da regra: "夜间优惠速率:非高峰期（北京时间
        # 0:00-8:00，即 UTC 16:00-24:00） 0.8x 消耗系数。" — o coeficiente e a janela.
        "夜间优惠消耗速率",
        key="xiaomi-token-plan",
    ),
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Iterable):
        for item in value:
            if isinstance(item, str):
                yield item


def registered_providers(
    auth: Mapping[str, Any],
    environment: Mapping[str, str],
    config: Mapping[str, Any],
    api_key_env_vars: Mapping[str, Any],
) -> Set[str]:
    """Return providers evidenced by OAuth, a known key, or a configured URL.

    The environment-variable names are supplied by Hermes' provider registry at
    the edge; this function never carries a duplicate provider→secret table.
    """
    found: Set[str] = set()
    pools = _mapping(auth).get("credential_pool")
    for provider, credentials in _mapping(pools).items():
        if isinstance(provider, str) and credentials:
            found.add(provider)
    for provider, credentials in _mapping(auth).items():
        if provider != "credential_pool" and isinstance(provider, str) and _mapping(credentials):
            found.add(provider)
    for provider, names in _mapping(api_key_env_vars).items():
        if isinstance(provider, str) and any(environment.get(name) for name in _strings(names)):
            found.add(provider)

    def visit(value: Any) -> None:
        current = _mapping(value)
        provider = current.get("provider")
        if isinstance(provider, str) and isinstance(current.get("base_url"), str):
            found.add(provider)
        for child in current.values():
            if isinstance(child, (Mapping, list, tuple)):
                visit(child)
        if isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(config)
    return found


def policy_providers(policy: Mapping[str, Any]) -> Set[str]:
    """Collect every provider currently named by the live routing policy."""
    found: Set[str] = set()

    def visit(value: Any) -> None:
        current = _mapping(value)
        provider = current.get("provider")
        if isinstance(provider, str):
            found.add(provider)
        for child in current.values():
            if isinstance(child, (Mapping, list, tuple)):
                visit(child)
        if isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(policy)
    return found


def _dotenv_names(path: Path) -> Set[str]:
    """Read only variable names: this gate must never load or expose a secret."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()
    names: Set[str] = set()
    for line in lines:
        key, separator, _value = line.partition("=")
        key = key.removeprefix("export ").strip()
        if separator and key.isidentifier():
            names.add(key)
    return names


def _fetch(url: str) -> str:
    """Fetch supplier HTML only at this isolated cron edge.

    MiMo's token-plan FAQ is rendered from its versioned browser bundle rather
    than the document shell.  Include that declared entry bundle so an anchor in
    the rendered FAQ remains observable without a browser or a brittle hashed
    asset URL in the policy.
    """
    import re
    from urllib.parse import urljoin, urlparse
    from urllib.request import Request, urlopen

    def get(target: str) -> str:
        request = Request(target, headers={"User-Agent": "Mozilla/5.0 (pricing provenance watcher)"})
        with urlopen(request, timeout=30) as response:  # noqa: S310 -- fixed supplier URLs above
            return response.read().decode("utf-8", errors="replace")

    page = get(url)
    if urlparse(url).hostname != "mimo.mi.com":
        return page
    entry = re.search(r'["\'](?P<src>/static/main\.[^"\']+\.js)["\']', page)
    if entry is None:
        return page
    bundle = get(urljoin(url, entry.group("src")))
    # The bundle is minified into one line.  Split object-property boundaries so
    # ProviderAdapter can preserve the actual localized clause, not the whole JS
    # program that happens to contain it.
    return page + "\n" + re.sub(r'(?=,"(?:[^"\\]|\\.)+":)', "\n", bundle)


def _create_card(payload: Dict[str, str]) -> None:
    """Open an unassigned review card; a watcher never edits the registry."""
    from hermes_cli.kanban_db import create_task
    from hermes_cli.kanban_db_connect import connect

    conn = connect(board=os.environ.get("HERMES_KANBAN_BOARD", "capability-router"))
    try:
        create_task(
            conn,
            title=payload["title"],
            body=payload["body"],
            created_by="price-watch",
            initial_status="running",
        )
    finally:
        conn.close()


def _hermes_inputs(hermes_home: Path) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, str], Mapping[str, Any]]:
    """Read authority facts without returning credentials or their values."""
    from hermes_cli.auth import PROVIDER_REGISTRY, _load_auth_store
    from hermes_cli.config import load_config_readonly

    auth = _load_auth_store(hermes_home / "auth.json")
    config = load_config_readonly()
    env_names = _dotenv_names(hermes_home / ".env") | set(os.environ)
    environment = {name: "present" for name in env_names}
    api_key_env_vars = {name: item.api_key_env_vars for name, item in PROVIDER_REGISTRY.items()}
    return auth, config, environment, api_key_env_vars


def run_daily(
    *,
    policy_path: Path,
    state_path: Path,
    hermes_home: Path,
    fetch: Callable[[str], str] = _fetch,
    create_card: Callable[[Dict[str, str]], None] = _create_card,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> WatchResult:
    """Run one offline-decision / online-observation pass for the live policy."""
    try:
        policy = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        policy = {}
    if not isinstance(policy, Mapping):
        policy = {}
    auth, config, environment, api_key_env_vars = _hermes_inputs(hermes_home)
    return run_watch(
        state_path=state_path,
        adapters=DEFAULT_ADAPTERS,
        registered_providers=registered_providers(auth, environment, config, api_key_env_vars),
        policy_providers=policy_providers(policy),
        registry=MODEL_CAPABILITIES,
        fetch=fetch,
        create_card=create_card,
        now=now,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check supplier pricing clauses and save provenance")
    parser.add_argument("--policy", type=Path, default=Path("router.yaml"))
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--hermes-home", type=Path, default=Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser())
    args = parser.parse_args(argv)
    result = run_daily(policy_path=args.policy, state_path=args.state, hermes_home=args.hermes_home)
    print(json.dumps(result.__dict__, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
