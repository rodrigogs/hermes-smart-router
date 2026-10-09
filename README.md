# hermes-smart-router

[![CI](https://github.com/rodrigogs/hermes-smart-router/actions/workflows/ci.yml/badge.svg)](https://github.com/rodrigogs/hermes-smart-router/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/branch%20coverage-100%25-brightgreen)](https://github.com/rodrigogs/hermes-smart-router/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-1959%20passed-brightgreen)](https://github.com/rodrigogs/hermes-smart-router/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.3.0-informational)](https://github.com/rodrigogs/hermes-smart-router)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/rodrigogs/hermes-smart-router/blob/main/LICENSE)

A [Hermes Agent](https://github.com/NousResearch/hermes-agent) plugin that adds a
`delegate_profile` tool for spawning subagents under a **different** Hermes
profile as a **fully isolated subprocess** — with an optional **capability router**
that auto-selects the best profile + model based on task difficulty.

## Why this exists (read this first)

The built-in `delegate_task` **already supports** `profile=` for *in-process*
cross-profile delegation — it swaps the child's config, secret scope, SOUL, and
toolsets within the same process. That path is fast.

This plugin is for the **subprocess-isolation** case, where you want a hard
process boundary around the child:

|                                     | `delegate_task(profile=...)` | `delegate_profile` (this plugin) |
|-------------------------------------|------------------------------|----------------------------------|
| Isolation                           | In-process (shared process)  | Separate OS process              |
| Child crashes parent?               | Yes (same process)           | No (process boundary)            |
| Child toolset                       | Can only **narrow** parent's | Target profile's **full** toolset|
| Different Hermes version in child?  | No                           | Yes                              |
| Overhead                            | Low                          | Process spawn + cold start       |
| Best for                            | Fast parallel subtasks       | Crash isolation / full toolset   |

Rule of thumb: if you'd be happy running the subagent in the current process,
use `delegate_task(profile=...)`. If the subprocess boundary itself is the
point — crash safety, the target profile's full toolset, a different Hermes
version — use `delegate_profile`.

```text
delegate_profile(goal="Run the firmware flash suite",     profile="firmware-engineer")
delegate_profile(goal="Review this PR for security",      profile="reviewer")
delegate_profile(goal="Summarize these papers",           profile="researcher", model="anthropic/claude-sonnet-4")
```

## What it does

- Adds a **`delegate_profile`** tool to the `delegation` toolset.
- For **cross-profile** calls, spawns a one-shot
  `hermes -p <profile> chat -q "<goal>"` subprocess. The child runs fully
  isolated under the target profile — its own process, session, skills,
  memory, and model.
- For **same-profile** calls (when an explicit profile matches the active profile),
  transparently routes to the built-in `delegate_task` via `ctx.dispatch_tool`
  (which wires up `parent_agent`) so you pay no subprocess overhead. Omit
  `profile` or pass `auto` to use the capability router instead.
- **Validates the target profile exists before spawning** — a typo produces an
  instant, actionable error listing available profiles, not a confusing
  subprocess failure.
- Registers a **`post_tool_call` hook** that logs an advisory warning if
  `delegate_task` is invoked with a `profile` argument.

## Install

From the plugin registry:

```bash
hermes plugins install rodrigogs/hermes-smart-router
hermes plugins enable delegate-profile
```

Or manually by cloning into your plugins directory:

```bash
git clone https://github.com/rodrigogs/hermes-smart-router.git \
  ~/.hermes/plugins/hermes-smart-router
hermes plugins enable delegate-profile
```

Restart your session (or run `/reset` in the CLI / `/restart` in the gateway)
after enabling — plugin loads happen at startup.

### Requirements

- Hermes Agent (any recent version)
- The `delegation` toolset enabled (`hermes tools enable delegation`)
- One or more additional Hermes profiles. Create one with
  `hermes profile create <name>` and list them with `hermes profile list`.
- **For the LLM classifier only — an explicit host trust grant.** Without it
  Stage 1 never runs, and the symptom does not say so.

  The classifier always passes `provider=` and `model=` to `ctx.llm.complete`,
  which trips Hermes' per-plugin LLM trust gate unless BOTH overrides are
  granted:

  ```yaml
  plugins:
    entries:
      hermes-smart-router:
        llm:
          allow_provider_override: true
          allow_model_override: true
  ```

  This is the plugin's one measured production incident. Of 47 recorded routing
  decisions, **zero** used the classifier and 16 ended in the fail-safe with
  `no_classifier` / `classifier_error` — because the refusal is caught and
  degraded, so a missing grant looks exactly like a router that simply never
  needed to classify. Deterministic Stage-0 routing works fine without the
  grant; only rules whose `then` carries `action: classify` need it.

## Usage

### Minimal

```python
delegate_profile(
    goal="Refactor the auth module to use async/await",
    profile="coder",
)
```

### With context and model override

```python
delegate_profile(
    goal="Review the diff in PR #142 for SQL injection and auth flaws",
    profile="reviewer",
    context="Repo: ~/projects/myapp. Stack: FastAPI + SQLAlchemy. See git diff origin/main.",
    model="anthropic/claude-sonnet-4",
    timeout=600,
)
```

### Parameters

| Parameter  | Type    | Required | Default | Notes |
|------------|---------|----------|---------|-------|
| `prompt`   | string  | yes      | —       | What the subagent should accomplish. Be self-contained — the child has no context from your session. (Legacy alias: `goal` — still accepted, prefer `prompt`.) |
| `profile`  | string  | no       | `auto`  | Target Hermes profile name (schema enum: known profiles plus `default`/`auto`). Must exist (`hermes profile list`). If omitted or `auto`, the capability router picks the best profile + model based on task difficulty. If it matches the active profile, routes to in-process `delegate_task`. |
| `context`  | string  | no       | —       | Background info: file paths, error messages, project structure, constraints. |
| `model`    | string  | no       | profile default | Model override passed as `-m` to the child. |
| `timeout`  | integer | no       | `600`   | Absolute hard-ceiling seconds for the subprocess. Tighter independent watchdogs also apply (no-first-output 120 s, inter-output idle 300 s). Override via `plugins.entries.hermes-smart-router.watchdog` in `config.yaml`, or `HERMES_DELEGATE_PROFILE_TIMEOUT`. |

### Result format

Returns a JSON string. On success:

```json
{
  "success": true,
  "subagent_id": "dp_a1b2c3d4e5f6",
  "profile": "reviewer",
  "result": "<stdout from the child hermes process, last 8000 chars>",
  "elapsed_s": 42.3
}
```

On failure (non-zero exit, timeout, or missing binary):

```json
{
  "success": false,
  "failure_kind": "nonzero_exit",
  "retryable": false,
  "subagent_id": "dp_a1b2c3d4e5f6",
  "profile": "reviewer",
  "model": "glm-5.3-flash",
  "error": "Subprocess exited with code 1",
  "stderr": "<last 2000 chars of stderr>",
  "elapsed_s": 12.1
}
```

`failure_kind` and `retryable` are the point of the envelope — they are what let
an orchestrator choose retry / fallback / give-up without parsing prose.

| `failure_kind` | `retryable` | meaning |
|---|---|---|
| `bad_args`         | —     | `prompt` or `profile` missing. |
| `unknown_profile`  | —     | no such profile; the envelope lists the real ones. |
| `binary_not_found` | —     | the `hermes` binary is not on `PATH`. |
| `nonzero_exit`     | false | the child exited non-zero — an app-level error a retry repeats. |
| `spawn_error`      | true  | the spawn itself failed. |
| `ttfb_stall`       | true  | no first byte of output — startup wedged. |
| `idle_stall`       | true  | no NEW output for the idle window — mid-run stall. |
| `hard_timeout`     | true  | the absolute ceiling was hit. |
| `crash` / `crash_or_oom` | true | the child died on a signal. |
| `quota_exhausted`  | true  | the rail is out of quota — overrides the kind above it. |
| `agent_error`      | true  | the child reported a failure in its own output. |
| `inline_error`     | true  | the same-profile in-process path raised. |
| `at_capacity`      | true  | no concurrency slot within the queue wait. |

Only `retryable` drives the loop: the executor advances to the next
(model, provider) target while it is true, and stops on the first false — so
`nonzero_exit` (and the three arg/environment failures, which never reach the
loop) end the turn, while every stall or quota failure fails over to the next
rail.

For a missing or non-existent profile:

```json
{
  "success": false,
  "failure_kind": "unknown_profile",
  "error": "Profile 'reviwer' does not exist. Create it with: hermes profile create reviwer",
  "profile": "reviwer",
  "available_profiles": ["coder", "reviewer", "tester"],
  "hint": "Available profiles: coder, reviewer, tester"
}
```

For missing required args, returns
`{"error": "prompt is required", "failure_kind": "bad_args"}` or
`{"error": "profile is required", "failure_kind": "bad_args"}`.

## How it works

1. Resolves the active profile name via Hermes's own `get_active_profile_name()`
   (falls back to `HERMES_PROFILE` env, then `default`).
2. Validates the target profile exists (`hermes_cli.profiles.profile_exists`).
3. If the target **equals** the active profile, routes to `delegate_task`
   through `ctx.dispatch_tool` (in-process, no spawn).
4. Otherwise resolves the `hermes` binary (prefers the one in the current
   venv, falls back to `PATH`), builds the prompt as
   `Context: <context>\n\nTask: <goal>` when `context` is provided, and
   spawns `hermes -p <profile> chat -q "<prompt>"` (plus `-m <model>` when set)
   with `capture_output=True` and the resolved `timeout`.
5. Forwards `HERMES_HOME` so the child resolves profiles from the same place
   as the parent, and sets `HERMES_DELEGATE_PROFILE_DISABLE=1` to prevent
   recursive delegation inside the child.
6. Returns a JSON envelope (see [Result format](#result-format)).

### Smart Router

When `profile` is omitted or set to `auto`, the plugin runs a **capability
router** that picks the best profile + model for the task:

- **Stage 0 — Deterministic rules:** matches task signals (verb class,
  code presence, keyword patterns) against user-defined rules in
  `router.yaml`. Fast, cheap, no LLM call.
- **Stage 1 — LLM classifier:** fires only when rules can't decide (rules
  with `action: classify`). Uses a pinned model (glm-5.3-flash/zai, temp=0,
  token-capped) to classify task difficulty.
- **Fail-safe:** if the classifier is unavailable, falls back to the
  `fail_safe` block in `router.yaml`. Note it is the CHEAPEST plan rail, not a
  strong model: shipped it is `glm-5.3-flash @ zai` with the same fallback
  queue as T1, duplicated on purpose so it stays readable by a caller that has
  no tier table. Keep it in step with T1 by hand.
- **Blocklist:** manual bans + auto-breaker with exponential backoff
  cooldowns for models that stall repeatedly.
- **Decision log:** every routing decision is recorded for observability.

Example routing behavior with the default config:

| Task | Router picks |
|------|-------------|
| `"Rename getCwd in utils.py"` | `coder` + `glm-5.3-flash` (T1) |
| `"Debug race condition in pool"` | `coder` + `gpt-5.5` @ `openai-codex` (T4) |
| `"Review PR for security"` | `reviewer` + classify (→ fail_safe if no LLM) |

Configure via `router.yaml` — rules, tiers, blocklist, classifier model,
and fail-safe are all user-editable. Run `python -m router.cli explain "task"`
to see routing decisions from the CLI.

The `post_tool_call` hook (`_on_post_tool_call`) is a no-op for every tool
except `delegate_task`. When `delegate_task` is invoked with a `profile`
parameter, it emits a `logger.warning`. The hook never blocks or modifies the
call — it's purely advisory (the built-in `delegate_task(profile=...)` is a
legitimate in-process path; the warning is a nudge for callers who actually
want subprocess isolation).

### Hermes One extension

The repository ships a **read-only** Smart Router panel for
[Hermes One](https://github.com/nesquena/hermes-webui). It is an extension
inside the existing WebUI, not a second public application:

```text
Hermes One panel
  → same-origin, consented extension-sidecar proxy
    → 127.0.0.1:8791 router sidecar (token-v1)
      → the plugin's router.yaml + router core
```

V1 exposes live status, declarative policy, real breaker state and **Trace
Route**, a deterministic Stage-0 dry-run. It never calls the classifier from
the UI and never presents simulations as real delegation telemetry. Policy
edits go through the guarded /plan + /apply write path — rule position and
state, tier destinations, never `when` conditions — and it never resets
breakers or dispatches agents.

Install the assets and generated user service:

```bash
python3 scripts/install_hermes_one_router.py \
  --extension-root ~/hermes-one-extensions \
  --systemd-dir ~/.config/systemd/user \
  --plugin-dir ~/.hermes/plugins/hermes-smart-router \
  --hermes-home "$HERMES_HOME" \
  --webui-state-dir "$HERMES_WEBUI_STATE_DIR"

systemctl --user daemon-reload
systemctl --user enable --now hermes-router-sidecar.service
curl http://127.0.0.1:8791/health
```

Reload Hermes One, then approve the declared **Smart Router** sidecar in
**Settings → Extensions**. That approval is intentionally manual: Hermes One
mints the private `token-v1` file and injects it only into consented proxy
requests. Until then `/status` returns `503`; do not create a token file
manually.

Security invariants:

- the sidecar binds to loopback only — never expose port `8791` through
  Tailscale Serve/Funnel or a reverse proxy;
- all routes except `/health` require `X-Hermes-Sidecar-Token` and fail closed
  with `401` (wrong/missing header) or `503` (token not provisioned);
- extension assets are copied, never symlinked, because the WebUI static server
  rejects symlinks escaping the extension bundle;
- the installer merges its entry into `extensions.json` without deleting or
  reordering sibling extensions such as Office 3D.

## Configuration

No config file is required. Behavior can be influenced via environment
variables:

| Variable                          | Default      | Effect |
|-----------------------------------|--------------|--------|
| `HERMES_PROFILE`                  | `default`    | Fallback for active-profile detection when Hermes's resolver is unavailable. |
| `HERMES_HOME`                     | `~/.hermes`  | Forwarded to the child so profiles resolve consistently. |
| `HERMES_DELEGATE_PROFILE_TIMEOUT` | unset        | Global default timeout (seconds) when no `timeout` arg is passed. |
| `HERMES_DELEGATE_PROFILE_DISABLE` | unset        | Set to `1` inside spawned children to prevent recursive delegation. |

## Development

```bash
git clone https://github.com/rodrigogs/hermes-smart-router.git
cd hermes-smart-router
pip install -e ".[dev]"
```

Project layout:

```
hermes-smart-router/
├── plugin.yaml          # manifest (name, version, provides_tools, provides_hooks)
├── __init__.py          # register() — registers the tool + post_tool_call hook
├── router.example.yaml  # shipped default policy (tracked)
├── router.yaml          # live policy — gitignored, seeded from the example
├── pyproject.toml       # project metadata + pytest/coverage config
├── README.md            # this file
├── LICENSE              # MIT
├── router/              # capability router library
│   ├── adapter.py       # Stage 0 → Stage 1 → delegate_profile bridge
│   ├── signals.py       # task signal extraction
│   ├── rules.py         # rule matching engine
│   ├── classify.py      # LLM difficulty classifier
│   ├── blocklist.py     # manual ban + auto-breaker state machine
│   ├── breaker.py       # circuit breaker state
│   ├── cache.py         # exact-hash cache + session model-floor pin
│   ├── capabilities.py  # model/price registry, chain shaping, price windows
│   ├── decision_log.py  # in-memory decision recorder (closed cause set)
│   ├── durable_decision_log.py  # routes.jsonl append + rotation
│   ├── threshold.py     # compaction-threshold curve
│   ├── price_watch.py   # supplier pricing-page change detector
│   ├── price_watch_runner.py    # its cron edge (the only network call)
│   ├── fixtures/        # classifier tier anchors
│   ├── service.py       # shared read-only policy view for web surfaces
│   ├── one_sidecar.py   # Hermes One loopback token-v1 sidecar
│   └── cli.py           # CLI: explain, chain, lint, blocklist, log
├── dashboard/           # Hermes Dashboard panel (React + Plugin SDK)
│   ├── manifest.json
│   ├── plugin_api.py    # FastAPI routes
│   └── dist/index.js    # bundled panel code
├── webui_extension/     # Hermes One assets (manifest + panel JS/CSS)
├── scripts/             # installers, transactional stack updater, stale-check poller, smoke
├── docs/                # ARCHITECTURE.md, design specs, operations runbooks
├── systemd/             # sidecar unit + stale-check service/timer templates
├── PRODUCT.md           # durable product record
└── tests/
    ├── test_delegate_profile.py
    ├── test_router_integration.py
    └── router/          # per-module unit tests
```

### Tests

```bash
# All tests:
pytest tests/ -v

# Router tests only:
pytest tests/ -v --ignore=tests/test_delegate_profile.py

# With coverage:
pytest --cov --cov-report=term tests/ -v
```

`register(ctx)` is called by Hermes at startup. It resolves the active profile
once, builds the handler via `_make_handler(current_profile, dispatch_delegate)`
(which captures a `dispatch_delegate` closure over `ctx.dispatch_tool`), and
registers the tool schema + `post_tool_call` hook.

## License

MIT

## Routing policy (`router.yaml`)

`router.yaml` is the **live** policy file. It is intentionally *not* tracked by
git: tuning tiers or rules for your own providers would otherwise leave the
checkout permanently dirty and conflict on every pull.

- `router.example.yaml` — tracked, the shipped default.
- `router.yaml` — gitignored, seeded from the example on first load, then yours.

To start over, delete `router.yaml` and it will be re-seeded.

## What leaves the machine (session-alignment judge)

The alignment judge (`router/alignment_judge.py`, guarded by `router/alignment_guard.py`)
sends a compacted, redacted transcript package to an LLM provider. It contains the card
title/body, the first user message, recent messages and truncated tool results (file
contents and command output may appear). Secrets matching the redaction patterns, images
and base64 blobs are removed before sending; redaction is best effort, not a guarantee.

- `privacy.allow_providers` is an allowlist. A provider not on it is never contacted; an
  empty or missing list refuses every send (verdict `continue`, `failed_open: privacy_refused`).
- `max_judge_calls_per_day` (default 40, UTC day) and `max_judge_input_tokens` (per call)
  are hard caps; `max_judge_tokens_per_day` is optional. Hitting one fails open to
  `continue` (`call_cap` / `token_cap`). The call counter is charged before the call.
- Copilot judge calls are premium requests. See docs/research/2026-10-copilot-premium-budget.md.
- Alerts carry card id, verdict and reasons, never the transcript.
