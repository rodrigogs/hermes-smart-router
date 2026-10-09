# F1 spike: session-alignment hook observability in a real kanban worker (t_ec8d3a31)

Follow-up to docs/research/2026-10-session-alignment-hook.md (§1, §9, §10-F1).
Measured 2026-10-09 against the live runtime (`hermes-runtime`, same checkout the gateway serves).

## Method

- Ephemeral profile `f1spike` (created with `hermes profile create --no-skills`, deleted afterwards) holding
  one throwaway plugin, `f1probe`, enabled only in that profile. Source kept for reproduction in
  `docs/research/f1-probe/probe_plugin.py.txt` (renamed .txt so it is not collected or loaded).
- Real workers: cards on the `routing-smoke` board, assignee `f1spike`, spawned by the normal dispatcher
  (`hermes -p f1spike --cli --accept-hooks chat -q ... --model claude-sonnet-5.5 --provider copilot`).
  Each card ran 3 `terminal` calls and `kanban_complete`.
- Nothing was added to the served plugin (`~/.hermes/plugins`, router plugin untouched). No gateway restart.
  The `coder` profile config and `auth.json` were not modified (the spike profile symlinked auth.json read-only
  and was deleted; the link target is intact).
- Caveat: the first attempt with the profile default (zai glm-5.3-flash) died on a 429 weekly quota, and
  openai-codex had no OAuth token in the cloned profile, so every measured run used copilot / claude-sonnet-5.5.
  Hook payloads are provider-independent, but `provider`/`model` values below are copilot's.

## 1. Real kwargs of pre_api_request / post_api_request (kanban worker)

pre_api_request (24 keys):
`api_call_count, api_mode, api_request_id, approx_input_tokens, base_url, conversation_history, max_tokens,
message_count, middleware_trace, model, platform, provider, request, request_char_count, request_messages,
retry_count, session_id, started_at, system_prompt, task_id, telemetry_schema_version, tool_count, turn_id,
user_message`

post_api_request (24 keys):
`api_call_count, api_duration, api_mode, api_request_id, assistant_content_chars, assistant_message, assistant_tool_call_count,
base_url, ended_at, finish_reason, first_chunk_at, message_count, moa_references, model, platform, provider, response,
response_model, session_id, started_at, task_id, telemetry_schema_version, turn_id, usage`

Observed values (call 1, kanban worker):
- `platform="cli"` (NOT "kanban"; the worker is a `hermes chat -q` process). `provider="copilot"`, `api_mode="chat_completions"`,
  `telemetry_schema_version="hermes.observer.v1"`, `tool_count=34`, `approx_input_tokens=3741`, `retry_count=0`.
- `task_id` is the SESSION id (`20261009_001629_c5d83f`), not the kanban task id. The kanban id must come from
  env `HERMES_KANBAN_TASK` (also `HERMES_KANBAN_RUN_ID`, `HERMES_SESSION_SOURCE=kanban`, all present inside the hook).
  Same for the `pre_tool_call` `task_id` kwarg. §10-F7/F8 must key on env, not on this kwarg.
- `session_id` == `task_id` == `20261009_001629_c5d83f`. `turn_id` = `<session>:<session>:<hash>`; `api_request_id` = `<turn_id>:api:<n>`.
- `user_message` = the worker prompt ("work kanban task t_...").
- `request` is a sanitised dict, truncated (`_truncated: true`, 50 000-char preview).
- `usage` (post) = `{input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens, request_count, prompt_tokens, total_tokens}`.
- `finish_reason`: `tool_calls` for calls 1-5, `stop` for the last.

Differences vs the §1.2 text: the payload also carries `turn_id`, `api_request_id`, `request`, `system_prompt`, `request_messages`,
`telemetry_schema_version`; `post_api_request` does NOT carry `conversation_history` (only pre does).

## 2. api_call_count progression

Clean run (observe): 1, 2, 3, 4, 5, 6, strictly +1 per provider call, pre and post agree, `retry_count=0`.
Six calls = kanban_show, terminal x3, kanban_complete, final text (`finish_reason=stop`). So the final "stop" turn counts.
Every scenario run showed the same +1 progression; a call blocked by `pre_tool_call` still costs an api call (the model's next turn).
The counter starts at 1 per `run_conversation` (one per classic worker). Not verified under goal-mode or after a retry within a run.

## 3. max_iterations: ABSENT

Not in the pre or post kwargs (searched the sorted key list; no `max_iterations`, `iteration_budget_*`, or `budget_*`).
The denominator must be resolved outside the payload. In the worker env `HERMES_MAX_ITERATIONS=150` WAS set
(equals `agent.max_turns: 150` in the coder profile config), so the env route in §1.2 option 1 works for kanban workers.
Note `config.yaml` also has `delegation.max_iterations: 250`, a different key; do not conflate.
F2 (upstream adding the kwargs) is still justified, since IterationBudget (500 parent) is the real stop condition and is not env-visible.

## 4. The four items that were NOT VERIFIED

### 4.1 One-shot steer via pre_tool_call: WORKS
- Probe armed at `api_call_count==2`; next non-`kanban_*` tool call (`terminal`) returned
  `{"action":"block","message":"[alignment steer] STOP: ..."}`; the following terminal calls ran normally (block fired exactly once).
- The model saw the message as the tool result and surfaced it in its final summary ("the first 'echo one' returned an error-field message [alignment steer] STO...").
  Message delivery to the model is confirmed. Whether it obeys the instruction text was not measured (it re-ran the call, which was the instruction).
- Cost as predicted: one wasted tool call, one extra api call. `kanban_*` tools were excluded by the probe's own filter and passed untouched.
- `pre_tool_call` kwargs: `api_request_id, args, middleware_trace, session_id, task_id, telemetry_schema_version, tool_call_id, tool_name, turn_id`.

### 4.2 Block externo vs violação de protocolo: NOT a protocol violation, but it does not stop the worker by itself
`kanban_db.block_task(conn, tid, reason=..., kind="needs_input", expected_run_id=<HERMES_KANBAN_RUN_ID>)` called from the hook inside the worker returned `True`.
Three variants, all ended `blocked`, none produced a `protocol_violation`/`crashed` event (event list: created, claimed, spawned, heartbeat, blocked; run outcome `blocked`):
1. probe vetoes non-kanban tools after the block, allows `kanban_*`: worker called `kanban_block` itself (allowed), exited. Final state `blocked`, reason = the probe's reason.
2. probe vetoes ALL tools including `kanban_*`: `kanban_block` was vetoed too; worker still stopped on its own (api call 5 `finish_reason=stop`). State `blocked`.
3. same veto, but the prompt forbade `kanban_*` use after a refusal: worker replied in plain text and exited rc=0 without any terminal kanban call. State stayed `blocked`, run `blocked`, no violation.
Reason: the violation path in `kanban_db` (around line 9010) only triggers for a task still `running` when the pid exits cleanly. After an external block the row is no longer `running`.
Implications for §1.3/F8:
- Safe to call from the worker process with the CAS `expected_run_id`; the dispatcher treats the exit as normal.
- The block does NOT interrupt the worker; it keeps going until the model stops. The "veto all later tools" trick is what ends it (took 2 more api calls here, 5 to 6 total). Without the veto the worker would keep working and could still call `kanban_complete`; not tested whether completing a `blocked` task succeeds (not measured: assume unknown).
- The `blocked` event carried `recurrences: 1` (feeds the §8 triage circuit).
- Only the `needs_input` kind was tested.

### 4.3 send_message headless Telegram/WhatsApp: the tool route DOES NOT EXIST; the CLI route works for Telegram only
- `ctx.dispatch_tool("send_message", ...)` from the hook inside a kanban worker returned `{"error": "Unknown tool: send_message"}` for both targets (twice, 0.0 s). Cause, confirmed in source:
  `tools/send_message_tool.py:2489` says `send_message` is intentionally NOT registered as an agent-callable tool.
  **§6 of the original report is wrong on this point and F9 cannot use `ctx.dispatch_tool("send_message")`.**
- `hermes send -t telegram "..." --json` (subprocess from the worker): `rc=0`, `success: true`, home channel `796660198`, message_id 6297, 5.15 s. A real test message reached the user's Telegram.
  (Exactly one Telegram message was sent in total: the first send run only hit the failing dispatch, the second produced message_id 6297.)
- `hermes send -t whatsapp ...`: `rc=1`, `"WhatsApp send failed: Cannot connect to host localhost:3000"`, 1.2 s. WhatsApp mode is `bot` and needs the local bridge on port 3000, which is not listening on this host right now (`ss` shows nothing on :3000). So WhatsApp from a headless worker is NOT proven either way; it depends on the bridge being up, and the CLI has no gateway-less path for it.
- Recommended F9 channel: subprocess `hermes send` (works with no gateway for bot-token platforms) or the outbox-plus-gateway fallback of §6. Budget ~5 s.

### 4.4 ctx.inject_message in a worker: DOES NOT WORK
`ctx.inject_message(...)` called at `api_call_count==1` returned `False` and `ctx._manager._cli_ref` was `None`.
Source: `inject_message` needs the CLI ref (`hermes_cli/plugins.py:2084`); the worker is a quiet `chat -q` run, so it is never set. The message never reached the model (final summary was plain "inject ok", no INJECTED-OK).
Conclusion: not usable in workers. Steering stays on §4.1 (one-shot veto) or an upstream `steer`.

## 5. Corrections to carry into the plan
1. §6/F9: drop `ctx.dispatch_tool("send_message")`; use `hermes send` subprocess or the outbox.
2. §1.2: `task_id` in hook kwargs is the session id. Use `HERMES_KANBAN_TASK` / `HERMES_KANBAN_RUN_ID` from env.
3. §1.2: denominator from env `HERMES_MAX_ITERATIONS` is confirmed present in workers.
4. §1.3: external `block_task` does not trip protocol violation, but also does not stop the process.
5. §2: `platform` is `"cli"` for kanban workers; detect scope by `HERMES_KANBAN_TASK`/`HERMES_SESSION_SOURCE=kanban`, never by `platform`.

## 6. Cleanup / side effects
- Profile `f1spike` deleted; `routing-smoke` still holds the spike cards (t_ecf2e902, t_1aa4ea18, t_3ac2c725, t_9db7d3a2, t_1acdef71, t_0a0cd7d0, t_6aca4e8b, t_866b065a, t_0e889806); they are throwaway and may be archived. Two of the blocks above remain `blocked` there.
- An aborted `profile create --clone-all` (3.4 GB) and an empty `f1-spike` board were created and removed (board archived under `kanban/boards/_archived/`).
- Telegram received 1 test message marked "[F1 spike test, ignore]". No WhatsApp message was delivered.
- Raw hook logs were not committed (they contain the full system prompt); the probe plugin source is.
