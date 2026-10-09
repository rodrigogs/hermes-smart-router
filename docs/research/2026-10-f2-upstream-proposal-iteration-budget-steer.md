# F2: upstream proposal - iteration budget in pre_api_request + steer return from post_api_request

Task t_742ba9a5. Source: 2026-10-session-alignment-hook.md section 10-F2; evidence: 2026-10-session-alignment-hook-f1-spike.md.
STATUS: DRAFT. Not filed upstream, no local patch applied. Both need Rodrigo's approval.

Checked against hermes-agent checkout at 512d699260 (~/.hermes/hermes-agent).

## Code facts (verified by reading)
- agent/conversation_loop.py:3359 fires `pre_api_request` with api_call_count, retry_count, max_tokens ... but NOT max_iterations or budget.
- agent/conversation_loop.py:7204 fires `post_api_request`; the return value of `_invoke_hook` is discarded (the call is a bare statement inside try/except: pass).
- run_agent.py:4867-4870 already computes `api_call_count`, `max_iterations`, `iteration_budget.used`, `iteration_budget.max_total` for the activity snapshot, so the values exist on the agent.
- `AIAgent.steer(text) -> bool` (run_agent.py:3912) is thread-safe and appends text to the last tool result at the next drain point.
- F1 measured: max_iterations absent from hook kwargs (only env HERMES_MAX_ITERATIONS=150 in workers); api_call_count 1..N, +1 per call; ctx.inject_message returns False in workers; one-shot steer via pre_tool_call veto works but is a hack (it blocks a real tool call).

---

## Issue / PR text (English, for the upstream repo)

Title: Hooks: expose iteration budget on pre_api_request and honour a `{"steer": str}` return from post_api_request

### Problem
Plugins that supervise a running agent (drift/alignment monitors, cost guards) need two things the hook API does not give them:

1. The remaining iteration budget. `pre_api_request` carries `api_call_count` but not the ceiling, so a plugin cannot tell "call 40 of 150" from "call 40 of 40". Workarounds (reading HERMES_MAX_ITERATIONS from the env) are unavailable for non-env configuration and for subagents with their own budget.
2. A way to nudge the model. `ctx.inject_message` returns False outside the interactive CLI (no `_cli_ref`), e.g. in kanban/cron workers. The only workaround is vetoing a real tool call from `pre_tool_call` with a message, which fails a legitimate call to deliver text.

### Proposal
A. Add three additive kwargs to `pre_api_request` (hook payloads already evolve additively; narrow-signature callbacks are unaffected):
   - `max_iterations: int`   (agent.max_iterations)
   - `iteration_budget_used: int`  (agent.iteration_budget.used)
   - `iteration_budget_max: int`   (agent.iteration_budget.max_total)
   Same names optionally on `post_api_request`.

B. In `post_api_request`, inspect the non-None return values already collected by `invoke_hook`. For each value that is a dict with a non-empty string `"steer"`, call `agent.steer(value["steer"])`. Other return shapes are ignored (today all are ignored). Document this next to the existing `pre_llm_call` `{"context": ...}` and `pre_tool_call` `{"action": "block"}` return contracts in `invoke_hook`'s docstring and the hooks docs.

Sketch (conversation_loop.py, post_api_request site):

    _results = _invoke_hook("post_api_request", ...)
    for _r in _results or []:
        if isinstance(_r, dict) and isinstance(_r.get("steer"), str):
            agent.steer(_r["steer"])

### Semantics / safety
- steer() already concatenates multiple pending steers and drains once per tool batch, so several plugins cannot flood a single turn beyond their own text.
- Failure is isolated: the existing try/except stays; a bad return is ignored.
- Hook stays timeout-bounded (`plugins.hook_callback_timeout`); no new blocking path.
- Opt-in by construction: no plugin returns a steer today.

### Tests
- pre_api_request receives the three kwargs with values equal to the agent's.
- Narrow-signature callback still works.
- post_api_request returning {"steer": "x"} -> agent.steer called once with "x"; returning "x", {} , {"steer": ""}, {"steer": 3} -> not called.
- Two plugins returning steers -> both queued, drained together.

---

## Decision: local patch vs wait for upstream (RECOMMENDATION, needs Rodrigo's approval)

Constraint (memory): `hermes update` tracks origin/main of the fork; committed-but-unpushed patches are LOST on reset --hard. Plugin-only solutions survive; core patches do not unless pushed to the fork.

Recommendation: WAIT for upstream; do not patch core locally now. Reasons:
1. The alignment feature does not depend on it: F8 already specifies a fallback chain (steer upstream if landed, else one-shot pre_tool_call, else pre_llm_call context in goal mode). F1 proved the pre_tool_call channel works.
2. max_iterations is obtainable now from env HERMES_MAX_ITERATIONS in workers (F1), so the budget half has a working substitute.
3. A core patch is fragile under the update policy and adds a fork to maintain for a nicety.
Revisit trigger: if F11 shadow data shows the pre_tool_call veto degrades worker runs (failed legitimate calls), then push the patch to the fork (not just commit locally) and file the PR at the same time.

Next actions that need Rodrigo: (1) approve filing the issue above on the upstream repo (or the fork); (2) confirm the WAIT decision.
