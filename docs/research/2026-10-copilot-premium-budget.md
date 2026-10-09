# Copilot premium-request budget (card I12, t_b030e295)

Status: instrumentation shipped; the one-week shadow has NOT elapsed. Nothing here touches the live `router.yaml`.

## 1. Quota and multipliers (sources, read 2026-10-09)

- Allowance (legacy request-based plans only): Pro 300/month, Pro+ 1500/month, extra requests $0.04 each, counters reset on the 1st at 00:00 UTC.
  https://docs.github.com/en/copilot/concepts/billing/copilot-requests
- Billing unit: one premium request per user prompt times the model multiplier; tool calls inside an agentic turn do not bill.
- Multipliers: https://docs.github.com/en/copilot/reference/copilot-billing/request-based-billing-legacy/model-multipliers-for-annual-plans
  Published for gpt-5.4 = 6. NOT published for claude-opus-5.5, claude-sonnet-5.5, claude-haiku-5.5, gpt-6.1-sol. No value is invented; the router counts them at 1.0 and lists them under `unpublished_multiplier`.
- Caveat: since 2026-06-01 usage-based billing has no multipliers or request allowance. Which plan this install is on is NOT verified, so the numbers below are valid only if it is a legacy annual plan.
- x-initiator: Hermes sends `x-initiator: user` on the first call of each user turn (docs/research/2026-10-hermes-integration.md §1), so every kanban worker and every `delegate_profile` child costs >= 1 premium request.

## 2. What the router now models

- `router/premium_budget.py`: 1 request per worker or delegated child (source `kanban` or `delegate`; older traces without `source` but with a `task_id` count as kanban; chat is excluded), times the multiplier, grouped by tier.
- `premium_requests` = turns whose head elo is Copilot (billed if the primary serves). `exposure_requests` = turns that list a Copilot hop in the fallback chain (upper bound, billed only on fall-through).
- Read it at `GET /premium-budget?days=7` on the sidecar, or `python scripts/premium_budget_report.py [days]`.
- Tests: tests/router/test_premium_budget.py (3 pass).

## 3. Shadow measurement (so far)

Trace window used: last 30 days, 159 decisions (`python scripts/premium_budget_report.py 30`).

- Copilot is not the head elo on any tier today, so measured premium requests = 0.
- Exposure (a fall-through to Copilot gpt-5.4 at x6): glm-5.3 tier 102, glm-5.3-flash tier 174 (17 and 104 turns). Worst case if every one fell through: 276 requests, i.e. 92% of the Pro allowance, 18% of Pro+.
- Tiers served by openai-codex only (gpt-5.6-terra 35 turns, gpt-5.6-sol 3): no Copilot hop.
- Caveat: this is a trace replay, not a week of live shadow. Only the dispatches that exist are counted; fall-through rates are not measured (`attempts.jsonl` would be needed).

## 4. Tier-map proposal (grounded in the numbers above)

1. Keep Copilot out of any tier head. One gpt-5.4 turn costs 6 requests; 50 workers a month would use the whole Pro allowance.
2. Keep Copilot gpt-5.4 only as a last hop, as in the current chain. Exposure is the worst case, and a fall-through happens only when three earlier hops fail.
3. If the classifier or alignment judge moves to Copilot (docs/router.proposed.yaml, session-alignment-hook), each call is a premium request at an unpublished multiplier. Cap it (`max_judge_calls_per_day`) before enabling it.
4. Before making Copilot a primary anywhere, close the open items: plan type, claude-* multipliers (read the Copilot usage page in the account UI), and a real week of shadow.

## 5. Remaining to finish the card

- Let the sidecar run for 7 days, then run `python scripts/premium_budget_report.py 7` and paste the table here.
- Confirm the plan (legacy annual vs usage-based) and the multipliers for claude-opus-5.5/sonnet-5.5/haiku-5.5 and gpt-6.1-sol; put them in `MULTIPLIERS`.

## 6. Judge multiplier (card F12, t_be596f96; checked 2026-10-09)

Result: NOT MEASURABLE from here, and not published. The GitHub Docs table (annual legacy plans) lists
Haiku 4.5 0.33, Sonnet 4.6 9, Opus 4.8 27, GPT-5.4 6, GPT-5.5 57; it has no row for claude-opus-5.5 or
gpt-6.1-sol. Usage-based plans (since 2026-06-01) have no multipliers at all. Reading the real value needs
the account's Copilot usage page before/after one judge call; no such access exists in this session.
Until then the judge is counted at 1.0 (unpublished) and the ceiling is `max_judge_calls_per_day`, which
bounds cost at N x multiplier per day: at 40 calls, an Opus-class multiplier of 27 would be 1080 requests/day.
Enable a Copilot judge only after reading the multiplier off the usage page and sizing the cap against it.
