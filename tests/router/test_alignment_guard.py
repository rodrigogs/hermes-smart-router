from types import SimpleNamespace

from router.alignment_guard import DailyBudget, allowed_hops, guarded_judge
from router.alignment_judge import Hop
from router.breaker import BreakerState

CHAIN = [Hop("claude-opus-5.5", "copilot"), Hop("gpt-6.1-sol", "openai-codex"),
         Hop("x", "other")]
OK = {"verdict": "continue", "confidence": 0.5, "reasons": [], "steer_message": "", "evidence": []}
DAY = 1_791_000_000.0


class Llm:
    def __init__(self):
        self.calls = []

    def complete_structured(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(parsed=OK)


def run(llm, budget, pkg="pkg", allow=("copilot", "openai-codex"), now=DAY, **kw):
    return guarded_judge(
        SimpleNamespace(llm=llm), pkg, chain=CHAIN, allow_providers=allow, budget=budget,
        now=now, worker_provider="zai", breaker=BreakerState({"threshold": 1}), **kw)


def test_allowed_hops_filters_and_empty_refuses_all():
    assert [h.provider for h in allowed_hops(CHAIN, ["copilot"])] == ["copilot"]
    assert allowed_hops(CHAIN, []) == [] and allowed_hops(CHAIN, None) == []


def test_disallowed_provider_is_refused_without_sending():
    llm = Llm()
    v = run(llm, DailyBudget(), allow=["anthropic"])
    assert v.failed_open == "privacy_refused" and v.verdict == "continue"
    assert llm.calls == []


def test_only_allowed_providers_are_called():
    llm = Llm()
    v = run(llm, DailyBudget(), allow=["openai-codex"])
    assert v.provider == "openai-codex" and llm.calls[0]["provider"] == "openai-codex"


def test_call_cap_per_day_and_resets_next_day(tmp_path):
    b = DailyBudget(tmp_path / "s" / "b.json")
    llm = Llm()
    assert run(llm, b, max_judge_calls_per_day=2).failed_open == ""
    assert run(llm, DailyBudget(tmp_path / "s" / "b.json"), max_judge_calls_per_day=2).failed_open == ""
    assert run(llm, b, max_judge_calls_per_day=2).failed_open == "call_cap"
    assert len(llm.calls) == 2
    assert run(llm, b, now=DAY + 86400, max_judge_calls_per_day=2).failed_open == ""


def test_per_call_token_cap_does_not_charge():
    b, llm = DailyBudget(), Llm()
    assert run(llm, b, pkg="x" * 4000, max_judge_input_tokens=10).failed_open == "token_cap"
    assert llm.calls == [] and b.usage(DAY)["calls"] == 0


def test_daily_token_cap():
    b, llm = DailyBudget(), Llm()
    assert run(llm, b, pkg="x" * 400, max_judge_tokens_per_day=150).failed_open == ""
    assert run(llm, b, pkg="x" * 400, max_judge_tokens_per_day=150).failed_open == "token_cap"


def test_corrupt_or_unwritable_state_is_tolerated(tmp_path):
    p = tmp_path / "b.json"
    p.write_text("{not json")
    assert DailyBudget(p).usage(DAY)["calls"] == 0
    blocked = tmp_path / "file"
    blocked.write_text("")
    DailyBudget(blocked / "sub" / "b.json").charge(DAY, 5)  # mkdir fails, no raise
