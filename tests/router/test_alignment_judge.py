import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from router import alignment_judge as aj
from router.alignment_judge import Hop, judge
from router.breaker import BreakerState
from router.model_lint import load_cache

CACHE = Path("/home/rodrigo/.hermes/provider_models_cache.json")
CHAIN = [Hop("claude-opus-5.5", "copilot"), Hop("gpt-6.1-sol", "copilot"),
         Hop("gpt-6.1-sol", "openai-codex")]
GOOD = {"verdict": "adjust", "confidence": 0.7, "reasons": ["r"], "steer_message": "go back",
        "evidence": [{"message_index": 1, "quote": "q" * 300}, "junk", {"message_index": "x"}]}


class Llm:
    def __init__(self, *outs):
        self.outs, self.calls = list(outs), []

    def complete_structured(self, **kw):
        self.calls.append(kw)
        out = self.outs.pop(0)
        if isinstance(out, BaseException):
            raise out
        return SimpleNamespace(parsed=out)


def run(llm, worker="zai", chain=CHAIN, br=None, **kw):
    br = br or BreakerState({"threshold": 1})
    return judge(SimpleNamespace(llm=llm), "pkg", chain=chain, worker_provider=worker,
                 breaker=br, now=100.0, **kw), br


def test_success_uses_first_hop_and_schema():
    llm = Llm(GOOD)
    v, _ = run(llm)
    assert (v.verdict, v.provider, v.model, v.failed_open) == ("adjust", "copilot", "claude-opus-5.5", "")
    assert v.evidence == [{"message_index": 1, "quote": "q" * 200}]
    c = llm.calls[0]
    assert c["json_schema"] is aj.JUDGE_SCHEMA and c["provider"] == "copilot"
    assert c["purpose"] == aj.PURPOSE and c["timeout"] == 90


def test_skips_worker_provider_and_can_be_disabled():
    llm = Llm(GOOD)
    v, _ = run(llm, worker="copilot")
    assert v.provider == "openai-codex"
    v, _ = run(Llm(GOOD), worker="copilot", require_distinct=False)
    assert v.provider == "copilot"
    v, _ = run(Llm(), worker="copilot", chain=CHAIN[:2])
    assert v.failed_open == "no_hop"
    v, _ = run(Llm(GOOD), worker=None)
    assert v.provider == "copilot"


def test_trust_gate_refusal_fails_open_without_trying_more_hops():
    llm = Llm(PermissionError("no grant"), GOOD)
    v, br = run(llm)
    assert v.verdict == "continue" and v.failed_open == "trust_gate" and len(llm.calls) == 1
    assert not br.would_block("copilot/claude-opus-5.5", 100.0)


@pytest.mark.parametrize("exc,kind", [(TimeoutError("t"), "timeout"), (RuntimeError("x"), "error"),
                                      (json.JSONDecodeError("m", "d", 0), "parse")])
def test_failures_trip_breaker_and_fall_to_next_hop(exc, kind):
    llm = Llm(exc, GOOD)
    v, br = run(llm)
    assert v.verdict == "adjust" and v.provider == "copilot" and v.model == "gpt-6.1-sol"
    assert br.would_block("copilot/claude-opus-5.5", 100.0)
    v2, _ = run(Llm(GOOD), br=br)  # open hop skipped without a call
    assert v2.model == "gpt-6.1-sol"


def test_all_fail_reports_last_kind_and_open_breaker_reports_breaker_open():
    v, br = run(Llm(TimeoutError(), TimeoutError()), chain=CHAIN[:2], worker="zai")
    assert v.verdict == "continue" and v.failed_open == "timeout"
    llm = Llm()
    v, _ = run(llm, chain=CHAIN[:2], br=br)
    assert v.failed_open == "breaker_open" and not llm.calls


@pytest.mark.parametrize("bad", [None, "x", {"verdict": "nope"}, {"verdict": "block", "confidence": "a"},
                                 {"verdict": "block", "confidence": 2},
                                 {"verdict": "block", "confidence": 1, "reasons": "r"},
                                 {"verdict": "block", "confidence": 1, "steer_message": 3}])
def test_unparseable_result_fails_open(bad):
    v, br = run(Llm(bad), chain=CHAIN[:1])
    assert v.verdict == "continue" and v.failed_open == "parse"
    assert br.would_block("copilot/claude-opus-5.5", 100.0)


def test_parse_chain_skips_malformed():
    assert aj.parse_chain([{"model": "m", "provider": "p"}, {"model": "m"}, 3]) == [Hop("m", "p")]
    assert aj.parse_chain(None) == []


def test_validate_chain_against_cache_and_usable():
    cache = {"copilot": {"a"}}
    chain = [Hop("a", "copilot"), Hop("b", "copilot"), Hop("a", "nope")]
    assert len(aj.validate_chain(chain, cache)) == 2
    assert aj.usable_chain(chain, cache) == [Hop("a", "copilot")]


@pytest.mark.skipif(not CACHE.exists(), reason="no live models cache")
def test_shipped_chain_exists_in_live_cache():
    assert aj.validate_chain(CHAIN, load_cache(CACHE)) == []
