"""Privacy and cost guards for the alignment judge (F12, §5/§10 of the research note).

Wraps ``alignment_judge.judge`` so nothing reaches a provider outside
``privacy.allow_providers`` and the judge cannot exceed ``max_judge_calls_per_day`` or
``max_judge_input_tokens`` (per call) / ``max_judge_tokens_per_day`` (optional).
Refusals return ``continue`` with ``failed_open`` set, like every other judge failure.

An empty/absent allowlist refuses everything: fail closed on privacy.
The day counter is a small JSON file keyed by UTC date; the caller passes ``now``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

try:
    from .alignment_judge import Hop, Verdict, judge
    from .alignment_package import estimate_tokens
except ImportError:  # pragma: no cover - flat harness
    from router.alignment_judge import Hop, Verdict, judge
    from router.alignment_package import estimate_tokens


def allowed_hops(chain: Sequence[Hop], allow_providers: Optional[Sequence[str]]) -> list:
    """Hops whose provider is in the allowlist; none when the list is empty."""
    allow = {str(p) for p in (allow_providers or [])}
    return [h for h in chain if h.provider in allow]


def _day(now: float) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(now))


class DailyBudget:
    """Calls and input tokens spent per UTC day, persisted to ``path`` (or memory)."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self._mem: Dict[str, Any] = {}

    def _load(self, now: float) -> Dict[str, Any]:
        data: Dict[str, Any] = dict(self._mem)
        if self.path and self.path.exists():
            try:
                data = json.loads(self.path.read_text())
            except (OSError, ValueError):
                data = {}
        if not isinstance(data, dict) or data.get("day") != _day(now):
            return {"day": _day(now), "calls": 0, "tokens": 0}
        return {"day": data["day"], "calls": int(data.get("calls", 0)),
                "tokens": int(data.get("tokens", 0))}

    def usage(self, now: float) -> Dict[str, Any]:
        return self._load(now)

    def charge(self, now: float, tokens: int) -> None:
        st = self._load(now)
        st["calls"] += 1
        st["tokens"] += max(0, int(tokens))
        self._mem = st
        if self.path:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(".tmp")
                tmp.write_text(json.dumps(st))
                os.replace(tmp, self.path)
            except OSError:  # a counter that cannot persist must not break the worker
                pass


def guarded_judge(
    ctx: Any,
    package_text: str,
    *,
    chain: Sequence[Hop],
    allow_providers: Optional[Sequence[str]],
    budget: DailyBudget,
    now: float,
    max_judge_calls_per_day: int = 40,
    max_judge_input_tokens: int = 100_000,
    max_judge_tokens_per_day: Optional[int] = None,
    **judge_kwargs: Any,
) -> Verdict:
    """``judge`` behind the privacy allowlist and the call/token ceilings."""
    hops = allowed_hops(chain, allow_providers)
    if not hops:
        return Verdict(failed_open="privacy_refused")
    tokens = estimate_tokens(package_text)
    if tokens > max_judge_input_tokens:
        return Verdict(failed_open="token_cap")
    used = budget.usage(now)
    if used["calls"] >= max_judge_calls_per_day:
        return Verdict(failed_open="call_cap")
    if max_judge_tokens_per_day is not None and used["tokens"] + tokens > max_judge_tokens_per_day:
        return Verdict(failed_open="token_cap")
    # Charged before the call: a failed attempt still costs a premium request.
    budget.charge(now, tokens)
    return judge(ctx, package_text, chain=hops, **judge_kwargs, now=now)
