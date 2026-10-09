"""Judge package assembler (session-alignment hook, F5).

Pure and deterministic: same inputs, same output. Builds the text the alignment
judge reads (transcript + kanban card + parents + comments) inside a token
budget, after redacting secrets and stripping images / base64.

Compaction policy: the first user message, assistant text and errors have
priority; tool results are truncated head+tail; when the budget is still
exceeded the oldest low-priority entries are dropped with an omission marker.
"""

from __future__ import annotations

import json
import re
from typing import Any

CHARS_PER_TOKEN = 4
TOOL_RESULT_CHARS = 600
ERROR_RESULT_CHARS = 2000
TEXT_CHARS = 4000
SECTION_CHARS = 3000
REDACTED = "[REDACTED]"
IMAGE_MARK = "[image removed]"
BASE64_MARK = "[base64 removed]"

_IMAGE_PART_TYPES = {"image", "image_url", "input_image"}

_SECRET_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)", re.S),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/\-]{16,}=*"),
    re.compile(r"(?<=://)[^/\s:@]+:[^/\s@]+(?=@)"),
]
_KV_SECRET = re.compile(
    r"""(?ix)
    (\b[\w\-]*(?:api[_\-]?key|secret|token|passw(?:or)?d|credential)[\w\-]*\b
     ["']?\s*[:=]\s*["']?)
    ([^\s"',;]{4,})
    """
)
_DATA_URI = re.compile(r"data:[\w/+.\-]+;base64,[A-Za-z0-9+/=_\-]+")
_BASE64_RUN = re.compile(r"[A-Za-z0-9+/_\-]{200,}={0,2}")


def estimate_tokens(text: str) -> int:
    return -(-len(text) // CHARS_PER_TOKEN)


def redact(text: str) -> str:
    for pat in _SECRET_PATTERNS:
        text = pat.sub(REDACTED, text)
    return _KV_SECRET.sub(lambda m: m.group(1) + REDACTED, text)


def strip_blobs(text: str) -> str:
    return _BASE64_RUN.sub(BASE64_MARK, _DATA_URI.sub(BASE64_MARK, text))


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    cut = len(text) - limit
    return f"{text[:head]}\n[... {cut} chars truncated ...]\n{text[len(text) - tail:]}"


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in _IMAGE_PART_TYPES:
                    parts.append(IMAGE_MARK)
                else:
                    parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return "\n".join(p for p in parts if p)
    return str(content)


def _clean(text: str, do_redact: bool) -> str:
    text = strip_blobs(text)
    return redact(text) if do_redact else text


_ERROR_RE = re.compile(r"(?i)\b(error|traceback|exception|failed|exit code [1-9])\b")


def _is_error(msg: dict, text: str) -> bool:
    return bool(msg.get("is_error") or msg.get("error") or _ERROR_RE.search(text[:2000]))


def _render_message(msg: dict, do_redact: bool) -> tuple[int, str]:
    """Returns (priority, rendered). Priority 1 = keep, 2 = droppable."""
    role = str(msg.get("role", "?"))
    text = _clean(_content_text(msg.get("content")), do_redact)
    if role == "tool":
        err = _is_error(msg, text)
        body = _clip(text, ERROR_RESULT_CHARS if err else TOOL_RESULT_CHARS)
        return (1 if err else 2), f"[tool{' ERROR' if err else ''}] {body}"
    calls = []
    for call in msg.get("tool_calls") or []:
        fn = call.get("function", {}) if isinstance(call, dict) else {}
        args = fn.get("arguments", "")
        if not isinstance(args, str):
            args = json.dumps(args, sort_keys=True, default=str)
        calls.append(f"{fn.get('name', '?')}({_clip(_clean(args, do_redact), 200)})")
    body = _clip(text, TEXT_CHARS)
    if calls:
        body = (body + "\n" if body else "") + "calls: " + "; ".join(calls)
    return (1 if role == "assistant" else 2), f"[{role}] {body}"


def _section(title: str, value: Any, do_redact: bool) -> str:
    if value in (None, "", [], {}):
        return ""
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return f"## {title}\n{_clip(_clean(value, do_redact), SECTION_CHARS)}\n"


def build_package(
    messages: list[dict],
    *,
    card: Any = None,
    parents: Any = None,
    comments: Any = None,
    max_input_tokens: int = 120000,
    redact_secrets: bool = True,
) -> dict:
    """Assemble the judge input within ``max_input_tokens``.

    Returns {"text", "tokens", "tokens_before", "dropped"}; ``tokens_before`` is
    the size of the uncompacted (but redacted) package.
    """
    head = "".join(
        _section(t, v, redact_secrets)
        for t, v in (("Card", card), ("Parents", parents), ("Comments", comments))
    )
    entries = [_render_message(m, redact_secrets) for m in messages if isinstance(m, dict)]
    first_user = next(
        (i for i, m in enumerate(m for m in messages if isinstance(m, dict))
         if m.get("role") == "user"),
        None,
    )
    full = head + "\n".join(e[1] for e in entries)
    tokens_before = estimate_tokens(full)

    limit_chars = max_input_tokens * CHARS_PER_TOKEN
    budget = max_input_tokens - estimate_tokens(head) - 16
    while True:
        text, dropped = _compose(head, entries, first_user, budget)
        if len(text) <= limit_chars or budget <= 0:
            break
        budget -= max(1, estimate_tokens(text) - max_input_tokens)

    # Hard guarantee: the first-user force path or oversized head may still overflow.
    if len(text) > limit_chars:
        text = text[:limit_chars]
    return {
        "text": text,
        "tokens": estimate_tokens(text),
        "tokens_before": tokens_before,
        "dropped": dropped,
    }


def _compose(head, entries, first_user, budget):
    keep: set[int] = set()
    used = 0

    def take(i: int, force: bool = False) -> None:
        nonlocal used
        cost = estimate_tokens(entries[i][1]) + 1
        if force or used + cost <= budget:
            keep.add(i)
            used += cost

    if first_user is not None:
        take(first_user, force=True)
    for prio in (1, 2):
        for i in range(len(entries) - 1, -1, -1):
            if entries[i][0] == prio and i not in keep:
                take(i)

    lines: list[str] = []
    dropped = 0
    gap = 0
    for i, (_, rendered) in enumerate(entries):
        if i in keep:
            if gap:
                lines.append(f"[... {gap} messages omitted ...]")
                gap = 0
            lines.append(rendered)
        else:
            gap += 1
            dropped += 1
    if gap:
        lines.append(f"[... {gap} messages omitted ...]")
    return head + "## Transcript\n" + "\n".join(lines), dropped
