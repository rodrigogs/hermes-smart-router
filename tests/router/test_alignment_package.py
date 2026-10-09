import base64

from router.alignment_package import (
    build_package,
    estimate_tokens,
    redact,
    strip_blobs,
)

SECRETS = [
    "sk-abcdefghijklmnop1234567890",
    "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3",
    "github_pat_" + "A" * 30,
    "xoxb-1234567890-abcdefghij",
    "AKIAABCDEFGHIJKLMNOP",
    "AIza" + "x" * 35,
    "eyJhbGciOiJIUzI1.eyJzdWIiOiIxMjM0.SflKxwRJSMeKKF2QT4",
    "Bearer abcdefghijklmnopqrstuvwxyz0123",
    "hunter2hunter2",
    "s3cr3tpassw0rd",
    "-----BEGIN RSA PRIVATE KEY-----\nMIIEabc\n-----END RSA PRIVATE KEY-----",
    "supersecretpw",
]


def test_redaction_removes_every_seeded_secret_everywhere():
    msgs = [
        {"role": "user", "content": f"use key {SECRETS[0]} and {SECRETS[1]}"},
        {
            "role": "assistant",
            "content": f"{SECRETS[2]} {SECRETS[3]} {SECRETS[4]}",
            "tool_calls": [
                {"function": {"name": "run", "arguments": f'{{"api_key": "{SECRETS[8]}"}}'}},
                {"function": {"name": "dict", "arguments": {"password": SECRETS[9]}}},
                "junk",
            ],
        },
        {"role": "tool", "content": f"{SECRETS[5]} {SECRETS[6]} {SECRETS[7]} {SECRETS[10]}"},
        {"role": "tool", "content": "db at postgres://admin:supersecretpw@host/db"},
    ]
    out = build_package(
        msgs,
        card=f"token: {SECRETS[0]}",
        parents=[{"summary": SECRETS[1]}],
        comments=f"password={SECRETS[9]}",
    )["text"]
    for s in SECRETS:
        for frag in s.split("\n"):
            if len(frag) > 12:
                assert frag not in out, frag
    assert "[REDACTED]" in out
    assert "MIIEabc" not in out


def test_redact_off_keeps_text():
    out = build_package(
        [{"role": "user", "content": SECRETS[0]}], redact_secrets=False
    )["text"]
    assert SECRETS[0] in out


def test_redact_and_strip_helpers():
    assert redact("nothing here") == "nothing here"
    blob = base64.b64encode(b"x" * 600).decode()
    assert "base64 removed" in strip_blobs(f"data:image/png;base64,{blob}")
    assert "base64 removed" in strip_blobs(blob)


def test_images_and_base64_removed():
    blob = base64.b64encode(b"y" * 900).decode()
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{blob}"}},
                "plain",
                {"type": "other"},
            ],
        },
        {"role": "tool", "content": blob},
        {"role": "tool", "content": None},
        {"role": "assistant", "content": 42},
    ]
    out = build_package(msgs)["text"]
    assert blob[:50] not in out
    assert "[image removed]" in out and "look" in out


def test_tool_results_truncated_errors_and_assistant_kept():
    msgs = [
        {"role": "user", "content": "first task"},
        {"role": "tool", "content": "ok " + "z " * 2500},
        {"role": "tool", "content": "Traceback (most recent call last): " + "e " * 700},
        {"role": "tool", "content": "fine", "is_error": True},
        {"role": "assistant", "content": "I will do it", "tool_calls": [
            {"function": {"name": "t", "arguments": "{}"}}]},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "t", "arguments": "{}"}}]},
    ]
    r = build_package(msgs)
    t = r["text"]
    assert "chars truncated" in t
    assert "z " * 400 not in t
    assert "e " * 700 in t  # error kept longer than normal tool result
    assert "[tool ERROR] fine" in t
    assert "I will do it" in t and "calls: t({})" in t
    assert r["dropped"] == 0


def test_sections_and_non_dict_messages():
    r = build_package(
        ["bad", {"role": "assistant", "content": "hi"}],
        card={"id": "t_1", "title": "x"},
        parents=[],
        comments="note",
    )
    assert "## Card" in r["text"] and "## Comments" in r["text"]
    assert "## Parents" not in r["text"]
    assert "[user]" not in r["text"]


def test_deterministic():
    msgs = [{"role": "user", "content": "a"}, {"role": "tool", "content": "b" * 900}] * 50
    assert build_package(msgs, max_input_tokens=500) == build_package(msgs, max_input_tokens=500)


def test_budget_keeps_first_user_and_assistant_drops_old_tool_results():
    msgs = [{"role": "user", "content": "FIRST-USER-MESSAGE"}]
    for i in range(150):
        msgs.append({"role": "tool", "content": f"tool-{i} " + "q " * 250})
        msgs.append({"role": "assistant", "content": f"assistant-{i}"})
    r = build_package(msgs, max_input_tokens=4000)
    assert r["tokens"] <= 4000
    assert "FIRST-USER-MESSAGE" in r["text"]
    assert "assistant-0" in r["text"] and "assistant-149" in r["text"]
    assert "tool-149 " in r["text"] and "tool-0 " not in r["text"]
    assert "messages omitted" in r["text"]
    assert r["dropped"] > 0 and r["tokens_before"] > r["tokens"]


def test_500k_token_transcript_fits_max_input_tokens():
    chunk = "word " * 400  # ~500 tokens
    msgs = [{"role": "user", "content": "start"}]
    for i in range(1000):
        msgs.append({"role": "assistant", "content": f"step {i} " + chunk,
                     "tool_calls": [{"function": {"name": "x", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "content": chunk * 2})
    before = sum(estimate_tokens(m["content"]) for m in msgs)
    assert before >= 500_000
    r = build_package(msgs, card={"id": "t_x"}, max_input_tokens=120_000)
    assert r["tokens"] <= 120_000
    assert r["tokens_before"] >= 500_000
    assert "start" in r["text"]


def test_hard_cap_when_head_and_first_user_exceed_budget():
    r = build_package(
        [{"role": "user", "content": "u " * 1500}], card="c " * 1500, max_input_tokens=100
    )
    assert r["tokens"] <= 100
