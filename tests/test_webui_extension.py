"""Static contract tests for the Hermes One extension assets.

NOTHING IN THIS FILE MAY BE GATED ON AN OPTIONAL IMPORT. These tests are the only
automated check on the browser-facing surface — the XSS-sink scan, the
one-wall-clock rule, the tab structure, the pt-BR vocabulary, the CSS token
contract, the ``CAUSE_WORDS`` closed-set agreement — and a module-level
``pytest.importorskip`` skips the WHOLE MODULE, not the tests after it.

That is not hypothetical: the dashboard plugin-API parity tests used to live at
the bottom of this file behind exactly such a gate, and CI installs no fastapi,
so all 67 tests here silently did not run in the job that owns the 100 %
coverage gate. Measured: 1964 passing with fastapi present, 1897 without. They
now live in ``tests/test_dashboard_plugin_api.py``, whose first statement is the
gate — see that file's docstring for the full account.

So: if something here needs a dependency this plugin does not declare (pyyaml is
the only one), it belongs in its own module, not behind a gate in this one.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent
EXTENSION = ROOT / "webui_extension" / "hermes-smart-router"


def _node() -> str:
    """The node binary, or a skip.

    A bare "node" in subprocess.run raises FileNotFoundError, which reads as a
    failing test rather than an unavailable tool - and the deployment keeps node
    in ~/.local/bin, which is not on the PATH of every runner that reaches these
    tests. A syntax check that cannot run is a skip; it is not a syntax error.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    return node

def test_extension_manifest_declares_token_v1_sidecar():
    manifest = json.loads((EXTENSION / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["id"] == "hermes-smart-router"
    assert manifest["scripts"] == ["router-nav.js"]
    assert manifest["stylesheets"] == ["router-nav.css"]
    assert manifest["sidecar"] == {
        "type": "loopback",
        "origin": "http://127.0.0.1:8791",
        "health_path": "/health",
        "proxy_auth": "token-v1",
    }


def test_extension_script_mounts_the_console_instead_of_duplicating_it():
    """The extension script is navigation plus a mount, not a second UI.

    It once reimplemented every read the console does, so the two surfaces could
    disagree. What is load-bearing now is the mount itself: the WebUI grants its
    CSRF token only to pages it renders, so the console can only write when it
    runs inside the host document — hence srcdoc, which inherits the host origin.
    NOT because the served page refuses framing: the sidecar sends no
    X-Frame-Options and no frame-ancestors, and never has.
    """
    script_path = EXTENSION / "router-nav.js"
    script = script_path.read_text(encoding="utf-8")
    for block in ("THESIS:", "OWN-WORLD:", "STORY:", "FIRST VIEWPORT:", "FORM:"):
        assert block in script
    assert "srcdoc" in script, "the console must be framed same-origin"
    assert "main-view" in script, "it mounts in the host's central panel"
    assert "/console" in script
    assert "frame.src =" not in script, "a URL-framed console is refused by the sidecar"
    # The duplicate reader UI must not creep back.
    for gone in ("/status", "/blocklist", "renderPolicyTab", "postJSON"):
        assert gone not in script, f"{gone} belongs to the console, not to the mount"
    # This file's own accessibility surface is the error state it renders — the one
    # thing it puts on screen. The nav BUTTON's aria-label and the MutationObserver
    # that installs it moved to the shared hermes-panel-nav module, where all three
    # surfaces get them from one implementation; asserting them here would pin a
    # copy that no longer exists and quietly stop testing the real one.
    for accessibility_hook in ("aria-live", "setAttribute('role', 'alert')"):
        assert accessibility_hook in script
    assert "HermesPanelNav" in script, "navigation comes from the shared module"
    assert "window.HermesPanelNav" in script and "console.error" in script, (
        "a missing shared module must say so: without the guard the script dies "
        "before installing anything and the symptom is a button that is not there"
    )
    for destructive_pattern in (
        "document.body.innerHTML",
        "document.querySelector('main').innerHTML",
        "document.querySelector(\"main\").innerHTML",
        "eval(",
        "new Function",
    ):
        assert destructive_pattern not in script
    assert "textContent" in script
    checked = subprocess.run(
        [_node(), "--check", str(script_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert checked.returncode == 0, checked.stderr


#: The console's inline `<script>`, matched so that an ATTRIBUTED tag counts.
#:
#: `<script\b[^>]*>`, not `<script>`. Every harness in this repo — three helpers
#: here, two matchers in tests/test_console_logic.js, and the `node --check` step
#: in ci.yml — extracted "the" inline script with the bare pattern, which does not
#: see `<script type="module">`. So a second script block added with ANY attribute
#: would have been invisible to all of them at once, while each one asserted that
#: exactly one existed.
_CONSOLE_SCRIPT_RE = re.compile(r"<script\b[^>]*>\n?(.*?)\n?\s*</script>", re.S)


def _console_script_match(html: str):
    """The console's single inline `<script>`, as a MATCH object — and COUNTED.

    Returns the match rather than the body because two callers need
    ``.start()``/``.end()`` to blank the script region out of the markup.

    The count is the point. The three call sites this replaces each said "the
    console must carry exactly one inline `<script>`" and then used
    ``re.search``, which asserts only that a FIRST one exists — the one thing the
    sentence claims was the one thing not checked.
    """
    matches = list(_CONSOLE_SCRIPT_RE.finditer(html))
    assert len(matches) == 1, (
        f"console.html must carry exactly one inline <script>, found "
        f"{len(matches)} — every harness in this repo extracts the FIRST one, so "
        f"a second block is code that ships untested"
    )
    return matches[0]


def _console_inline_script() -> str:
    """Return the single inline <script> body of the impeccable console."""
    html = (EXTENSION / "console.html").read_text(encoding="utf-8")
    return _console_script_match(html).group(1)


def test_console_html_is_xss_safe_and_syntax_valid(tmp_path):
    """The console renders persisted, attacker-influenceable route/task text in
    replay, so its inline script must never use raw-markup sinks and must render
    via textContent. This guards the highest-XSS-surface code in the project.
    """
    script = _console_inline_script()
    for forbidden in ("innerHTML", "insertAdjacentHTML", "outerHTML", "eval(", "new Function", "document.write"):
        assert forbidden not in script, f"console.html inline script must not use {forbidden}"
    assert "textContent" in script
    # The replay wiring must be present: it reads real recorded traces and draws
    # the path they took, with no charting dependency.
    for token in ("drawPath", "/routes", "stepOutcome", "renderStep"):
        assert token in script, f"console.html must wire {token}"
    # Syntax must be valid (write the extracted body to a temp file for node --check).
    script_file = tmp_path / "console_inline.js"
    script_file.write_text(script, encoding="utf-8")
    checked = subprocess.run(
        [_node(), "--check", str(script_file)],
        text=True, capture_output=True, check=False,
    )
    assert checked.returncode == 0, checked.stderr


def test_wall_clock_is_read_in_exactly_one_place():
    """DESIGN.md §7: nowUtc() is the ONE wall-clock reader.

    Every time-dependent value takes the instant as a parameter — ``ago``
    receives it from the caller — so a second reader (a formatter that called
    ``Date.now()``, or a bare ``new Date()`` anywhere else) would make every
    rendering test pass at 05:00 UTC and fail at 07:00. ``new Date(x)`` with an
    argument is a timestamp CONVERSION, not a read, and stays legal.
    """
    import re

    script = _console_inline_script()
    assert "Date.now()" not in script, (
        "Date.now() must not appear anywhere in the console script; the only "
        "wall-clock read is the bare new Date() inside nowUtc()"
    )
    # A `new Date` not followed by an argument list is a wall-clock read too
    # (e.g. `(new Date).getTime()`); only the constructor-with-argument form
    # converts a timestamp.
    assert not re.search(r"new Date\s*(?!\()", script), (
        "a paren-less new Date is a wall-clock read; convert with new Date(x)"
    )
    bare = [m.start() for m in re.finditer(r"new Date\s*\(\s*\)", script)]
    assert len(bare) == 1, (
        "the bare new Date() (no args) must appear exactly once, inside "
        f"nowUtc() — found {len(bare)}"
    )
    definition = script.index("const nowUtc = ")
    assert definition < bare[0] < definition + 200, (
        "the one bare new Date() must live inside the nowUtc definition"
    )


def test_console_html_declares_its_six_screens_and_their_surfaces():
    """The console is six screens; each must exist with the hooks its screen
    needs. Tarefas, Decisões and the ladder read as ordered vertical sequences —
    a policy is a first-match table and a trace is a short path, so neither is a
    free-form canvas and the operator learns one way of reading this console."""
    html = (EXTENSION / "console.html").read_text(encoding="utf-8")
    for tab in ("tarefas", "simular", "modelos", "precos", "decisoes", "politica"):
        assert f'data-tab="{tab}"' in html
        assert f'id="panel-{tab}"' in html
    assert 'id="sheet"' in html, "the Tarefas screen is the ordered decision sheet"
    assert 'id="probeTask"' in html, "an operator must be able to try a task"
    assert 'id="ladder"' in html, "the capability ladder shows where tasks can land"
    assert 'id="failSafeBox"' in html, "§1.2 item 5: the last resort has its own block on the Modelos tab"
    assert 'id="routesTable"' in html
    assert 'id="replayPath"' in html, "replay lists the steps a real decision took"
    assert 'id="policyEditor"' in html, "the whole-file editor is its own tab now"
    assert "<svg" not in html, "no canvas survives: the screens are read as lists"


def test_console_tabs_read_as_tasks_simular_models_prices_decisions_policy():
    """The approved 2026-08-27 split: six destinations, in the operator's order.

    The task list is the axis the whole redesign hangs on, so it is the FIRST
    tab and the one born selected. The simulator leaves Tarefas (it occupied the
    first viewport and pushed down the sheet that answers the tab's question),
    the 24-hour band leaves Modelos, and the whole-file editor leaves its
    <details> — each for a tab of its own. The sidebar mirrors the same six
    words in the same order — one vocabulary, not two.
    """
    import re

    html = (EXTENSION / "console.html").read_text(encoding="utf-8")
    nav = re.search(r'<nav class="tabs".*?</nav>', html, re.S)
    assert nav, "the console declares its tab list"

    order = re.findall(r'id="(tab-\w+)".*?<span class="tab-name">([^<]+)</span>', nav.group(0), re.S)
    assert [tab for tab, _ in order] == [
        "tab-tarefas", "tab-simular", "tab-modelos", "tab-precos",
        "tab-decisoes", "tab-politica",
    ], "Tarefas leads; the six names say what they are"
    assert [label for _, label in order] == [
        "Tarefas", "Simular", "Modelos", "Preços", "Decisões", "Política",
    ]

    # Born selected: the markup itself carries the state, not a script pass.
    first = re.search(r'<button class="tab" id="tab-tarefas"[^>]*>', nav.group(0))
    assert first and 'aria-selected="true"' in first.group(0), "Tarefas is the tab an operator lands on"
    for other in ("tab-simular", "tab-modelos", "tab-precos", "tab-decisoes", "tab-politica"):
        line = re.search(rf'<button class="tab" id="{other}"[^>]*>', nav.group(0))
        assert line and 'aria-selected="false"' in line.group(0)

    panel = re.search(r'<section class="screen active" id="panel-tarefas"', html)
    assert panel, "the Tarefas panel is born active"
    assert re.search(r'<section class="screen" id="panel-simular"', html), "Simular starts inactive"
    assert re.search(r'<section class="screen" id="panel-politica"', html), "Política starts inactive"

    # The script state agrees with the markup it is born into.
    assert re.search(r"tab: 'tarefas',", html), "state.tab starts on Tarefas"

    # And the sidebar says the same six words in the same order. The source
    # escapes non-ASCII, so compare the escaped spellings it actually ships.
    nav_js = (EXTENSION / "router-nav.js").read_text(encoding="utf-8")
    sections = re.search(r"for \(const \[tab, label\] of \[([\s\S]*?)\]\)", nav_js, re.S)
    assert sections, "the sidebar declares its section list"
    pairs = re.findall(r"\['(\w+)', '([^']+)'\]", sections.group(0))
    assert pairs == [
        ("tarefas", "Tarefas"), ("simular", "Simular"), ("modelos", "Modelos"),
        ("precos", "Pre\\u00e7os"), ("decisoes", "Decis\\u00f5es"), ("politica", "Pol\\u00edtica"),
    ], "the sidebar mirrors the tabs: same words, same order"


def test_console_transcribes_the_classifier_tier_anchors():
    """The four group names on screen are the classifier's rubric, not a copy of it.

    `T1`..`T4` mean nothing to someone opening this console, and that was the
    operator's complaint: the screen named groups it never explained. The authority
    already exists in the engine — ``classify.TIER_ANCHORS`` is the text the
    classifier scores against — so the console transcribes it instead of minting a
    second vocabulary that can drift from the one that routes.

    Asserted against the module rather than against a copy of the phrases, so adding
    a tier to the rubric, or rewording one, fails HERE until the screen follows. A
    key outside the rubric is legal in ``tiers`` and the classifier can never pick
    it, so the console needs a sentence for that case too.
    """
    import re
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from router import classify

    html = (EXTENSION / "console.html").read_text(encoding="utf-8")
    block = re.search(r"const TIER_MEANING = \{(.*?)\n\s*\};", html, re.S)
    assert block, "the console must declare TIER_MEANING to explain the groups"

    entries = dict(re.findall(r"(\w+): \{ label: '([^']+)'", block.group(1)))
    assert set(entries) == set(classify.TIER_ANCHORS), (
        "the screen's groups and the classifier's rubric must be the same set: "
        f"screen={sorted(entries)} rubric={sorted(classify.TIER_ANCHORS)}"
    )
    for key in classify.TIER_ANCHORS:
        assert entries[key], f"{key} needs a short label"
        what = re.search(rf"{key}: \{{ label: '[^']+', what: '([^']+)'", block.group(1))
        assert what and len(what.group(1)) > 30, (
            f"{key} needs the rubric's own description, not just a label"
        )

    assert "TIER_UNKNOWN_WHAT" in html, (
        "a tier key outside the rubric is legal, so the screen owes it a sentence"
    )


import re as _re

# The second argument of ``el(tag, cls, text)`` is a class name, and so is every
# argument of the classList calls. Matched on what PRECEDES the literal, because the
# only reliable difference between `tier-fact rails` and a sentence is where it sits.
CLASS_POSITION = _re.compile(
    r"""(?:
          el\(\s*['"`][a-z0-9]+['"`]\s*,\s*
        | classList\.(?:add|remove|toggle)\(\s*
        | className\s*(?:=|\+=)\s*
        | dataset\.[A-Za-z_$][\w$]*\s*=\s*
        )$""",
    _re.X,
)


def _console_rendered_text() -> list[tuple[str, int, str]]:
    """Every string this console can put on screen, as ``(where, line, text)``.

    Four things are deliberately NOT rendered text, and counting them would force
    changes the spec forbids elsewhere:

    * HTML comments and the ``<style>`` block — this file keeps its design rationale
      in comments, two of which discuss the word ``rail`` on purpose.
    * ``${...}`` bodies inside template literals, which are code. Without this,
      ``reach${stale ? ' is-stale' : ''}`` reads as prose carrying ``stale`` when all
      it carries is a CSS class and a variable.
    * Single-token literals: identifiers, endpoint paths and the lint error codes the
      server sends. ``eloRow`` may not be renamed (spec §5.5 freezes the exported
      surface), the id ``#staleBanner`` is required by §4.2 and ``call('/blocklist')``
      is a route.
    * Anything in a class-name position. ``el(tag, cls, text)`` puts the class second
      and the text third, so the second argument is excluded by POSITION rather than
      by how it looks: ``el('div', `tier-fact rails${…}`)`` carries a space and would
      otherwise read as a sentence, and ``.tier-fact.rails`` is a real rule in the
      ``<style>`` — a class name is not something a reader is shown.
    * Attribute values that are not shown to anyone (``data-*``, ``class``, ``id``).
      ``placeholder``, ``title`` and ``aria-label`` ARE shown, so they are included.
    """
    html = (EXTENSION / "console.html").read_text(encoding="utf-8")
    script = _console_script_match(html)

    def blank(mo):
        return "\n" * mo.group(0).count("\n")

    markup = html[: script.start()] + html[script.end():]
    markup = re.sub(r"<style>.*?</style>", blank, markup, flags=re.S)
    markup = re.sub(r"<!--.*?-->", blank, markup, flags=re.S)

    out: list[tuple[str, int, str]] = []
    for attr in ("placeholder", "title", "aria-label"):
        for mo in re.finditer(attr + r'="([^"]+)"', markup):
            out.append(("markup", markup[: mo.start()].count("\n") + 1, mo.group(1)))
    for number, line in enumerate(markup.split("\n"), 1):
        stripped = re.sub(r"<[^>]*>", " ", line).strip()
        if stripped:
            out.append(("markup", number, stripped))

    # Hand-rolled because a regex cannot tell a quote inside a comment from a real
    # one, and this file has apostrophes in prose and quotes inside regex literals.
    body = script.group(1)
    start_line = html[: script.start(1)].count("\n") + 1
    index, size, line_number = 0, len(body), start_line
    while index < size:
        char = body[index]
        if char == "\n":
            line_number += 1
            index += 1
        elif char == "/" and body[index + 1:index + 2] == "/":
            while index < size and body[index] != "\n":
                index += 1
        elif char == "/" and body[index + 1:index + 2] == "*":
            index += 2
            while index + 1 < size and body[index:index + 2] != "*/":
                line_number += body[index] == "\n"
                index += 1
            index += 2
        elif char in "'\"`":
            quote, opened, buffer = char, line_number, []
            before = body[max(0, index - 48):index]
            index += 1
            while index < size:
                if body[index] == "\\":
                    buffer.append(body[index:index + 2])
                    index += 2
                    continue
                if body[index] == quote:
                    index += 1
                    break
                line_number += body[index] == "\n"
                buffer.append(body[index])
                index += 1
            if not CLASS_POSITION.search(before):
                out.append(("script", opened, "".join(buffer)))
        else:
            index += 1

    def without_interpolations(text: str) -> str:
        kept, depth, cursor = [], 0, 0
        while cursor < len(text):
            if text.startswith("${", cursor):
                depth += 1
                cursor += 2
                continue
            if depth and text[cursor] == "{":
                depth += 1
            elif depth and text[cursor] == "}":
                depth -= 1
                cursor += 1
                continue
            if not depth:
                kept.append(text[cursor])
            cursor += 1
        return "".join(kept)

    prose = []
    for where, line, text in out:
        clean = without_interpolations(text).strip() if where == "script" else text.strip()
        if " " in clean:  # a single token is a key, a class or a path — not prose
            prose.append((where, line, clean))
    return prose


def _console_ui_strings() -> list[tuple[str, int, str]]:
    """Every string this console can put on screen, single tokens included.

    ``_console_rendered_text`` deliberately drops single-token literals (keys,
    classes, paths). This guard needs them: ``el('button', 'btn go', 'Apply')``
    is a single token and exactly the label the §4.7 rule forbids. Comments and
    the ``<style>`` block are skipped for the same reasons the shared extractor
    skips them — this file keeps its rationale in comments, and a comment is
    not something the reader is shown.
    """

    html = (EXTENSION / "console.html").read_text(encoding="utf-8")
    script = _console_script_match(html)

    def blank(mo):
        return "\n" * mo.group(0).count("\n")

    markup = html[: script.start()] + html[script.end():]
    markup = re.sub(r"<style>.*?</style>", blank, markup, flags=re.S)
    markup = re.sub(r"<!--.*?-->", blank, markup, flags=re.S)

    out: list[tuple[str, int, str]] = []
    for attr in ("placeholder", "title", "aria-label"):
        for mo in re.finditer(attr + r'="([^"]+)"', markup):
            out.append(("markup", markup[: mo.start()].count("\n") + 1, mo.group(1)))
    for number, line in enumerate(markup.split("\n"), 1):
        stripped = re.sub(r"<[^>]*>", " ", line).strip()
        if stripped:
            out.append(("markup", number, stripped))

    body = script.group(1)
    start_line = html[: script.start(1)].count("\n") + 1
    index, size, line_number = 0, len(body), start_line
    while index < size:
        char = body[index]
        if char == "\n":
            line_number += 1
            index += 1
        elif char == "/" and body[index + 1:index + 2] == "/":
            while index < size and body[index] != "\n":
                index += 1
        elif char == "/" and body[index + 1:index + 2] == "*":
            index += 2
            while index + 1 < size and body[index:index + 2] != "*/":
                line_number += body[index] == "\n"
                index += 1
            index += 2
        elif char in "'\"`":
            quote, opened, buffer = char, line_number, []
            index += 1
            while index < size:
                if body[index] == "\\":
                    buffer.append(body[index:index + 2])
                    index += 2
                    continue
                if body[index] == quote:
                    index += 1
                    break
                line_number += body[index] == "\n"
                buffer.append(body[index])
                index += 1
            out.append(("script", opened, "".join(buffer)))
        else:
            index += 1
    return out


def test_console_vocabulario():
    """CA7: the words the screen says are the glossary's, and nobody else's.

    ``elo``, ``hop``, ``shadowed`` and ``stale`` are not in DESIGN.md's domain list
    (rule 6), so they are invented vocabulary that the rule itself forbids; ``rail``
    IS in the list, as a synonym of ``provider``, and the screen picks one of the two
    — **provedor**. Matching is on whole words: a literal ``grep -oi elo`` also hits
    ``modelo`` and ``pelo``, which are required Portuguese, so the criterion can only
    mean the word standing on its own.

    The five domain words that ARE allowed appear only glossed, in parentheses right
    after the Portuguese term, exactly as §4.6 writes them — the screen may say
    "Grupo de modelos (tier)" and may not say "tier" alone.
    """
    import re

    prose = _console_rendered_text()
    assert len(prose) > 200, "the extractor stopped seeing the screen's own sentences"

    for word in ("elo", "hop", "rail", "shadowed", "stale"):
        # The plural is the same invented word: the file said "these elos" and "3
        # hops" before this criterion, and a guard blind to the -s would have let
        # both back in. Verified by mutation: without the `s?` the plural passes.
        pattern = re.compile(r"(?<![\w-])" + word + r"s?(?![\w-])", re.I)
        found = [(w, ln, t) for w, ln, t in prose if pattern.search(t)]
        assert not found, (
            f"'{word}' is not this screen's vocabulary (§4.6), and it reaches the "
            f"reader in {len(found)} place(s): "
            + "; ".join(f"{w}:{ln} {t[:70]!r}" for w, ln, t in found[:6])
        )

    for word in ("tier", "breaker", "fail-safe", "blocklist", "profile"):
        pattern = re.compile(r"(?<![\w-])" + re.escape(word) + r"(?![-\w])", re.I)
        glossed = re.compile(r"\(\s*" + re.escape(word) + r"\s*\)", re.I)
        bare = [
            (w, ln, t) for w, ln, t in prose
            if pattern.search(t) and not glossed.search(t)
        ]
        assert not bare, (
            f"'{word}' is a domain word §4.6 allows only as a gloss in parentheses "
            f"after the Portuguese term, and it stands alone in {len(bare)} place(s): "
            + "; ".join(f"{w}:{ln} {t[:70]!r}" for w, ln, t in bare[:6])
        )


def test_last_tried_dating_says_tried_never_served():
    """Card t_0a3cff85: the elo dating says ``tentou``.

    The log records the head the executor dispatched and never an outcome, so
    ``atendeu`` would claim a fact nobody measured — and ``nunca`` would claim
    an absence that may just be an absent log (DESIGN.md rule 1). Both words
    are banned inside the dating literal itself; the extractor strips the
    ``${...}`` interpolation, so the fixed prefix is what is pinned here.
    """

    prose = _console_rendered_text()
    dating = [t for _, _, t in prose if "última decisão que tentou este modelo" in t]
    assert dating, "the elo dating phrase must exist and be renderable text"
    for t in dating:
        assert "atendeu" not in t, f"a dating phrase claims an outcome: {t!r}"
        assert "nunca" not in t, f"a dating phrase invents an absence: {t!r}"


def test_share_phrase_says_tried_never_served():
    """Card t_d2210a7c: the chain-entry share says ``tentada``, never ``atendeu``.

    The share is over the decisions that TRIED the elo (the log records the
    head the executor dispatched and never an outcome), so ``atendeu`` would
    assert a fact nobody measured. The extractor strips the ``${...}``
    interpolations, so the fixed prefix and the denominator word are what is
    pinned here — the window the phrase cites is pinned behaviourally in
    test_console_logic.js, where the fixture's span is known.
    """

    prose = _console_rendered_text()
    share = [t for _, _, t in prose if "tentada em" in t]
    assert share, "the chain-entry share phrase must exist and be renderable text"
    for t in share:
        assert "atendeu" not in t, f"a share phrase claims an outcome: {t!r}"
        assert "decisões" in t, f"a share phrase lost its denominator: {t!r}"


def test_cap_mark_and_bypass_speak_the_cards_own_words():
    """Card t_eed59abb: the per-attempt ceiling mark is one fixed vocabulary.

    ``acima do teto agora`` names the condition and ``entra no teto às``
    names the hour the multiplier falls back within the cap; the auto-shutdown
    phrase says the mechanism on the group with the card's exact literal. The
    true/false pairing of the bypass sentence (present only when
    ``time_cap_bypassed`` is true) is pinned by the render test in
    test_console_logic.js; this test pins that the words exist as renderable
    text at all — and that the group phrase never drifts into the preset's,
    because ``se isso fosse deixar a fila vazia`` describes the CONFIG while
    ``aplicá-lo deixaria a fila vazia`` describes a decision that happened.
    """

    prose = _console_rendered_text()
    mark = [t for _, _, t in prose if "acima do teto agora" in t]
    assert mark, "the per-attempt mark must be renderable text"
    assert any("entra no teto às" in t for _, _, t in prose), (
        "the coming-back-under sentence (second half of the mark) must exist"
    )
    bypass = [t for _, _, t in prose if "o teto se desligou nesta decisão" in t]
    assert bypass, "the auto-shutdown phrase must exist as renderable text"
    for t in bypass:
        assert "se isso fosse" not in t, (
            "the group phrase must not reuse the preset's config sentence: "
            f"{t!r}"
        )


def test_console_hour_field_speaks_the_honest_limit():
    """Card t_fbdc3e38: the test-hour field and its limit sentence exist together.

    The Hora do teste (UTC) field only reprices and reorders what the console
    DISPLAYS; the engine's decision used the server's hour — the sentence is
    the card's contract ("sem ela, o card não está feito"), so it must sit in
    the markup beside the field, verbatim, and the "hora escolhida" mark the
    override carries must be renderable text. The wall-clock rule (§7) is
    already pinned by test_wall_clock_is_read_in_exactly_one_place; this test
    pins that the field cannot exist without its limit.
    """

    html = (EXTENSION / "console.html").read_text(encoding="utf-8")
    phrase = "Isto reprecifica e reordena a fila mostrada nesta hora. A decisão do motor usa a hora do servidor."
    assert 'id="probeHour"' in html, "the task test carries a Hora do teste (UTC) field"
    assert 'id="probeNow"' in html, "back-to-now is an explicit control (Agora)"
    assert phrase in html, "the limit sentence must exist verbatim beside the field"
    field_at = html.index('id="probeHour"')
    phrase_at = html.index(phrase)
    assert abs(field_at - phrase_at) < 1200, (
        "the sentence must sit beside the field, not somewhere the operator "
        "reading the field cannot see"
    )
    # The mark the override carries is renderable text, pinned the same way the
    # other cards pin their literals (the ${...} interpolation is stripped, so
    # the fixed prefix is what is asserted).
    prose = _console_rendered_text()
    mark = [t for _, _, t in prose if "hora escolhida" in t]
    assert mark, "the 'hora escolhida: {HH}:00 UTC' mark must be renderable text"


def test_console_write_surface_speaks_one_pt_br_vocabulary():
    """§4.7/§6.10: the write surface has no English labels or messages.

    The same write used to carry two vocabularies on one screen: the editor and
    the presets said the §4.7 literals while the inspector said Apply/Preview/
    Revert with Rejected/Invalid/Checking…/Written. messages. Every one of the
    seven English forms is pinned here as a label or a UI message, so the second
    vocabulary cannot come back as a rename or a copy.

    Matching is whole-word and case-sensitive: ``/apply`` is a route and
    ``doPreview`` a symbol — neither is rendered — while the words as the spec's
    table spells them (capitalised) are what the operator would read. The plural
    check covers every surface: ``el()`` labels, ``setMsg`` messages, markup
    text and placeholders alike, single tokens included.
    """

    import re

    strings = _console_ui_strings()
    assert len(strings) > 100, "the extractor stopped seeing the console's strings"
    for word in ("Apply", "Preview", "Revert", "Rejected", "Invalid", "Checking", "Written"):
        pattern = re.compile(r"(?<![\w-])" + re.escape(word) + r"(?![-\w])")
        found = [(w, ln, t) for w, ln, t in strings if pattern.search(t)]
        assert not found, (
            f"'{word}' is not this screen's write vocabulary (§4.7/§6.10), and it "
            f"reaches the reader in {len(found)} place(s): "
            + "; ".join(f"{w}:{ln} {t[:70]!r}" for w, ln, t in found[:6])
        )


def test_console_english_terms_of_this_card_stay_out():
    """§4.1/§3.2/§3.3/§6.10: the eight English forms this card removed never
    come back as rendered text.

    Each term is matched the way an operator would read it, and the two
    extractors already say what is NOT rendered: comments and the ``<style>``
    block are stripped, and identifiers are never string literals.

    * Single-token labels (``Refresh``, ``Refreshing``, ``Edit``, ``Done``,
      ``banned``, ``left``, ``Routing``) are matched whole-word and
      case-sensitive over ``_console_ui_strings()`` — the extractor that keeps
      single tokens. ``_console_rendered_text`` drops them, and a bare
      ``'banned'`` returning as ``el('span', 'state', 'banned')`` is exactly
      the regression to catch. Whole-word, case-sensitive matching is what
      keeps ``editMode``/``refresh`` (identifiers), ``Editar`` (a different
      word) and ``ArrowLeft`` (capital L) out of the result.
    * ``Stop editing`` is a phrase, matched as such.
    * ``on``/``off`` count only as EXACT standalone script literals — the
      shape the routing fact's value had. The same letters legitimately build
      CSS classes (``' on'``/``' off'`` fragments) and the ``autocomplete``
      attribute, so a substring or whole-word sweep would ban innocent uses;
      the fact value is an exact string, and that is the position pinned.
    * ``ago`` is whole-word: the pt-BR form this replaced ("há 5m") carries
      no English word to trip on, and "5m ago" has the word standing alone.

    The one location this test cannot see is the `` of `` inside the
    cap-bypass sentence: it sits in a nested template inside a template
    interpolation, which both extractors stop at — that preposition is pinned
    by the render test in test_console_logic.js (the "de 1,5×" sentence).
    """

    import re

    strings = _console_ui_strings()
    assert len(strings) > 100, "the extractor stopped seeing the console's strings"
    for word in ("Refresh", "Refreshing", "Edit", "Done", "banned", "left", "Routing", "ago"):
        pattern = re.compile(r"(?<![\w-])" + re.escape(word) + r"(?![\w-])")
        found = [(w, ln, t) for w, ln, t in strings if pattern.search(t)]
        assert not found, (
            f"'{word}' is English an operator would read (§4.1/§3.2/§6.10), and it "
            f"reaches the reader in {len(found)} place(s): "
            + "; ".join(f"{w}:{ln} {t[:70]!r}" for w, ln, t in found[:6])
        )
    phrase = re.compile(r"(?<![\w-])Stop editing(?![\w-])")
    found = [(w, ln, t) for w, ln, t in strings if phrase.search(t)]
    assert not found, (
        "'Stop editing' is English an operator would read (§4.1), and it reaches "
        f"the reader in {len(found)} place(s): "
        + "; ".join(f"{w}:{ln} {t[:70]!r}" for w, ln, t in found[:6])
    )
    exact = [(w, ln, t) for w, ln, t in strings if w == "script" and t in ("on", "off")]
    assert not exact, (
        "the routing fact's value must be 'ligado'/'desligado', not English: "
        + "; ".join(f"{ln} {t!r}" for _, ln, t in exact)
    )


def test_console_lint_error_sentence_lives_once_in_the_map():
    """§4.7: the lint-error sentence exists once — in the WRITE map.

    The literal scan (comment #92) found it twice: the map's template (fill()
    at the preset banner) and a same-head inline copy in renderWarnings that
    spelled its own plural. The count is on the sentence's HEAD, up to the
    interpolation — a re-spelled copy ("2 erros no arquivo" instead of
    "2 erro(s) no arquivo") would still trip it, which is exactly how the
    second copy was born. The extractor strips comments and the ``<style>``
    block, so a mention in prose does not count.
    """

    strings = _console_ui_strings()
    head = "Não é possível salvar enquanto houver erro"
    found = [(w, ln, t) for w, ln, t in strings if head in t]
    assert len(found) == 1, (
        f"the lint sentence's head must exist once (the WRITE map), found {len(found)}: "
        + "; ".join(f"{w}:{ln} {t[:70]!r}" for w, ln, t in found[:6])
    )


def test_console_text_editor_warning_is_present_once_from_the_map():
    """§4.7: the whole-file editor's warning is present, once, from the map.

    The literal scan (comment #92) found the sentence missing from the
    editor's ``<details>`` entirely. It quotes "Ver o que muda" and "Salvar", so it
    cannot re-spell them: the sentence lives in the WRITE map and boot stamps
    the empty ``#jsonNote`` paragraph, exactly like the three write buttons.
    The script therefore carries exactly one copy of the sentence (the map),
    and the markup carries none outside it. Since the 2026-08-27 split the
    editor is its own tab — the tab IS the disclosure — so the paragraph's
    home is the Política panel.
    """

    html = (EXTENSION / "console.html").read_text(encoding="utf-8")
    politica = _panel(html, "politica")
    assert 'id="jsonNote"' in politica, "the warning's paragraph lives on the Política tab"
    strings = _console_ui_strings()
    sent = [(w, ln, t) for w, ln, t in strings if "Aqui você edita o arquivo de política inteiro" in t]
    assert len(sent) == 1, (
        "the whole-file warning must exist once, in the WRITE map — found "
        + f"{len(sent)}: " + "; ".join(f"{w}:{ln} {t[:70]!r}" for w, ln, t in sent[:6])
    )
    script = _console_inline_script()
    assert "WRITE.textEdit" in script, "boot must stamp the warning from the map"
    assert "['refresh', WRITE.refresh]" in script, "the refresh button is stamped from the map too"


def test_extension_css_only_dresses_the_nav_button():
    """This stylesheet's whole job is the rail button.

    It used to style a five-tab panel; then it styled the panel and the frame. Both
    of those now live in the shared hermes-panel.css that every surface loads, so
    anything about .hermes-panel, .hp-frame or .hp-error asserted HERE would pin a
    duplicate — which is precisely how the three surfaces drifted into looking like
    three different products.
    """
    import re

    css = (EXTENSION / "router-nav.css").read_text(encoding="utf-8")

    assert ".hermes-smart-router-nav" in css
    assert "hover" in css, "the nav button needs a hover state"
    assert "@media" in css, "the sidebar label collapses on a narrow shell"
    # Colour means state in this system, so a nav button uses contrast, not a hue.
    assert "var(--muted)" in css and "var(--text)" in css
    assert "#8fb2ff" not in css, "the hard-coded brand blue is gone"

    # This button lives in the HOST document, not inside a panel, so it can only
    # use tokens the host itself defines. Measured live on the running WebUI, the
    # host sets --bg, --text, --muted and --accent at :root and does NOT define
    # --surface-raised or --faint — so a rule written against those resolved to
    # nothing and the hover state silently did nothing at all. The panel tokens
    # are only in scope inside .hermes-panel.
    HOST_DEFINES = {"--bg", "--text", "--muted", "--accent"}
    used = set(re.findall(r"var\((--[a-z-]+)", css))
    assert used <= HOST_DEFINES, (
        f"{used - HOST_DEFINES} are panel tokens, undefined in the host document"
    )

    # The shared contract must NOT be restated here.
    for shared in (".hp-frame", ".hp-error", "main > .main-view", "prefers-reduced-motion"):
        assert shared not in css, f"{shared} belongs to hermes-panel.css, not here"


def test_every_css_token_the_console_uses_resolves_to_something():
    """A `var()` on an undefined token is not a fallback — it kills the declaration.

    Same defect class the nav-button test above already guards, which had gone
    unguarded inside the console: `var(--soft)` was used three times and defined
    nowhere. CSS treats a shorthand carrying an unresolvable `var()` as INVALID AT
    COMPUTED-VALUE TIME, so the property takes its initial value rather than
    ignoring the one component — `border-top: 1px solid var(--soft)` computed to
    `none` and `background: var(--soft)` to `transparent`. Two panel dividers did
    not render at all, and the marked compaction chip had no fill, distinguishable
    only by its border.

    The contract, stated as the three legitimate cases:

      * defined in this file's own `<style>` — the normal case;
      * a `--host-*` bridge input, which the host may or may not set, so it MUST
        carry a fallback (this file's fallback values are its no-theme design);
      * set at runtime from JS via `setProperty`, which likewise must carry a
        fallback for the frames before the script runs.

    Anything else is a token that resolves to nothing.
    """
    html = (EXTENSION / "console.html").read_text(encoding="utf-8")
    # Attribute-tolerant, same reason as _CONSOLE_SCRIPT_RE: `<style media="...">`
    # is still a stylesheet, and a second one this missed would carry rules nothing
    # here checks.
    styles = re.findall(r"<style\b[^>]*>(.*?)</style>", html, re.S)
    assert len(styles) == 1, f"expected one inline <style>, found {len(styles)}"
    style = styles[0]
    # The script block is counted by the shared helper, so this test also fails if
    # a second one appears.
    _console_script_match(html)

    defined = set(re.findall(r"(--[a-z0-9-]+)\s*:", style))
    js_set = set(re.findall(r"setProperty\(\s*['\"](--[a-z0-9-]+)['\"]", html))

    # Every use, split by whether it named a fallback.
    with_fallback = set(re.findall(r"var\(\s*(--[a-z0-9-]+)\s*,", style))
    bare = set(re.findall(r"var\(\s*(--[a-z0-9-]+)\s*\)", style))

    unresolvable = sorted(
        token for token in bare
        if token not in defined and not token.startswith("--host-")
    )
    assert not unresolvable, (
        f"used with no fallback and never defined, so the whole declaration is "
        f"dropped: {unresolvable}"
    )

    # A bridge input or a JS-driven token must never be used bare: the host may not
    # set it, and the script has not run for the first paint.
    needs_fallback = sorted(
        token for token in bare
        if token.startswith("--host-") or token in js_set
    )
    assert not needs_fallback, (
        f"these are set from outside the stylesheet and must carry a fallback: "
        f"{needs_fallback}"
    )

    # And a token used only WITH a fallback still has to come from somewhere, or
    # the fallback is the only value that will ever apply — which is a defect
    # dressed as a default.
    orphans = sorted(
        token for token in with_fallback
        if token not in defined
        and not token.startswith("--host-")
        and token not in js_set
    )
    assert not orphans, (
        f"no definition and no setProperty, so only the fallback can ever apply: "
        f"{orphans}"
    )


def test_embedded_console_hides_its_own_title_and_tabs():
    """Inside the Hermes One panel the shell already names the surface twice —
    the rail label and the sidebar's panel head — so the console's own masthead
    and tab row must not be drawn a second time. The review counted "Capability
    Router" three times on one screen (router-01-abertura.png). The
    .is-embedded class is set by the init path (window.self !== window.top),
    and the standalone page at /console still draws both."""
    html = (EXTENSION / "console.html").read_text(encoding="utf-8")
    assert ".is-embedded nav.tabs { display: none; }" in html, "tabs stay in the DOM but are not drawn twice"
    assert ".is-embedded .view-title { display: none; }" in html, "the shell names the surface; the title must not repeat it"


def test_health_badge_counts_exceptions_and_wears_amber():
    """The Health badge counts bans + breaker cooldowns (zero hides it) and
    wears the amber attention colour. Counting elos made the badge FALL from
    8 to 1 the moment a problem appeared; the elo total lives in the lede."""
    html = (EXTENSION / "console.html").read_text(encoding="utf-8")
    assert ".count.is-warn { color: var(--warn-text); }" in html
    assert "bans.length + breakers.length, true" in html
    assert "|| models.length" not in html, "the badge must not fall back to the elo inventory"


def test_health_facts_are_only_the_two_that_exist_nowhere_else():
    """The summary keeps ROUTING and CLASSIFIER — the rules count repeated the
    sheet's numbered list and the invalid count repeated the lint banner."""
    html = (EXTENSION / "console.html").read_text(encoding="utf-8")
    assert "fact(WRITE.routing" in html and "fact('classifier'" in html
    assert "fact('rules'" not in html and "fact('invalid'" not in html


def test_json_actions_markup_is_what_the_js_harness_mirrors():
    """CA8: the write buttons live in ``#jsonActions``, UNLABELLED — the labels are the map's.

    ``tests/test_console_logic.js`` measures whether a ``Salvar`` exists **in the DOM**
    while the file carries a lint error — absent, not disabled (DESIGN.md:435-463). Its
    DOM stub has no markup, so the test seeds the box the way this file writes it
    (``seedJsonActions``). That mirror is only honest while the markup really is this,
    which is what this test pins: rename the container or move a button out of it and
    the JS test would keep passing over a fiction until this one fails.

    §4.7 decision 2: the LABELS are not in the markup at all. They live in the WRITE
    map — the single source — and boot stamps them (``stampWriteLabels``); a label in
    the markup would be the second copy the rule exists to prevent, and the JS mirror
    seeds the very labels the map spells.
    """
    html = (EXTENSION / "console.html").read_text(encoding="utf-8")
    assert '<div class="actions" id="jsonActions">' in html, (
        "the write buttons need a named container for a test to ask what is inside it"
    )
    box = html.split('<div class="actions" id="jsonActions">', 1)[1].split("</div>", 1)[0]
    for element_id in ("jsonApply", "jsonPreview", "jsonRevert"):
        assert f'id="{element_id}"' in box, f"{element_id} must live inside #jsonActions"
    # §4.7: no write vocabulary in the markup — the WRITE map is the only copy,
    # and stampWriteLabels hands the labels to these very buttons at boot.
    for label in ("Salvar", "Ver o que muda", "Voltar à versão anterior"):
        assert label not in box, (
            f"{label!r} is a §4.7 literal; hardcoded in the markup it would be a "
            "second copy of the map the console stamps from"
        )
    # jsonApply first in the file, so the console can put it back where it was after
    # detaching it: the JS test asserts that order and it comes from here.
    assert box.index('id="jsonApply"') < box.index('id="jsonPreview"'), (
        "Salvar precedes Ver o que muda in the restored order, and the markup is "
        "the order the console restores it to"
    )
    script = _console_inline_script()
    assert "function stampWriteLabels()" in script, (
        "boot stamps the static buttons from the map; without it the editor's "
        "buttons would render empty"
    )
    for key in ("WRITE.save", "WRITE.plan", "WRITE.revert"):
        assert key in script, f"the map must be the source of {key}"


def test_console_gates_the_save_button_on_the_lint_errors():
    """The gate exists, and it DETACHES rather than disabling.

    A static read, because the behaviour is measured dynamically in the JS suite: what
    this pins is that the console never reaches for ``$('jsonApply')`` unguarded, which
    would throw the moment the node is detached — the failure mode that turns a lint
    error into a blank screen.
    """
    script = _console_inline_script()
    assert "function syncSaveButtons()" in script
    assert "removeChild(save)" in script, "absent, not disabled (DESIGN.md:435-463)"
    assert "function saveButton()" in script, (
        "one accessor for a node that may be detached; getElementById cannot find it then"
    )
    assert "$('jsonApply').disabled" not in script, (
        "an unguarded read of a detachable node is the crash this accessor prevents"
    )


def _panel(html: str, name: str) -> str:
    """The markup of one screen, from its <section> to the next one."""
    start = html.index(f'id="panel-{name}"')
    rest = html[start:]
    end = rest.index("</section>")
    return rest[:end]


def test_console_reads_in_the_order_the_spec_fixes():
    """§1.2: each tab carries its own subject, in the order the reader needs it.

    Two moves this pins, both measured as reading problems before they were made:

    * **The groups of models live on Modelos, under the presets.** They were on Tarefas,
      below the rule list, which put "which models exist and in what order each group
      tries them" as a second subject under "where does a task land". CA4 is verified by
      opening Modelos, so the placement is part of the criterion and not decoration.
    * **The simulator is its own tab, not the first thing on Tarefas.** It occupied the
      first viewport of the rules tab and pushed down the list that answers the tab's
      question — a list answers by reading, a simulation is verification, which you
      seek when you want it (2026-08-27 split). The 24-hour band moves with it out of
      Modelos.
    """
    html = (EXTENSION / "console.html").read_text(encoding="utf-8")
    modelos = _panel(html, "modelos")
    tarefas = _panel(html, "tarefas")
    simular = _panel(html, "simular")
    precos = _panel(html, "precos")

    assert 'id="ladder"' in modelos, "the groups of models are read on Modelos"
    assert 'id="ladder"' not in tarefas, "and are not a second subject under the rule list"
    assert 'id="presetBox"' in modelos, "the presets are the first control on Modelos"
    assert modelos.index('id="presetBox"') < modelos.index('id="ladder"'), (
        "choose the strategy, then read what it produced"
    )

    assert 'id="sheet"' in tarefas
    assert 'id="probeForm"' not in tarefas, "the simulator left the rules tab"
    assert 'id="probeForm"' in simular, "and has a tab of its own"
    assert simular.index('id="probeForm"') < simular.index('id="probeResult"'), (
        "the task first, then what the probe found"
    )
    assert 'id="priceStrip"' in precos, "the 24-hour band is read on Preços"
    assert 'id="priceStrip"' not in modelos, "and is not a second subject on Modelos"

    # The ids themselves are the contract router-nav.js and the JS suite match on, so
    # a move must never be a rename.
    for element_id in ("sheet", "probeTask", "ladder", "routesTable", "replayPath", "chainPlan", "clockbar"):
        assert f'id="{element_id}"' in html, f"{element_id} keeps its historic id"


def test_console_cause_map_covers_the_closed_set_exactly():
    """The cause column reads pt-BR over the engine's CLOSED vocabulary.

    ``router.decision_log.VALID_CAUSES`` is the one authority for which cause
    strings exist; ``CAUSE_WORDS`` in console.html is the one authority for
    how they read. This test binds the two: every member has a phrase, no
    phrase exists without a member, and both rendering points call the map.
    It is the gate half of card t_e10949c5 — the JS suite pins the rendered
    words, this pins that a NEW VALID_CAUSES member cannot reach production
    without its phrase (the raw enum on an all-Portuguese screen is exactly
    the defect the card fixes, and "a new cause lands with its test, never
    silently" is ROUTINE_CAUSES' own rule for the clause side).
    """
    from router.decision_log import VALID_CAUSES

    script = _console_inline_script()

    # The map, parsed out of the inline script the way the harness runs it:
    # a static read here, the rendered behaviour in test_console_logic.js.
    match = _re.search(r"const CAUSE_WORDS = \{([\s\S]*?)\};", script)
    assert match, "console.html must define the one CAUSE_WORDS map"
    members = _re.findall(r"^\s{8}(\w+):\s*'([^']+)'", match.group(1), _re.M)
    assert members, "the map parsed empty — the regex no longer matches the file"
    words = dict(members)

    missing = VALID_CAUSES - set(words)
    assert not missing, (
        f"cause(s) without a pt-BR phrase: {sorted(missing)} — add them to "
        "CAUSE_WORDS in console.html, or the column renders the raw enum"
    )
    invented = set(words) - VALID_CAUSES
    assert not invented, (
        f"phrase(s) for causes the engine does not produce: {sorted(invented)} — "
        "the log's closed set coerces unknowns to unknown_cause, so these rows "
        "can never render"
    )
    # The two mechanisms the card named as the misread, pinned as WORDS:
    # profile_ignored must say who PREVAILED (the profile), and
    # role_out_of_scope must say which half applied (the model's).
    assert "perfil" in words["profile_ignored"].lower(), (
        "profile_ignored read as 'the profile was ignored' for 135 of 158 "
        "measured decisions while the mechanism was the opposite — the word "
        "must name who prevailed"
    )
    assert "modelo" in words["role_out_of_scope"].lower(), (
        "role_out_of_scope must name the half that applied: the rule's model "
        "half ran, the role half was never this path's to move"
    )

    # ONE authority, used by BOTH rendering points: the row's column and the
    # replay step's chip. A second replace() beside the map is how the two
    # surfaces drift into two vocabularies for the same fact.
    assert script.count("causeWord(") >= 3, (
        "causeWord must be called at both rendering points (plus its own "
        "definition) — a point that renders a cause without it speaks a "
        "second vocabulary"
    )
    assert "'cause', String(" not in script, (
        "a rendering point still builds the cause span from the raw value "
        "('cause', String(...)) — both points must go through causeWord"
    )


def test_the_rail_window_baseline_matches_the_registry():
    """`RAIL_WINDOWS` is a transcription, so it must be asserted equal to its source.

    The console needs it: `GET /capabilities` is an OPTIONAL read (DESIGN.md §7)
    and the clock line has to answer "which rail is expensive now" without it. So
    the table stays — but nothing compared it to `MODEL_CAPABILITIES`, and it had
    drifted on both rails, in the two directions that cost most:

      * `xiaomi: [{hours: [16, 24], multiplier: 0.8}]` — the registry publishes NO
        window for either mimo elo, deliberately. `capabilities.py` records that the
        0.8x is a prepaid Token Plan credit coefficient while this install bills
        pay-as-you-go, so carrying it said metered cost fell 20 % for 8 h/day when
        real cost was 1.25x the estimate there. And `railWindowRows` only overrides a
        rail the registry prices, so the console announced the discount ALWAYS, on
        four tabs, even with /capabilities answered.
      * `deepseek` with no `weekdays` gate — the registry gates both windows Mon-Fri
        (added after a silent vendor edit), so the console priced deepseek at 2x for
        14 h every weekend the vendor bills at 1x.

    Compared as SETS OF WINDOWS PER PROVIDER, not as text: the two spellings differ
    by design (`hours` vs `hours_utc`), and what has to agree is the pricing fact.
    """
    from router.capabilities import MODEL_CAPABILITIES

    script = _console_inline_script()
    # Capture THROUGH the newline before the closing brace, so the last entry
    # keeps the terminator the per-rail regex below needs. (Without it the final
    # rail parsed as absent and the comparison passed for the wrong reason —
    # caught by the non-vacuity assertion below.)
    block = re.search(r"const RAIL_WINDOWS = \{(.*?\n)\s*\};", script, re.S)
    assert block, "RAIL_WINDOWS is gone or reformatted — this test cannot see it"

    # Parse the JS object literal into {provider: {(start, end, weekdays, mult)}}.
    console_windows: dict[str, set] = {}
    for rail, body in re.findall(r"(\w+):\s*\[(.*?)\],?\n", block.group(1), re.S):
        entries = set()
        for entry in re.findall(r"\{([^{}]*)\}", body):
            hours = re.search(r"hours:\s*\[(\d+),\s*(\d+)\]", entry)
            mult = re.search(r"multiplier:\s*([\d.]+)", entry)
            weekdays = re.search(r"weekdays:\s*\[([\d,\s]*)\]", entry)
            assert hours and mult, f"unparsed RAIL_WINDOWS entry: {entry}"
            days = (
                tuple(int(d) for d in weekdays.group(1).replace(" ", "").split(",") if d)
                if weekdays else None
            )
            entries.add((int(hours.group(1)), int(hours.group(2)),
                         days, float(mult.group(1))))
        console_windows[rail] = entries
    declared_rails = re.findall(r"^\s{8}(\w+):", block.group(1), re.M)
    assert sorted(console_windows) == sorted(declared_rails), (
        "the RAIL_WINDOWS regex did not match every rail in the table — a partial "
        f"parse would compare a subset and pass for the wrong reason: {console_windows}"
    )

    registry_windows: dict[str, set] = {}
    for entry in MODEL_CAPABILITIES.values():
        for window in entry.get("price_windows") or []:
            start, end = window["hours_utc"]
            days = window.get("weekdays")
            registry_windows.setdefault(entry["provider"], set()).add(
                (int(start), int(end),
                 tuple(days) if days is not None else None,
                 float(window["multiplier"]))
            )

    assert console_windows == registry_windows, (
        "the console's offline price-window baseline disagrees with the capability "
        f"registry.\n  console:  {console_windows}\n  registry: {registry_windows}"
    )


def test_the_consoles_capability_field_list_is_a_subset_of_the_registrys():
    """`CAP_FIELDS` is a transcription, and it had drifted by two invented names.

    It carried `peak_multiplier` and `peak_windows_utc` — names that exist NOWHERE in
    Python and, per `git log -S` over `router/`, never have. The console was the only
    thing in the repo that understood them, so a hop written that way was priced HERE
    and priced flat everywhere else: a display contradicting the run on exactly the
    input an operator is trying to diagnose.

    This is the check that would have caught it, and it is a SUBSET rather than an
    equality: the console legitimately renders fewer fields than the registry stores
    (it has no use for `price_windows_verified` or `notes`), but it must never claim
    a field the registry would drop.
    """
    from router.capabilities import _REGISTRY_FIELDS

    script = _console_inline_script()
    block = re.search(r"const CAP_FIELDS = \[(.*?)\];", script, re.S)
    assert block, "CAP_FIELDS is gone or reformatted — this test cannot see it"
    fields = set(re.findall(r"'([a-z_]+)'", block.group(1)))
    assert fields, "the CAP_FIELDS regex matched nothing — vacuous"

    unknown = sorted(fields - set(_REGISTRY_FIELDS))
    assert not unknown, (
        f"the console reads capability fields the registry does not keep, so a hop "
        f"declaring them prices here and nowhere else: {unknown}"
    )


def test_every_writable_key_has_a_control_that_is_not_the_json_editor():
    """JSON is OPTIONAL: every key the write gate accepts is editable through a form.

    ``_HOT_KEYS`` is the closed set the server will merge, so it is the exact
    definition of "what can be changed". Two of its nine members had no control on
    any screen and could only be reached by typing YAML into the Política editor:

      * ``blocklist`` — the manual-ban list could be REMOVED from (each row carries
        its lift) and never added to, and the "Fora de rotação" block hid itself
        whenever nobody was banned, which is exactly the state in which an operator
        wants to ban somebody.
      * ``enabled`` — the router's master switch. The Modelos lede REPORTED it
        ("Roteamento: ligado") and nothing anywhere could change it.

    A third was reachable only by accident: ``classifier`` is in the console's
    ``EDITABLE`` set and ``renderInspector`` has always had a branch for it, and no
    pickBind call site named the bind — the editor existed, worked, and had no door.

    This is an AGREEMENT test, not a restatement: the covered set is DERIVED (the
    inspector's binds are read out of the console source) plus a declared table of
    the keys that have their own form, each entry verified by the expression that
    actually builds its patch. A new hot key fails here until it has a control, and
    a control that is deleted fails here too.
    """
    from router.service import _HOT_KEYS

    script = _console_inline_script()

    # The inspector's binds, read from the console rather than restated here.
    editable = re.search(r"const EDITABLE = new Set\(\[(.*?)\]\)", script, re.S)
    assert editable, "EDITABLE is gone or reformatted — this test cannot see it"
    binds = set(re.findall(r"'([a-z_]+)'", editable.group(1)))
    assert binds, "the EDITABLE regex matched nothing — vacuous"
    # 'rule' and 'tier' are the singular bind names for the plural policy tables.
    from_inspector = {{"rule": "rules", "tier": "tiers"}.get(b, b) for b in binds}

    # The keys with their own dedicated form, and the expression that builds the
    # patch each one posts. Greps, so a control that is removed or renamed fails
    # here rather than leaving a stale claim behind.
    dedicated = {
        "blocklist": "{ blocklist: { manual_ban: bans.concat([{ model }]) } }",
        "enabled": "{ enabled: !routingOn }",
        "price_windows": "return { price_windows: {",
        "compaction": "return { compaction: {",
        # Reordering rules is its own write: the drag handle and the arrow keys
        # post the whole list, never a fragment of the inspector's draft.
        "rules": "{ rules: next }",
    }
    for key, marker in dedicated.items():
        assert marker in script, (
            f"the form that writes '{key}' is gone: expected {marker!r} in the "
            f"console's inline script, so this key is JSON-only again"
        )

    covered = from_inspector | set(dedicated)
    # DELIBERATELY JSON-only: the session-alignment block (spec 2026-10-session-alignment-hook
    # §10 F13) gets no edit controls until F11's shadow-run review is accepted.
    # Reachable via plan/apply and the Política editor; a form would invite tuning
    # thresholds nobody has measured yet.
    covered |= {"alignment"}
    missing = sorted(_HOT_KEYS - covered)
    assert not missing, (
        f"the write gate accepts {missing} and no form on any tab can produce it, "
        f"so changing it requires the JSON editor — give it a control, or take it "
        f"out of _HOT_KEYS"
    )


def test_no_test_decides_its_verdict_from_the_operators_untracked_policy():
    """A test that READS `router.yaml` is a test whose answer depends on whose machine ran it.

    `router.yaml` is gitignored: it is the operator's policy, seeded from
    `router.example.yaml` on first load. `tests/conftest.py::_seed_live_router_config`
    copies the example over **only when router.yaml is ABSENT**, so a suite that reads
    it is hermetic in exactly the two places nobody operates from — a fresh CI checkout
    and a dev clone whose copy nobody edited — and reads real operator policy
    everywhere else.

    THIS COST A DEPLOY. On 2026-09-02 `router-deploy.sh` on the reference install
    aborted at its own gate with
    `test_a_banned_primary_missing_from_fallback_chain_still_routes` failing: the test
    asserts its premise (that `glm-5.3` is absent from the flat `fallback_chain`) and
    the operator's live chain there lists it. Green on CI, green on the developer's
    Mac, red on the only box that serves traffic — and the failure was a TRUE statement
    about the operator's policy, not a defect in the code under test. Three helpers had
    the bug, and all three docstrings already said "shipped": `test_adapter._live_config`,
    `test_rules._live_policy`, and the read inside
    `test_classifier_trust.test_the_shipped_policy_names_a_classifier_the_grant_must_cover`.

    Reading it as a PATH is still allowed — `test_one_sidecar_e2e` boots a real
    RouterService over the file and is asserting the sidecar's plumbing, not the
    policy's content. What is forbidden is parsing it to assert on what is inside.
    """
    root = Path(__file__).resolve().parent.parent
    offenders = []
    for path in sorted(root.joinpath("tests").rglob("test_*.py")):
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            code = line.split("#", 1)[0]
            if 'router.yaml"' not in code:
                continue
            # A parse of the repo-root file is the forbidden shape. A tmp_path
            # router.yaml a test writes itself is the whole point of a fixture, and
            # handing the path to RouterService asserts plumbing, not content.
            if "safe_load" not in code and "read_text" not in code:
                continue
            if "root / " not in code and "ROOT / " not in code and "parents[" not in code:
                continue
            offenders.append(f"{path.relative_to(root)}:{lineno}: {line.strip()}")

    assert not offenders, (
        "these read and parse the OPERATOR's untracked router.yaml, so their verdict "
        "depends on the machine and they can fail a deploy gate on a healthy box — "
        "read router.example.yaml instead:\n  " + "\n  ".join(offenders)
    )


def test_the_suite_isolates_hermes_home_so_no_test_reads_the_live_install():
    """The harness guarantee that keeps machine-specific state out of every verdict.

    Three plugin readers resolve under ``HERMES_HOME``: ``blocklist._state_dir`` (breaker
    cooldowns), ``durable_decision_log.routes_path`` (the trace) and
    ``service.configured_models`` (the AGENT's config.yaml). Only the trace has a
    per-file override, which is why runbook §25.6 could measure that every
    ``router-deploy.sh`` run cleared the operator's live breaker state and could not
    cheaply fix it.

    The third reader turned the same asymmetry into a RED GATE: "an install with no agent
    config offers no models" is true on CI and on a developer's Mac and false on the only
    machine that serves traffic, so the deploy aborted on a healthy box — the same shape
    as the ``router.yaml`` defect the sibling guard below covers.

    Isolating the variable in an autouse fixture fixes all three at once, and it is
    asserted here because it is exactly the kind of guarantee that regresses silently:
    nothing FAILS when it disappears, tests simply start reading the operator's install
    again, and the failure surfaces on someone else's machine.
    """
    conftest = (Path(__file__).resolve().parent / "conftest.py").read_text(encoding="utf-8")
    # EVERY autouse fixture's body, not the first one: conftest carries several (the
    # spawn guard is one), and pinning the isolation to a particular name would fail on
    # a rename that changed nothing about the guarantee.
    bodies = re.findall(
        r"@pytest\.fixture\(autouse=True\)\s*\ndef \w+\([^)]*\):(.*?)(?=\n@pytest\.fixture|\ndef |\Z)",
        conftest,
        re.S,
    )
    assert bodies, "no autouse fixture found in conftest at all"
    body = "\n".join(bodies)
    assert 'monkeypatch.setenv("HERMES_HOME"' in body, (
        "the autouse fixture must isolate HERMES_HOME itself, not only the trace file: "
        "blocklist._state_dir and service.configured_models have no per-file override, "
        "so without it the suite reads and writes the operator's live install"
    )
    assert 'monkeypatch.setenv("HERMES_ROUTE_TRACE_FILE"' in body, (
        "and the trace override stays: HERMES_HOME alone would still let a test that "
        "sets HERMES_HOME itself write the operator's trace"
    )


def test_the_console_yaml_fragment_parses_to_what_it_claims():
    """The console emits YAML for a human to paste. This parses it and checks the meaning.

    The console has no YAML library — the fragment is built by string concatenation in
    ``agentQueueYaml`` — so the JS test can only assert the TEXT. Whether that text means
    what it looks like is a question only a parser answers, and it is a real question here:
    ``us.anthropic.claude-haiku-4-5-20251001-v1:0`` ends in ``v1:0``. A bare plain scalar
    with a colon is legal only because no space follows it, which parses today and breaks
    on the next id shape. The emitter quotes every scalar for that reason, and this is the
    check that the quoting is right rather than merely present.

    The expected strings are duplicated from tests/test_console_logic.js on purpose: this
    is an AGREEMENT test, and an agreement whose two sides read the same variable proves
    nothing. If the emitter changes, one side fails and a human reconciles them.
    """
    main_fragment = (
        'model:\n'
        '  default: "us.anthropic.claude-opus-5"\n'
        '  provider: "bedrock"\n'
        'fallback_providers:\n'
        '- provider: "bedrock"\n'
        '  model: "us.anthropic.claude-haiku-4-5-20251001-v1:0"\n'
    )
    parsed = yaml.safe_load(main_fragment)
    assert parsed == {
        "model": {"default": "us.anthropic.claude-opus-5", "provider": "bedrock"},
        "fallback_providers": [
            {"provider": "bedrock", "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0"}
        ],
    }, "the main chain's fragment means model.default plus its reserves"
    # The colon-bearing id survives as ONE string rather than becoming a mapping.
    assert isinstance(parsed["fallback_providers"][0]["model"], str)
    assert parsed["fallback_providers"][0]["model"].endswith("v1:0")

    vision_fragment = (
        'auxiliary:\n'
        '  vision:\n'
        '    provider: "bedrock"\n'
        '    model: "us.anthropic.claude-opus-5"\n'
        '    fallback_chain:\n'
        '    - provider: "zai"\n'
        '      model: "glm-4.5v"\n'
    )
    assert yaml.safe_load(vision_fragment) == {
        "auxiliary": {
            "vision": {
                "provider": "bedrock",
                "model": "us.anthropic.claude-opus-5",
                "fallback_chain": [{"provider": "zai", "model": "glm-4.5v"}],
            }
        }
    }

    # And the emitter really is the thing under test: the literals above must appear in the
    # console's own source, or this test is checking YAML rather than the console.
    script = _console_inline_script()
    for marker in (
        'model:\\n  default: ${q1(head.model)}\\n  provider: ${q1(head.provider)}\\n',
        "- provider: ${q1(a.provider)}\\n  model: ${q1(a.model)}\\n",
        "    fallback_chain:\\n",
    ):
        assert marker in script, f"the emitter no longer builds {marker!r}"


def test_the_plugin_manifest_name_is_the_key_both_installs_enable():
    """The manifest name is the ENABLE key, and it must match what config.yaml enables.

    `hermes_cli/plugins.py:_parse_manifest` computes `key = name` for a flat user plugin
    (the directory is never consulted at prefix=""), and that key is what
    `plugins.enabled` is matched against AND the `<plugin_id>` the LLM trust gate reads
    `plugins.entries.<plugin_id>.llm` from.

    This said `delegate-profile` from the 2026-08-27 rename until 2026-09-04 while both
    installs enabled `hermes-smart-router`, so the plugin never loaded on either box:
    `hermes plugins list` reported `delegate-profile │ not enabled`. Nothing was routed,
    while /status said `enabled: true` and the console rendered.

    Held to the SAME constant the plugin reads its own settings under, so the two cannot
    drift apart again.
    """
    import yaml

    manifest = yaml.safe_load((ROOT / "plugin.yaml").read_text(encoding="utf-8"))
    plugin_module = ROOT / "__init__.py"
    source = plugin_module.read_text(encoding="utf-8")

    assert manifest["name"] == "hermes-smart-router", (
        "the manifest name is the enable key; a mismatch means the plugin never loads"
    )
    assert '_CONFIG_NAMESPACE = "hermes-smart-router"' in source, (
        "non-vacuity: the config namespace this is held equal to really is that string"
    )
    # The tool name is deliberately NOT renamed (runbook §23.2: it appears in toolset
    # allowlists), and it lives in its own field — so the two must differ.
    assert manifest["provides_tools"] == ["delegate_profile"]
    # Every hook register() subscribes must be declared. The field is not enforced by the
    # loader, which is exactly why nothing caught `pre_kanban_dispatch` missing.
    subscribed = set(re.findall(r'register_hook\(\s*["\']([a-z_]+)["\']', source))
    assert subscribed, "non-vacuity: register() really does subscribe hooks"
    assert set(manifest["provides_hooks"]) == subscribed, (
        f"manifest declares {sorted(manifest['provides_hooks'])} but register() "
        f"subscribes {sorted(subscribed)}"
    )
