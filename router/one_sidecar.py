"""Hermes One Smart Router sidecar.

A stdlib-only loopback HTTP service consumed through Hermes One's consented
extension-sidecar proxy.  Its only state-changing credential is WebUI's
``token-v1`` secret: every route except ``/health`` and ``/console`` requires the
per-extension ``X-Hermes-Sidecar-Token`` header.

``/console`` is named here because this text is the ``--help`` an operator reads and it
used to say ``/health`` was the only exemption.  Measured 2026-09-03: ``GET /console``
with no token returns 200 and the full HTML shell.  That is deliberate and is the same
argument ``/health`` rests on — the screen has to load in order to EXPLAIN a token
failure — but a DATA route it is not, and the distinction belongs in the sentence rather
than in a comment further down.

READ AND WRITE, and the split is the important part — this text is also the
``--help`` output (``description=__doc__``), so it is what an operator reads.

  GET  is read-only over everything: policy, capabilities, blocklist, breaker
       state, the decision trace.  Nothing on a GET path mutates.
  POST is the guarded WRITE path.  ``/plan`` previews, ``/apply`` commits
       ``router.yaml`` behind ``rules.lint`` plus an optimistic ``base_hash``
       check, and ``/apply`` with ``action: "compaction"`` is RESTART-class: it
       stages a candidate for Hermes' own ``config.yaml`` and hands it to an
       out-of-repo dead-man switch.  ``/apply/revert`` restores the single
       ``.bak`` snapshot and is NOT hash-guarded (``apply_revert()`` takes no
       ``base_hash``), so it is the one write that cannot detect drift.

It still cannot mutate breaker state: that is written only by the plugin process
recording a delegation outcome, and every read here goes through
``Blocklist.would_block``, which consumes no probe slot.

This docstring used to say "The service itself is read-only over the router
policy; it cannot edit rules, change providers, or mutate breaker state" — in the
module that owns every route above.  Only the last clause was true.
"""
from __future__ import annotations

import argparse
import gzip
import hmac
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml
from urllib.parse import parse_qs, urlsplit

# Relative first, absolute second — the shapes ``rules.py``/``service.py`` use.
# ``python -m router.one_sidecar`` (the systemd unit) satisfies the relative
# form; loading the plugin as ``hermes_plugins.<slug>.router.one_sidecar``
# satisfies only the relative form; running this file as a bare script satisfies
# only the absolute one. No None fallback: both are hard requirements.
try:
    from .paths import resolve_policy_path
    from .service import RouterService
    from .threshold import apply_dynamic_thresholds, compute_model_thresholds, p_eff
except ImportError:  # pragma: no cover - bare-script/flat layout
    from router.paths import resolve_policy_path
    from router.service import RouterService
    from router.threshold import apply_dynamic_thresholds, compute_model_thresholds, p_eff

EXTENSION_ID = "hermes-smart-router"
TOKEN_HEADER = "X-Hermes-Sidecar-Token"
_VERSION = 1
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "[::1]", "localhost"}

# Provenance: captured at process boot so /status can report three ages.
# - _PROCESS_STARTED_AT: wall-clock ISO 8601 UTC captured once at boot
# - _CODE_MTIME: mtime of the router/ package directory (newest .py file)
_PROCESS_STARTED_AT = datetime.now(timezone.utc).isoformat()


def _code_mtime() -> str:
    """Return the newest mtime among router/*.py files, ISO 8601 UTC."""
    router_dir = Path(__file__).resolve().parent
    newest = 0.0
    for py_file in router_dir.glob("*.py"):
        try:
            mtime = py_file.stat().st_mtime
            if mtime > newest:
                newest = mtime
        except OSError:
            pass
    if newest:
        return datetime.fromtimestamp(newest, tz=timezone.utc).isoformat()
    # Fallback: directory mtime
    try:
        mtime = router_dir.stat().st_mtime
        return datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    except OSError:
        return _PROCESS_STARTED_AT


_CODE_MTIME = _code_mtime()

# The impeccable console ships beside the extension; the sidecar serves it as a
# static, same-origin HTML shell (auth-exempt like /health). All data it renders
# still flows through the token-gated JSON routes.
_CONSOLE_PATH = (
    Path(__file__).resolve().parent.parent
    / "webui_extension"
    / "hermes-smart-router"
    / "console.html"
)

# Routes grouped by the HTTP methods each accepts. A known route hit with a
# method it does not accept is a 405 (before auth, matching the historical
# POST/health contract); a route in neither set is a 404 (after auth).
#
# ``/explain`` is the one route in BOTH sets, and only because of size. It sizes a
# turn from the composed prompt (context + goal), and a context worth previewing
# is routinely larger than an HTTP request LINE may be: http.server refuses a
# request line over 65536 bytes with a 414, so the 120k-char context that made
# this parameter necessary cannot travel in a query string at all. GET is kept
# byte-for-byte compatible — a link-shaped probe with a small context still
# works — and POST is what makes the parameter usable for the case it exists for.
# Every other route stays single-method, so every historical 405 is unchanged.
_GET_ROUTES = frozenset(
    # /console is in here for the METHOD guard, not because it serves JSON. It was in
    # neither set, so `allowed` came back empty, the guard was skipped, and POST /console
    # fell through to `404 unknown route` — the one route where a wrong method was
    # indistinguishable from a typo'd path, against this module's own stated contract that
    # every single-method route answers 405. Measured 2026-09-03: POST /console -> 404
    # while POST /status -> 405 and POST /nope -> 404. Nothing shipped POSTs here, so no
    # client changes; a route table audit stops lying.
    {"/health", "/status", "/policy", "/blocklist", "/liveness",
     "/compaction", "/lint", "/explain", "/routes", "/outcomes", "/alignment", "/alignment-panel", "/tier-cost", "/premium-budget", "/capabilities", "/console"}
)
_POST_ROUTES = frozenset(
    {"/explain", "/plan", "/apply", "/apply/revert"}
)

# Context windows for the dynamic-threshold curve are NOT declared here. They come
# from `RouterService.compaction_windows()`, which derives the key set from the live
# policy's tier members and each window from the capability registry.
#
# This module used to carry its own MODEL_WINDOWS dict and a SUMMARIZER_WINDOW
# constant. Three of its four entries disagreed with the registry — including the
# shipped `compaction.model`, whose window was 272,000 against the registry's
# 131,072, so `summarizer_cap` was computed from 2.07x the real window and the
# RESTART-class apply wrote that into Hermes' own config.yaml. The fourth entry was
# a vendor alias the plan serves with a different model.
#
# Worse than the numbers: the READ path used the module constants while the APPLY
# path used the injectable instance attributes, so the screen could show thresholds
# the apply would not write. There is one source now, asked once per request, and
# `test_the_served_compaction_windows_come_from_the_registry` asserts it against
# `capabilities.MODEL_CAPABILITIES`.

# The exact phrase an operator must echo to arm the RESTART-class compaction
# apply, mirrored server-side (the console gates it client-side too).
_COMPACTION_CONFIRM = "COMPACT"

# The proven dead-man switch: validate -> backup -> detached(apply + restart +
# health-poll -> auto-revert). It owns the config.yaml mutation and recovery; the
# sidecar only hands it a fully-formed candidate config and returns immediately.
_SAFE_RESTART = Path.home() / "bin" / "hermes-safe-restart.sh"


def _effective_apply_state() -> Dict[str, Any]:
    """Can the RESTART-class projection run here? Named path when it cannot.

    The reason carries the exact path, because installing that launcher is the whole
    remedy and a generic "unavailable" would send the operator looking.
    """
    if _SAFE_RESTART.exists():
        return {"available": True, "reason": ""}
    return {
        "available": False,
        "reason": (
            f"safe-restart launcher not found at {_SAFE_RESTART} — the compaction choice "
            f"is recorded in router.yaml but nothing projects it into config.yaml here"
        ),
    }


def resolve_core_config_path() -> Path:
    """Resolve the Hermes core (profile) config.yaml — the compaction target.

    RESTART-class: unlike router.yaml this is not hot-reloaded, so edits here go
    exclusively through the safe-restart dead-man switch.
    """
    explicit = os.environ.get("HERMES_CORE_CONFIG_FILE")
    if explicit:
        return Path(explicit)
    home = os.environ.get("HERMES_HOME")
    base = Path(home) if home else Path.home() / ".hermes"
    # HERMES_HOME is already profile-scoped in a running agent
    # (~/.hermes/profiles/<name>), so appending profiles/<name> again yields
    # .../profiles/rodrigo/profiles/rodrigo/config.yaml and every compaction
    # attempt dies with ENOENT. Only reach for the profile when the home given
    # is the base directory. Same trap as routes_path().
    if base.parent.name == "profiles":
        return base / "config.yaml"
    profile = os.environ.get("HERMES_PROFILE", "")
    if not profile:
        return base / "config.yaml"
    return base / "profiles" / profile / "config.yaml"


def _default_restart_runner(candidate_path: Path) -> Dict[str, Any]:
    """Invoke the safe-restart script on a candidate config, returning promptly.

    The script backgrounds the apply+restart+health-poll+auto-revert via
    systemd-run and returns in well under a second, so a short timeout only trips
    on a genuinely missing/broken launcher, never on the restart itself.
    """
    if not _SAFE_RESTART.exists():
        return {"ok": False, "error": f"safe-restart launcher not found at {_SAFE_RESTART}"}
    try:
        proc = subprocess.run(
            ["bash", str(_SAFE_RESTART), str(candidate_path)],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "error": f"safe-restart invocation failed: {exc}"}
    if proc.returncode != 0:
        return {"ok": False, "error": "safe-restart rejected the candidate config",
                "detail": (proc.stderr or proc.stdout or "").strip()[-400:]}
    return {"ok": True, "restart": "scheduled"}


@dataclass(frozen=True)
class TokenState:
    token: Optional[str]
    present: bool


def resolve_token_path(extension_id: str = EXTENSION_ID) -> Path:
    """Resolve the token-v1 file using the WebUI's documented precedence."""
    explicit = os.environ.get("HERMES_EXT_SIDECAR_TOKEN_FILE")
    if explicit:
        return Path(explicit)

    state_dir = os.environ.get("HERMES_WEBUI_STATE_DIR")
    if state_dir:
        return Path(state_dir) / "sidecar-auth" / f"{extension_id}.token"

    hermes_home = os.environ.get("HERMES_HOME")
    if hermes_home:
        return Path(hermes_home) / "webui" / "sidecar-auth" / f"{extension_id}.token"

    if platform.system() == "Windows":
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
        return base / "hermes" / "webui" / "sidecar-auth" / f"{extension_id}.token"
    return Path.home() / ".hermes" / "webui" / "sidecar-auth" / f"{extension_id}.token"


def read_expected_token(extension_id: str = EXTENSION_ID) -> TokenState:
    """Read the expected token. Missing/unreadable means sidecar unavailable."""
    path = resolve_token_path(extension_id)
    try:
        return TokenState(token=path.read_text(encoding="utf-8").strip(), present=True)
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError, OSError):
        return TokenState(token=None, present=False)


def _error(status: int, message: str) -> Tuple[int, Dict[str, Any]]:
    return status, {"error": message}


def _accepts_gzip(value: Optional[str]) -> bool:
    """Return whether an Accept-Encoding field permits gzip.

    The WebUI proxy forwards this field unchanged. Honour a client explicitly
    refusing gzip with ``q=0`` rather than treating a token substring as consent.
    """
    if value is None:
        return False
    for coding in value.split(","):
        parts = [part.strip() for part in coding.split(";")]
        if not parts or parts[0].lower() != "gzip":
            continue
        for parameter in parts[1:]:
            name, separator, raw_value = parameter.partition("=")
            if separator and name.strip().lower() == "q":
                try:
                    return float(raw_value.strip()) > 0
                except ValueError:
                    return False
        return True
    return False


def _omitted_as_text(value: Any) -> Any:
    """Read an absent or null JSON field as the empty string, nothing else.

    A missing key and an explicit ``null`` both mean "not supplied", which is what
    the query-string form expresses as an empty value. Every other type is returned
    UNCHANGED so the caller's type check can refuse it — coercing here would turn a
    caller's mistake into a different question silently answered.
    """
    return "" if value is None else value


def parse_json_body(
    content_length: Optional[str], reader: Callable[[int], bytes]
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """Parse a POST body into ``(value, ok)``.

    ``ok`` is False only on malformed JSON. A missing/zero/invalid length or an
    empty body yields ``({}, True)`` so a no-payload POST (e.g. revert) still
    reaches dispatch cleanly. Pure over an injected reader so it is unit-testable
    without binding a socket.
    """
    try:
        length = int(content_length or 0)
    except (TypeError, ValueError):
        length = 0
    if length <= 0:
        return {}, True
    raw = reader(length)
    if not raw:
        return {}, True
    try:
        return json.loads(raw.decode("utf-8")), True
    except (ValueError, UnicodeDecodeError):
        return None, False


class SidecarApp:
    """Authenticated read-only request dispatcher, independent of HTTP sockets."""

    def __init__(
        self,
        service: RouterService,
        token_path: Callable[[], Path] = resolve_token_path,
        console_path: Path = _CONSOLE_PATH,
        core_config_path: Optional[Callable[[], Path]] = None,
        restart_runner: Callable[[Path], Dict[str, Any]] = _default_restart_runner,
        process_started_at: Optional[str] = _PROCESS_STARTED_AT,
        code_mtime: Optional[str] = _CODE_MTIME,
    ):
        self._service = service
        self._token_path = token_path
        self._console_path = console_path
        self._core_config_path = core_config_path or resolve_core_config_path
        self._restart_runner = restart_runner
        # Console gzip is expensive enough to hit the proxy ceiling first; cache
        # by the deployed file's version so reopening the panel does not recompress.
        self._console_gzip_cache: Optional[Tuple[int, int, bytes]] = None
        # Inject provenance into the service so /status can report it
        if process_started_at is not None:
            service._process_started_at = process_started_at
        if code_mtime is not None:
            service._code_mtime = code_mtime

    def render_console(self) -> Tuple[int, bytes, str]:
        """Return the console HTML shell as ``(status, body, content_type)``.

        Read-only and auth-exempt: it is the container the browser loads, and
        every datum it shows is fetched afterwards through the token-gated JSON
        routes. A missing file degrades to a JSON 404, never a traceback.
        """
        try:
            return 200, self._console_path.read_bytes(), "text/html; charset=utf-8"
        except OSError:
            return 404, b'{"error":"console not found"}', "application/json"

    def _gzip_console(self, body: bytes) -> bytes:
        """Return the cached gzip body for the current on-disk console version."""
        try:
            stat = self._console_path.stat()
        except OSError:
            return gzip.compress(body)
        version = (stat.st_mtime_ns, stat.st_size)
        cached = self._console_gzip_cache
        if cached is not None and cached[:2] == version:
            return cached[2]
        compressed = gzip.compress(body)
        self._console_gzip_cache = (*version, compressed)
        return compressed

    def encode_response(
        self, body: bytes, accept_encoding: Optional[str], is_console: bool = False
    ) -> Tuple[bytes, Dict[str, str]]:
        """Negotiate an HTTP response body without changing its decoded contract."""
        headers = {"Vary": "Accept-Encoding"}
        if not _accepts_gzip(accept_encoding):
            return body, headers
        encoded = self._gzip_console(body) if is_console else gzip.compress(body)
        return encoded, {**headers, "Content-Encoding": "gzip"}

    def _expected_token(self) -> TokenState:
        try:
            raw = self._token_path().read_text(encoding="utf-8")
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError, OSError):
            return TokenState(token=None, present=False)
        return TokenState(token=raw.strip(), present=True)

    def _authorize(self, headers: Dict[str, str]) -> Optional[Tuple[int, Dict[str, Any]]]:
        expected = self._expected_token()
        if not expected.present or not expected.token:
            return _error(503, "sidecar token not provisioned")
        supplied = next(
            (value for name, value in headers.items() if name.lower() == TOKEN_HEADER.lower()), None
        )
        # Compare BYTES, not str. hmac.compare_digest raises TypeError on a str with
        # any non-ASCII character, and http.server decodes header values as latin-1,
        # so a header of "café" reached this line as a non-ASCII str and the
        # TypeError escaped dispatch(). Measured against the running sidecar: the
        # client got zero bytes back — an empty reply instead of a 401 — on every
        # token-gated route. surrogateescape so an undecodable byte still compares
        # rather than raising on the way in.
        if supplied is None or not hmac.compare_digest(
            supplied.encode("utf-8", "surrogateescape"),
            expected.token.encode("utf-8", "surrogateescape"),
        ):
            return _error(401, "invalid sidecar token")
        return None

    def dispatch(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        query: Optional[Dict[str, List[str]]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        """Serve an HTTP-shaped request without binding a socket."""
        query = query or {}

        # Method-per-route check runs before auth so a wrong-method hit on a
        # known route is a 405 whether or not a token was supplied (preserves
        # the historical POST /health -> 405 and POST /status -> 405 contract).
        # Membership decides the allowed SET, not a single value: /explain is in
        # both sets, and asking "GET or POST?" would have silently 405'd whichever
        # one lost the tie.
        allowed = set()
        if path in _GET_ROUTES:
            allowed.add("GET")
        if path in _POST_ROUTES:
            allowed.add("POST")
        if allowed and method not in allowed:
            return _error(405, "method not allowed")

        # /health is the only auth-exempt data route. It reports the token's
        # state, but stays 200 in both cases ON PURPOSE: this same process
        # serves /console, and the screen must load so it can EXPLAIN the
        # failure — a 503 here would take the explanation down with the
        # problem. Whoever wants the state reads the field.
        #
        # 2026-08-26, measured: for three hours /health said ok while every
        # token-gated route answered 503 "sidecar token not provisioned", and
        # the operator's panel was blind. The field exists so that can never
        # read as healthy again. Consumers verified additive-safe on this
        # deploy: the webui proxy probes status only (200 vs not), the console
        # reads it as a truthy liveness dot, router-deploy.sh greps for
        # "ok": true, and smoke-live-sidecar.sh checks the HTTP code — none
        # asserts the payload is an exact object.
        if path == "/health":
            expected = self._expected_token()
            return 200, {
                "ok": True,
                "service": EXTENSION_ID,
                "version": _VERSION,
                "token": "present" if expected.present and expected.token else "missing",
            }

        denial = self._authorize(headers)
        if denial is not None:
            return denial

        if method == "GET":
            return self._dispatch_get(path, query)
        if method == "POST":
            return self._dispatch_post(path, body)
        # Unknown route with an unmodelled method: method check above only fired
        # for known routes, so this is genuinely not found.
        return _error(404, "unknown route")

    def _dispatch_get(
        self, path: str, query: Dict[str, List[str]]
    ) -> Tuple[int, Dict[str, Any]]:
        if path == "/status":
            return 200, self._service.status()
        if path == "/policy":
            return 200, self._service.policy()
        if path == "/blocklist":
            return 200, self._service.blocklist()
        if path == "/liveness":
            return 200, self._service.liveness()
        if path == "/capabilities":
            # The model catalogue the console's price audit reads: per model the
            # capability facts, the billing mode, the published prices (``None``
            # where nothing is published, which is NOT the same as 0.0) and the
            # declared price windows. Token-gated like every other data route, and
            # a catalogue — it carries no credential and serves an allowlisted set
            # of registry fields, never the registry entry whole.
            #
            # The console calls this on EVERY load. While the route did not exist
            # the call 404'd, and the panel did not go blank: it rendered every elo
            # as capability-unverified and every rail as publishing no per-token
            # price, which is the opposite of the truth for the metered ones. A
            # read path this cheap is not optional when its absence is a false
            # answer rather than a missing one.
            return 200, self._service.capabilities()
        if path == "/compaction":
            try:
                aggressiveness = int((query.get("aggr") or ["50"])[0])
            except (TypeError, ValueError):
                return _error(400, "aggr must be an integer between 0 and 100")
            if not 0 <= aggressiveness <= 100:
                return _error(400, "aggr must be an integer between 0 and 100")
            summarizer_window, model_windows = self._service.compaction_windows()
            threshold_fraction = p_eff(summarizer_window, aggressiveness)
            threshold_tokens = int(summarizer_window * threshold_fraction)
            # The RESOLVED compaction choice (model + fallback queue), or None
            # when the policy declares none or the block is refused. Read through
            # the same authority the RESTART-class apply uses, so this screen can
            # never show a queue the apply would refuse. The refusal itself rides
            # in `compaction_errors`, not as a 400: /compaction is a read path the
            # console opens alongside a broken config.
            aux, aux_errors = self._service.resolve_compaction()
            return 200, {
                "aggressiveness": aggressiveness,
                "model_thresholds": compute_model_thresholds(
                    model_windows.items(), aggressiveness
                ),
                "summarizer_window": summarizer_window,
                "threshold_fraction": threshold_fraction,
                "threshold_tokens": threshold_tokens,
                "warning": threshold_tokens >= summarizer_window,
                "compaction": aux if aux else None,
                "compaction_errors": aux_errors,
                # WHETHER THE CHOICE CAN BE MADE EFFECTIVE, which is part of the same
                # answer as what the choice IS.
                #
                # Writing `compaction` into router.yaml goes through the ordinary hot
                # /plan + /apply path and works. Projecting it into Hermes' own
                # `auxiliary.compression` is the RESTART-class apply, which hands a
                # candidate config to the safe-restart launcher — and on 2026-09-03 that
                # launcher was absent on BOTH installs (the WSL box and the docker
                # container). The console does not expose that step at all.
                #
                # So an operator could choose a model, press Gravar, read "Salvo", and
                # nothing about how conversations are summarised would change. The
                # refusal was honest wherever it was reached; nothing reached it. A
                # screen that reports success for a declaration it cannot enact is
                # worse than one that says which half it did.
                "effective_apply": _effective_apply_state(),
            }
        if path == "/routes":
            # Read-only route-trace access for visual replay. ?id=X fetches one
            # full trace (incl. steps[]); bare returns the recent list + the
            # resolved trace_path/count so an empty list is diagnosable. Using a
            # query param (not a /routes/{id} path segment) keeps exact-match
            # dispatch and the 405 method guard intact.
            rid = (query.get("id") or [""])[0]
            if rid:
                entry = self._service.route(rid)
                if entry is None:
                    return _error(404, "route trace not found")
                return 200, entry
            try:
                limit = int((query.get("limit") or ["50"])[0])
            except (TypeError, ValueError):
                limit = 50
            src = (query.get("source") or [""])[0]
            policy_only = (query.get("policy_only") or [""])[0].lower() in ("1", "true", "yes")
            return 200, self._service.routes(
                limit=limit, source=src or None, policy_only=policy_only)
        if path == "/outcomes":
            return 200, self._service.outcomes()
        if path == "/alignment":
            return 200, self._service.alignment()
        if path == "/alignment-panel":
            return 200, self._service.alignment_panel()
        if path == "/tier-cost":
            try:
                days = int((query.get("days") or ["7"])[0])
            except (TypeError, ValueError):
                days = 7
            return 200, self._service.tier_cost(days=days)
        if path == "/premium-budget":
            try:
                days = int((query.get("days") or ["7"])[0])
            except (TypeError, ValueError):
                days = 7
            return 200, self._service.premium_budget(days=days)
        if path == "/lint":
            return 200, self._service.lint()
        if path == "/explain":
            return self._explain(
                task=(query.get("task") or [""])[0],
                at=(query.get("at") or [""])[0],
                prompt_text=(query.get("prompt_text") or [""])[0],
            )
        return _error(404, "unknown route")

    def _explain(
        self, task: Any, at: Any, prompt_text: Any
    ) -> Tuple[int, Dict[str, Any]]:
        """Shared /explain body for GET and POST.

        Three parameters, all named exactly as ``RouterService.explain`` names them,
        because the whole point of the shared vocabulary is that a console does not
        need a per-surface translation table.

          ``task``         the goal. Required.
          ``at``           the evaluation clock, as an ISO-8601 timestamp
                           (``2026-08-17T07:00:00Z`` / ``+00:00`` / naive-UTC).
                           Empty means "now", which is the historical behaviour.
          ``prompt_text``  the composed prompt (context + goal) the turn would
                           really send, so ``est_input_tokens`` and the capability
                           filter reproduce the production decision instead of
                           measuring the goal line. Empty means "same as task".

        ``at`` is passed STRAIGHT THROUGH to the service rather than parsed here:
        ``RouterService._resolve_at`` is the one validator both surfaces share, and
        a second parser is how two surfaces end up disagreeing about what ``07:00``
        meant. It fails CLOSED exactly as the CLI does — an unusable value is a 400
        naming the accepted form, never a silent fall back to "now", because an
        audit surface that answers a different question than the one asked is worse
        than one that refuses. The CLI's bare-hour and ``HH:MM`` sugar deliberately
        stays in the CLI: choosing which DATE a bare hour belongs to is an
        interface decision, and the weekday it picks decides whether the
        weekday-gated zai peak applies.

        Non-string values (a JSON number or list in a POST body) are refused rather
        than coerced: ``at=7`` would be a bare hour this surface does not define,
        and guessing is the failure mode above.
        """
        for name, value in (("task", task), ("at", at), ("prompt_text", prompt_text)):
            if not isinstance(value, str):
                return _error(400, f"{name} must be a string")
        try:
            return 200, self._service.explain(
                task.strip(), at=at.strip() or None, prompt_text=prompt_text
            )
        except ValueError as exc:
            return _error(400, str(exc))

    def _dispatch_post(
        self, path: str, body: Optional[Dict[str, Any]]
    ) -> Tuple[int, Dict[str, Any]]:
        if not isinstance(body, dict):
            return _error(400, "request body must be a JSON object")

        if path == "/explain":
            # Same handler, same parameter names, same validation as the GET form
            # — the body is only a wider pipe for prompt_text (see _POST_ROUTES).
            # An explicit JSON null reads as "not supplied" (the query-string form
            # cannot express null at all); anything else non-string is refused by
            # _explain rather than coerced, including the 0 that `or ""` would have
            # quietly turned into "now".
            return self._explain(
                task=_omitted_as_text(body.get("task")),
                at=_omitted_as_text(body.get("at")),
                prompt_text=_omitted_as_text(body.get("prompt_text")),
            )

        if path == "/plan":
            policy = body.get("policy", body.get("changes"))
            if not isinstance(policy, dict):
                return _error(400, "plan requires a 'policy' object")
            try:
                return 200, self._service.plan(policy)
            except ValueError as exc:
                return _error(400, str(exc))

        if path == "/apply":
            # The console overloads /apply for the RESTART-class compaction
            # action, disambiguated only by body.action.
            if body.get("action") == "compaction":
                return self._apply_compaction(body)
            return self._commit_policy(body)

        # There is no /apply/confirm. It was an alias whose whole body was
        # `return self._commit_policy(body)`, justified by a comment describing
        # "the console's second-stage commit button" that console.html:2065 itself
        # records as DELETED ("two buttons, one effect" was unexplainable on
        # screen). Nothing posts it: the console posts /apply and /apply/revert,
        # plugin_api.py has no write routes, and the live smoke script never
        # touches it. An alias with no client is a second spelling of the write
        # path for an attacker to find and for a reader to wonder about.

        if path == "/apply/revert":
            return 200, self._service.apply_revert()

        return _error(404, "unknown route")

    def _commit_policy(self, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """The hash-checked policy write behind POST /apply."""
        plan = body.get("plan")
        if not isinstance(plan, dict):
            return _error(400, "apply requires the 'plan' returned by /plan")
        base_hash = plan.get("base_hash")
        policy = body.get("policy", plan.get("policy"))
        if not isinstance(base_hash, str) or not base_hash:
            return _error(400, "apply requires plan.base_hash")
        if not isinstance(policy, dict):
            return _error(400, "apply requires a 'policy' object")
        try:
            result = self._service.apply(base_hash, policy)
        except ValueError as exc:
            return _error(400, str(exc))
        if result.get("conflict"):
            # Optimistic-concurrency drift: someone wrote router.yaml since the
            # plan was computed. 409 lets the UI re-plan against fresh state.
            return 409, result
        if not result.get("ok"):
            return 400, result
        return 200, result

    def _apply_compaction(self, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """RESTART-class: recompute compaction thresholds and hand a candidate
        config.yaml to the safe-restart dead-man switch.

        This does NOT ride RouterService.apply (that is the router.yaml hot
        path) and never writes config.yaml inline — the dead-man switch owns the
        mutation and the health-gated auto-revert. Requires a server-side
        type-to-confirm mirroring the console's client gate.
        """
        if body.get("confirm") != _COMPACTION_CONFIRM:
            return _error(400, f"compaction requires confirm={_COMPACTION_CONFIRM}")
        aggressiveness = body.get("aggressiveness", 50)
        if not isinstance(aggressiveness, int) or not 0 <= aggressiveness <= 100:
            return _error(400, "aggressiveness must be an integer between 0 and 100")

        core_path = self._core_config_path()
        try:
            current = yaml.safe_load(core_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            return _error(400, f"could not read core config: {exc}")
        if not isinstance(current, dict):
            return _error(400, "core config root must be a mapping")

        # The SAME source `/compaction` reads from, so the screen the operator
        # confirmed cannot show thresholds this write does not produce.
        summarizer_window, model_windows = self._service.compaction_windows()
        candidate = apply_dynamic_thresholds(
            current, aggressiveness, summarizer_window, model_windows
        )
        # Resolve the declarative compaction choice (model + fallback queue) from
        # the LIVE router policy and merge it under auxiliary.compression. One
        # authority (RouterService.resolve_compaction) does the resolution for both
        # the write-path lint and this RESTART-class apply, so a tier deleted after
        # the block was written is refused BY NAME here rather than compressing on
        # a queue that no longer exists. The merge only sets provider/model/
        # base_url/fallback_chain — everything else the operator keeps in
        # auxiliary (api_key, timeout, extra_body, sibling auxiliary.<task>
        # sections) survives untouched.
        aux, aux_errors = self._service.resolve_compaction()
        if aux_errors:
            return _error(400, "; ".join(aux_errors))
        if aux:
            aux_section = candidate.get("auxiliary")
            if aux_section is None:
                aux_section = {}
                candidate["auxiliary"] = aux_section
            elif not isinstance(aux_section, dict):
                return _error(400, "auxiliary in core config must be a mapping")
            compression = aux_section.get("compression")
            if compression is None:
                compression = {}
                aux_section["compression"] = compression
            elif not isinstance(compression, dict):
                return _error(400, "auxiliary.compression in core config must be a mapping")
            compression.update(aux)
        # Write the candidate to a temp file for the launcher to validate + apply.
        fd, tmp_path = tempfile.mkstemp(suffix=".yaml", prefix="compaction-candidate-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                yaml.safe_dump(candidate, handle, sort_keys=False)
            result = self._restart_runner(Path(tmp_path))
        except OSError as exc:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            return _error(500, f"could not stage candidate config: {exc}")
        if not result.get("ok"):
            return 502, result
        # 202 Accepted: the restart is scheduled and health-gated; it has not
        # necessarily completed when this returns.
        return 202, {**result, "aggressiveness": aggressiveness}


def _make_handler(app: SidecarApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _write(
            self, status: int, body: bytes, content_type: str, is_console: bool = False
        ) -> None:
            encoded, response_headers = app.encode_response(
                body, self.headers.get("Accept-Encoding"), is_console
            )
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            for name, value in response_headers.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(encoded)))
            # NADA daqui pode ser cacheado, e este é o único ponto por onde toda
            # resposta passa — console e rotas JSON.
            #
            # O console porque um deploy que não aparece é pior que um deploy que
            # falha: em 2026-08-26 o operador continuou vendo a tela antiga depois
            # de um deploy verificado, e a causa foi esta — resposta 200 sem
            # Cache-Control, sem ETag e sem Last-Modified, buscada com a política
            # default do fetch. As rotas JSON pelo mesmo motivo mais forte ainda:
            # /status, /policy e /liveness são estado VIVO, e uma cópia cacheada
            # delas é exatamente a classe de mentira que o DESIGN.md combate na
            # tela (um número velho apresentado como agora).
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def _serve(self, method: str, body: Optional[Dict[str, Any]] = None) -> None:
            parts = urlsplit(self.path)
            # The console is a static HTML shell, served outside the JSON path.
            if method == "GET" and parts.path == "/console":
                status, html, content_type = app.render_console()
                self._write(status, html, content_type, is_console=True)
                return
            status, payload = app.dispatch(
                method,
                parts.path,
                {name: value for name, value in self.headers.items()},
                parse_qs(parts.query),
                body,
            )
            self._write(status, json.dumps(payload).encode("utf-8"), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            self._serve("GET")

        def do_POST(self) -> None:  # noqa: N802
            parsed, ok = parse_json_body(
                self.headers.get("Content-Length"), self.rfile.read
            )
            if not ok:
                self._write(
                    400,
                    json.dumps({"error": "request body is not valid JSON"}).encode("utf-8"),
                    "application/json",
                )
                return
            self._serve("POST", parsed)

    return Handler


def build_server(host: str, port: int, app: SidecarApp) -> ThreadingHTTPServer:
    """Build a server only on a loopback address; never expose the sidecar."""
    if host not in _LOOPBACK_HOSTS:
        raise ValueError("sidecar host must be loopback")
    return ThreadingHTTPServer((host, port), _make_handler(app))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument(
        "--config",
        type=Path,
        # The SAME resolver the plugin uses, so a sidecar started bare cannot serve a
        # different policy than the one dispatch routes on. This was
        # `parent.parent / "router.yaml"` — the plugin directory — which is precisely the
        # assumption that split the two on the docker stack (see
        # router.paths.resolve_policy_path for the measurement). Both installs pass
        # --config explicitly, so this default was never what bit; two answers to one
        # question was.
        default=resolve_policy_path(Path(__file__).resolve().parent.parent),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    app = SidecarApp(RouterService(args.config))
    server = build_server(args.host, args.port, app)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
