"""Hoisted AgentCore Code Interpreter wrapper — preserves kernel state across calls.

Critical for ML work: sklearn models, DataFrames, imports must survive between
tool invocations. The naive `with code_session(region) as client: ...` pattern from
the AgentCore sample opens/closes the session per call, losing kernel state.

We hoist the context manager into a module-level singleton with `atexit` teardown
(MetaGPT `ExecuteNbCode` shape applied to AgentCore).

Canonical reference:
    amazon-bedrock-agentcore-samples/01-tutorials/05-AgentCore-tools/
      01-Agent-Core-code-interpreter/02-code-execution-with-agent-using-code-interpreter/
      strands-agent-code-execution-code-interpreter.ipynb

Injection hook: `configure_session_factory(callable)` injects a fake session factory so
tests never need the real bedrock-agentcore SDK.
"""

from __future__ import annotations

import atexit
import json
import os
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from typing import Any

from ..config.aws_region import resolve_aws_region

SessionFactory = Callable[[str], AbstractContextManager[Any]]

# SEC-A15 (OWASP LLM05/LLM06): the AgentCore Code Interpreter executes agent/model-staged
# Python. The sandbox is platform-isolated, but network EGRESS is a runtime/infra posture —
# this env seam lets a deployment assert "egress disabled" as the secure default (prod) and
# opt IN deliberately only where outbound network is genuinely required. The actual egress
# enforcement is an AgentCore/VPC runtime control; this seam is the single source of truth the
# code path + IaC read so the posture is explicit and testable.
_SANDBOX_EGRESS_ENV = "AMLAS_SANDBOX_EGRESS"
_TRUE_TOKENS = frozenset({"1", "true", "yes", "on"})

# Subtask-13 kernel keep-alive: the SDK's ``code_session`` hardcodes a 900s (15-min)
# session timeout, so a 30-min / 50-trial cloud run idle-expires the kernel mid-run
# (subtask-7 §4, subtask-10 §5). We open the hoisted session with a LONG timeout so a
# full run fits inside one session — the timeout IS the keep-alive. Overridable via
# ``AMLAS_SANDBOX_SESSION_TIMEOUT_SECONDS`` (ops knob); default 3600s. Capped at the
# service max the SDK accepts (28800s / 8h) so a bad env value can't be rejected.
_SESSION_TIMEOUT_ENV = "AMLAS_SANDBOX_SESSION_TIMEOUT_SECONDS"
_DEFAULT_SESSION_TIMEOUT_SECONDS = 3600
_MAX_SESSION_TIMEOUT_SECONDS = 28800

# Subtask-13 auto-reprovision: substrings (case-insensitive) in an ``invoke`` exception
# that mean "the kernel session is gone/stuck and a FRESH session would recover" — as
# opposed to an ordinary transport fault or a user-code error (which must NOT reprovision).
# Sourced from the REAL messages the runs surfaced (string-label classification-cw-runtime.log:597/:1144)
# plus the botocore/AgentCore error shapes for a lost/inactive/throttled session.
_KERNEL_RECOVERABLE_TOKENS: tuple[str, ...] = (
    "expired",
    "stuck",
    "session not found",
    "sessionnotfound",
    "resourcenotfound",
    "invalidsession",
    "no active session",
    "session is inactive",
    "session is not active",
    "session has been",
    "session was terminated",
    "session terminated",
    "throttl",
    # FIX-27: transient CAPACITY/QUOTA conditions are recoverable — they clear after a wait
    # (handled with exponential backoff + jitter, NOT an immediate reprovision-retry).
    "servicequota",
    "service quota",
    "quotaexceeded",
    "quota exceeded",
    "limitexceeded",
    "limit exceeded",
    "too many requests",
    "toomanyrequests",
    "capacity",
    "provisionedthroughputexceeded",
)

# FIX-27: the SUBSET of recoverable signatures that are TRANSIENT-CAPACITY (quota/throttle)
# rather than a dead kernel. These are retried with EXPONENTIAL BACKOFF + JITTER (wait, then
# retry) instead of an immediate reprovision — an instant retry just re-collides on the
# still-exhausted quota (root cause of the two-concurrent-runs runtime-wide stall).
_QUOTA_TOKENS: tuple[str, ...] = (
    "servicequota",
    "service quota",
    "quotaexceeded",
    "quota exceeded",
    "limitexceeded",
    "limit exceeded",
    "too many requests",
    "toomanyrequests",
    "throttl",
    "capacity",
    "provisionedthroughputexceeded",
)

# Subtask-13 FIX-20 — y_pred write-through mirror. Auto-reprovision (Part A) keeps the RUN
# alive across a kernel-expired/quota error, but a FRESH kernel has an EMPTY sandbox fs — so
# a y_pred the agent already wrote to the PRIOR kernel's /tmp is LOST before the Arbiter reads
# it (string-label classification run #4: agent hit CV f1_macro=0.80008 + wrote y_pred, then ServiceQuotaExceeded
# → KERNEL_REPROVISION wiped /tmp → 4 iters, no grade, budget_exhausted). Fix: the instant a
# sandbox invoke that TOUCHES y_pred succeeds (kernel still healthy), opportunistically egress
# the canonical y_pred file OUT to the runtime PROCESS fs. ``load_predictions``/``_y_pred_loader``
# already prefer the process-fs copy (``p.exists()`` short-circuits egress), so the Arbiter grades
# the mirrored copy even after a later reprovision. Overridable via ``AMLAS_YPRED_MIRROR`` (default
# ON; set 0/false/no/off to disable). The canonical sandbox paths the ML Lead's trainer_tuner writes.
_YPRED_MIRROR_ENV = "AMLAS_YPRED_MIRROR"
_YPRED_SANDBOX_CANDIDATES: tuple[str, ...] = (
    "/tmp/amlas/y_pred_test.npy",  # nosec B108 — sandbox path, not a host tempfile
    "/tmp/amlas/y_pred_test.csv",  # nosec B108
)
# Re-entrancy guard: the mirror egresses via ``execute_python`` (→ back through this seam), and
# the read-file code embeds the y_pred path (so it "touches y_pred" too). Without this flag the
# mirror would recurse infinitely. It is set for the duration of a mirror attempt so the nested
# egress invoke is a plain passthrough.
_mirroring: bool = False

# Subtask-13 FIX-22 — input re-staging on reprovision (the mirror-IN counterpart of FIX-20's
# mirror-OUT). Auto-reprovision (Part A) opens a FRESH kernel whose /tmp is EMPTY. FIX-20 saves
# the OUTPUT (y_pred) out to the process fs, but NOTHING re-staged the INPUT dataset back into
# the fresh sandbox — so post-reprovision the trainer's ``pd.read_csv(<staged path>)`` raises
# FileNotFoundError, never fits, and never writes y_pred (string-label classification runs #7-#10: KERNEL_REPROVISION
# then YPRED_MIRROR=0 / arbiter_report=absent, the agent's own log: "same FileNotFoundError").
# Fix: remember every file the run staged pre-flight, and REPLAY those writes into the fresh
# kernel immediately after a reprovision (before the retried invoke). Overridable via
# ``AMLAS_RESTAGE_INPUTS`` (default ON; set 0/false/no/off to disable).
_RESTAGE_INPUTS_ENV = "AMLAS_RESTAGE_INPUTS"
# sandbox_target_path -> local_source_path, recorded by ``register_staged_file`` (called from
# ``sandbox_staging.stage_files_into_sandbox`` on each successful stage). Module-level so it
# survives across invokes within one run; cleared on ``shutdown_session`` reset is NOT done (the
# same inputs are valid for the whole run, and the fresh kernel needs exactly them).
_STAGED_FILES: dict[str, str] = {}
# Re-entrancy guard: re-staging replays writes via ``execute_python`` (→ back through this seam).
# Without this flag a nested recoverable error during re-staging would recurse infinitely.
_restaging: bool = False


def register_staged_file(sandbox_target_path: str, local_source_path: str) -> None:
    """Record a ``{sandbox_target: local_source}`` pair so FIX-22 can re-stage it on reprovision.

    Called by ``sandbox_staging.stage_files_into_sandbox`` for each file it successfully writes
    into the sandbox. Idempotent (latest local source wins). Kept here (not in sandbox_staging)
    so ``_invoke_with_reprovision`` can replay it without importing sandbox_staging at module
    load (that module imports ``execute_python`` from here → cycle; the replay uses a lazy
    import of ``build_write_file_code``, mirroring FIX-20's egress import).
    """
    _STAGED_FILES[sandbox_target_path] = local_source_path


def sandbox_egress_enabled() -> bool:
    """True only if sandbox network egress is EXPLICITLY enabled (SEC-A15; default OFF).

    Reads ``AMLAS_SANDBOX_EGRESS`` (standard truthy tokens, case-insensitive). Default
    **disabled** — the secure prod posture for model-staged code execution. An operator
    opts in deliberately where a sandbox genuinely needs outbound network.
    """
    return os.environ.get(_SANDBOX_EGRESS_ENV, "").strip().lower() in _TRUE_TOKENS


def _session_timeout_seconds() -> int:
    """Sandbox session timeout (s) — the keep-alive lever (Subtask-13).

    Reads ``AMLAS_SANDBOX_SESSION_TIMEOUT_SECONDS`` (default 3600, i.e. 1h — far longer
    than the SDK's 900s default so a 30-min run never idle-expires the kernel). Invalid
    / out-of-range values fall back to the default; clamped to the service max (8h).
    """
    raw = os.environ.get(_SESSION_TIMEOUT_ENV, "").strip()
    if not raw:
        return _DEFAULT_SESSION_TIMEOUT_SECONDS
    try:
        val = int(raw)
    except ValueError:
        return _DEFAULT_SESSION_TIMEOUT_SECONDS
    if val <= 0:
        return _DEFAULT_SESSION_TIMEOUT_SECONDS
    return min(val, _MAX_SESSION_TIMEOUT_SECONDS)


def _is_kernel_recoverable_error(exc: BaseException) -> bool:
    """True if ``exc`` from a kernel ``invoke`` means "session gone/stuck → reprovision".

    Deliberately NARROW: matches only expiry/stuck/lost-session/throttle/quota signatures so an
    ordinary transport fault or a user-code error is NOT masked by a needless reprovision.
    """
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(tok in msg for tok in _KERNEL_RECOVERABLE_TOKENS)


def _is_quota_error(exc: BaseException) -> bool:
    """True if ``exc`` is a TRANSIENT capacity/quota/throttle condition (FIX-27).

    The emitter is the Bedrock AgentCore Code Interpreter's ``invoke("executeCode")`` (or the
    session ``start``): rapid kernel-reprovision churn — compounded by concurrent runs — exhausts
    the account's CONCURRENT CODE-INTERPRETER SESSION quota → ``ServiceQuotaExceededException``.
    Unlike a dead kernel (reprovision + retry once, no wait), a quota error clears only after a
    WAIT, so it is retried with exponential backoff + jitter (never an instant re-collide).
    """
    msg = f"{type(exc).__name__}: {exc}".lower()
    return any(tok in msg for tok in _QUOTA_TOKENS)


# FIX-27 backoff knobs (env-tunable; sleep is injectable for hermetic tests).
_QUOTA_MAX_RETRIES_ENV = "AMLAS_QUOTA_MAX_RETRIES"
_QUOTA_BACKOFF_BASE_ENV = "AMLAS_QUOTA_BACKOFF_BASE_SECONDS"
_QUOTA_BACKOFF_CAP_ENV = "AMLAS_QUOTA_BACKOFF_CAP_SECONDS"
_DEFAULT_QUOTA_MAX_RETRIES = 6
_DEFAULT_QUOTA_BACKOFF_BASE = 3.0
_DEFAULT_QUOTA_BACKOFF_CAP = 60.0
# Monotonic counter to vary jitter deterministically WITHOUT a real RNG (Math.random/seed-free —
# resume-safe, and so two lockstep sessions still de-sync). Incremented per delay computation.
_quota_jitter_counter: int = 0


def _env_int(name: str, default: int) -> int:
    try:
        v = int(os.environ.get(name, "").strip() or default)
        return v if v >= 0 else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        v = float(os.environ.get(name, "").strip() or default)
        return v if v > 0 else default
    except ValueError:
        return default


def _quota_sleep(seconds: float) -> None:
    """Injectable wait between quota retries (overrideed in tests → no real sleeping)."""
    import time

    time.sleep(seconds)  # nosemgrep: arbitrary-sleep


def _quota_backoff_delay(attempt: int, base: float, cap: float) -> float:
    """Full-jitter exponential backoff: uniform(0, min(cap, base*2**attempt)] (FIX-27).

    Full-jitter (AWS's recommended strategy) spreads concurrent retriers across the whole
    window so they do not re-collide in lockstep on the shared quota. Jitter is derived from a
    process counter + PID (not a seeded RNG) so it is resume-safe and still de-synchronizes
    parallel sessions. Always returns a positive delay bounded by ``cap``.
    """
    global _quota_jitter_counter
    nominal: float = min(float(cap), float(base) * float(2 ** max(0, attempt)))
    _quota_jitter_counter += 1
    # A cheap, dependency-free pseudo-fraction in (0, 1] from the counter + pid.
    seed = (_quota_jitter_counter * 2654435761 + os.getpid() * 40503) % 10_000
    frac: float = (seed + 1) / 10_001.0  # in (0, 1)
    return max(0.001, nominal * frac)

# --- Module-level singleton state ---
_session_factory: SessionFactory | None = None
_ci_ctx: AbstractContextManager[Any] | None = None
_ci_client: Any = None
_ci_region: str | None = None
_atexit_registered: bool = False


class CodeInterpreterError(RuntimeError):
    """Raised on SDK-level failures (region unavailable, session start failed, etc.).

    Note: *within-code* errors (Python exceptions thrown by user code) are NOT raised —
    they surface as `CodeExecResult(is_error=True, ...)` so the Strands agent loop can
    reinject them as toolResult errors per the canonical reinjection contract.
    """


@dataclass(frozen=True)
class CodeExecResult:
    """Canonical result from one execute_python call. Mirrors AgentCore shape."""

    session_id: str
    id: str
    is_error: bool
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    content: list[dict[str, Any]] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Joined text content blocks, one per block."""
        return "\n".join(
            c.get("text", "") for c in self.content if c.get("type") == "text"
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_session_factory(factory: SessionFactory | None) -> None:
    """Install a session factory. Tests call this; production leaves it None.

    Resets any existing singleton so the new factory is used on next execute_python().
    Pass `None` to revert to the default (imports `bedrock_agentcore`).
    """
    global _session_factory
    shutdown_session()
    _session_factory = factory


def shutdown_session() -> None:
    """Close the hoisted session (if any). Idempotent."""
    global _ci_ctx, _ci_client, _ci_region
    if _ci_ctx is not None:
        try:
            _ci_ctx.__exit__(None, None, None)
        except Exception:  # pragma: no cover — best-effort teardown  # nosec B110
            pass
    _ci_ctx = None
    _ci_client = None
    _ci_region = None


def execute_python(
    code: str,
    *,
    description: str = "",
    region: str | None = None,
    clear_context: bool = False,
    require_approval: bool = False,
    approved: bool = True,
) -> CodeExecResult:
    """Execute Python in the hoisted sandbox.

    `clear_context` defaults to False — kernel state (imports, DataFrames, models) is
    preserved between calls. Set True only to explicitly reset (rarely useful — prefer
    `shutdown_session()` + next call re-opens).

    SEC-A15 approval gate: when ``require_approval`` is True the caller MUST also pass
    ``approved=True`` (the control-plane decision) or the execution is REFUSED — the
    model/agent-staged code never opens a session or invokes the kernel, and a structured
    ``CodeExecResult(is_error=True)`` is returned for the agent loop to reinject. The
    default (``require_approval=False``) preserves the existing behavior: the platform-
    isolated, egress-disabled small-sandbox is auto-approved under the approval taxonomy
    (``CHEAP_READ`` "small-sandbox"). This keeps the gating decision in the control plane
    (no approvals import here → no circular dependency).

    Never raises on user-code exceptions — those propagate via `CodeExecResult.is_error`.
    Only raises `CodeInterpreterError` for SDK-level failures (session can't open, etc.).
    """
    # SEC-A15: refuse model/agent-staged code execution that has not been approved. Do this
    # BEFORE opening any session or invoking the kernel so unapproved code truly never runs.
    if require_approval and not approved:
        return CodeExecResult(
            session_id="",
            id="",
            is_error=True,
            stdout="",
            stderr="code execution refused: approval required and not granted (SEC-A15)",
            exit_code=1,
            execution_time=0.0,
            content=[{"type": "text",
                      "text": "Code Interpreter execution refused — approval gate not satisfied"}],
        )

    if description:
        code = f"# {description}\n{code}"

    payload = {"code": code, "language": "python", "clearContext": clear_context}

    try:
        response = _invoke_with_reprovision(region, payload)
    except CodeInterpreterError:
        # Session-open / region-switch failures are SDK-level control faults that MUST
        # propagate (the caller decides) — never mask them as an in-band structured error.
        raise
    except Exception as e:
        # SDK-level fault → structured error for the agent loop to see + reinject
        return CodeExecResult(
            session_id="",
            id="",
            is_error=True,
            stdout="",
            stderr=f"{type(e).__name__}: {e}",
            exit_code=1,
            execution_time=0.0,
            content=[{"type": "text", "text": f"Code Interpreter invoke failed: {e}"}],
        )

    for event in response.get("stream", []):
        return _parse_result(event.get("result", {}))

    return CodeExecResult(
        session_id="",
        id="",
        is_error=True,
        stdout="",
        stderr="empty stream",
        exit_code=1,
        execution_time=0.0,
        content=[{"type": "text", "text": "Code Interpreter returned no events"}],
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _default_factory(region: str) -> AbstractContextManager[Any]:
    """Default factory — lazy-imports the real SDK so tests don't need it installed.

    Subtask-13 keep-alive: the SDK's ``code_session`` hardcodes a 900s session timeout,
    which idle-expires on a 30-min run. We construct ``CodeInterpreter`` directly and
    ``start`` it with ``_session_timeout_seconds()`` (default 3600s) so a full run fits
    inside one session, wrapping it in a CM that ``stop``s on exit (same lifecycle as
    ``code_session``). If the long-timeout start is unsupported by the SDK, fall back to
    the stock ``code_session`` (auto-reprovision still recovers a mid-run expiry).
    """
    from bedrock_agentcore.tools.code_interpreter_client import CodeInterpreter, code_session

    timeout = _session_timeout_seconds()

    class _LongLivedSession(AbstractContextManager[Any]):
        def __enter__(self) -> Any:
            self._client = CodeInterpreter(region)
            self._client.start(session_timeout_seconds=timeout)
            return self._client

        def __exit__(self, *exc: Any) -> None:
            try:
                # bedrock-agentcore 1.x annotates ``stop()``; the previous
                # ``type: ignore[no-untyped-call]`` became an unused-ignore error
                # once the SDK moved off the 0.x line (CVE-2026-16796 bump).
                self._client.stop()
            except Exception:  # pragma: no cover — best-effort teardown  # nosec B110
                pass

    try:
        # Probe that the SDK exposes the timeout-aware start (it does in the pinned
        # version). Constructing the CM is cheap; the real start happens on __enter__.
        return _LongLivedSession()
    except Exception:  # pragma: no cover — defensive fallback to stock behavior
        return code_session(region)


def _invoke_with_quota_backoff(
    region: str, payload: dict[str, Any], *, first_error: BaseException
) -> dict[str, Any]:
    """Retry ``executeCode`` through a TRANSIENT quota error with exponential backoff + jitter.

    FIX-27: the emitter is the Bedrock AgentCore Code-Interpreter concurrent-session quota.
    Each attempt: WAIT a full-jittered exponential delay, drop the (quota-blocked) session so
    its slot releases, open a fresh one, re-stage inputs (FIX-22), and retry. Bounded by
    ``AMLAS_QUOTA_MAX_RETRIES`` — after which the last error propagates to the structured-error
    path (a clean failure, never an infinite loop or a multi-hour socket hang). Non-quota errors
    on a retry propagate immediately (the caller's classifier already vetted the first one).
    """
    max_retries = _env_int(_QUOTA_MAX_RETRIES_ENV, _DEFAULT_QUOTA_MAX_RETRIES)
    base = _env_float(_QUOTA_BACKOFF_BASE_ENV, _DEFAULT_QUOTA_BACKOFF_BASE)
    cap = _env_float(_QUOTA_BACKOFF_CAP_ENV, _DEFAULT_QUOTA_BACKOFF_CAP)
    last_exc: BaseException = first_error
    for attempt in range(max_retries):
        delay = _quota_backoff_delay(attempt, base, cap)
        print(
            f"QUOTA_BACKOFF: {type(last_exc).__name__} (AgentCore CodeInterpreter session quota) "
            f"— attempt {attempt + 1}/{max_retries}, waiting {delay:.1f}s before retry",
            flush=True,
        )
        _quota_sleep(delay)
        # Release the quota-blocked session's slot before grabbing a new one, then re-stage.
        shutdown_session()
        fresh = _get_client(region)
        _restage_inputs_into_fresh_kernel(region)
        try:
            return fresh.invoke("executeCode", payload)  # type: ignore[no-any-return]
        except Exception as e:
            last_exc = e
            if not _is_quota_error(e):
                # A different failure now (dead kernel / transport) — stop backing off and let
                # it surface; the caller's structured-error path handles it.
                raise
    # Exhausted the bounded retries — surface the last quota error cleanly.
    raise last_exc


def _invoke_with_reprovision(region: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Invoke ``executeCode`` on the hoisted client with kernel auto-reprovision.

    Subtask-13: on a kernel-expired/stuck ``invoke`` error (the dominant long-run ceiling
    — subtask-7 §4, subtask-10 §5), drop the dead session, open a FRESH one, and retry the
    SAME execution EXACTLY ONCE. Because every sandbox call (data staging, the agents'
    ``execute_python_tool``, AND the Arbiter's ``y_pred`` egress) funnels through here, a
    long run can ALWAYS grade — the grade is never lost to a stale kernel. Retry-once-only
    (a local flag, not a module global) guarantees a permanently-dead kernel can't loop.
    Non-kernel errors (transport faults, user-code errors) are NOT reprovisioned — they
    propagate to ``execute_python``'s structured-error path unchanged (no regression).
    """
    client = _get_client(region)
    try:
        response: dict[str, Any] = client.invoke("executeCode", payload)
    except Exception as e:
        if not _is_kernel_recoverable_error(e):
            raise
        if _is_quota_error(e):
            # FIX-27: TRANSIENT capacity/quota (ServiceQuotaExceeded on the AgentCore
            # Code-Interpreter concurrent-session quota). An INSTANT reprovision-retry just
            # re-collides on the still-exhausted quota (root cause of the two-concurrent-runs
            # runtime-wide stall). WAIT with exponential backoff + full jitter, then retry —
            # bounded by AMLAS_QUOTA_MAX_RETRIES so it can never loop forever.
            response = _invoke_with_quota_backoff(region, payload, first_error=e)
        else:
            # Dead/stale kernel — reprovision + retry ONCE (no wait; the session is simply gone).
            # Metadata-only marker (never data/predictions) so the reprovision is provable
            # server-side in CloudWatch — mirrors the ARBITER_GRADED/TAVILY_SECRET markers.
            print(
                f"KERNEL_REPROVISION: recoverable error ({type(e).__name__}) — "
                f"dropping dead session, opening fresh one, retrying once",
                flush=True,
            )
            shutdown_session()
            fresh = _get_client(region)
            # FIX-22: the fresh kernel has an EMPTY /tmp — re-stage the run's INPUT files into it
            # BEFORE the retry so the retried invoke (and every subsequent trainer/FE invoke) can
            # read the dataset again. Without this, the retried invoke's pd.read_csv raises
            # FileNotFoundError, no model is fit, and no y_pred is ever produced (runs #7-#10).
            _restage_inputs_into_fresh_kernel(region)
            # Retry EXACTLY once on the fresh session; if it fails again, let it propagate to
            # the structured-error path (no infinite loop).
            response = fresh.invoke("executeCode", payload)
    # FIX-20 write-through: the invoke SUCCEEDED, so the kernel is healthy RIGHT NOW. If this
    # invoke's code TOUCHED y_pred (likely just wrote it), mirror it out to the durable process
    # fs BEFORE a later invoke can trigger a reprovision that wipes the sandbox /tmp.
    _mirror_ypred_if_present(region, payload.get("code", ""))
    return response


def _ypred_mirror_enabled() -> bool:
    """True unless ``AMLAS_YPRED_MIRROR`` is explicitly falsy (default ON — FIX-20)."""
    raw = os.environ.get(_YPRED_MIRROR_ENV, "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _code_touched_ypred(code: str) -> bool:
    """True if ``code`` references a canonical y_pred path — the invoke likely wrote it.

    Targeting only y_pred-touching invokes keeps the mirror cheap: it does NOT add an egress
    roundtrip after every probe/CV call (which would itself add Code-Interpreter quota pressure
    — the very ServiceQuotaExceeded trigger we are mitigating), only after the write.
    """
    return "y_pred_test" in code or "y_pred" in code


def _mirror_ypred_if_present(region: str, code: str = "") -> None:
    """Best-effort write-through: egress the canonical y_pred file(s) sandbox → process fs (FIX-20).

    Runs after a SUCCESSFUL sandbox invoke whose code TOUCHED y_pred (kernel is healthy) so a
    y_pred written to the CURRENT kernel is durably copied to the runtime PROCESS fs. If the
    sandbox is later reprovisioned (empty /tmp), ``load_predictions`` still finds the mirrored
    copy and the Arbiter can grade. Always OVERWRITES the mirror (write-through = latest wins),
    so a refined y_pred on a later iteration replaces the stale one. Guarded by a re-entrancy
    flag (the egress itself invokes the kernel → this seam), an env toggle, and a blanket
    try/except — it NEVER affects the primary invoke's result or raises.
    """
    global _mirroring
    if _mirroring or not _ypred_mirror_enabled() or not _code_touched_ypred(code):
        return
    _mirroring = True
    try:
        # Lazy import (cloud.sandbox_staging imports execute_python from here → cycle).
        from ..cloud.sandbox_staging import egress_file_from_sandbox

        for candidate in _YPRED_SANDBOX_CANDIDATES:
            try:
                if egress_file_from_sandbox(candidate, candidate, region=region):
                    print(f"YPRED_MIRROR: mirrored {candidate} sandbox→process fs", flush=True)
            except Exception:  # pragma: no cover — best-effort per candidate  # nosec B112
                continue
    except Exception:  # pragma: no cover — mirror must never break the primary invoke  # nosec B110
        pass
    finally:
        _mirroring = False


def _restage_inputs_enabled() -> bool:
    """True unless ``AMLAS_RESTAGE_INPUTS`` is explicitly falsy (default ON — FIX-22)."""
    raw = os.environ.get(_RESTAGE_INPUTS_ENV, "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _restage_inputs_into_fresh_kernel(region: str) -> None:
    """Best-effort: re-write every registered INPUT file into the freshly-reprovisioned kernel (FIX-22).

    Runs inside ``_invoke_with_reprovision`` right after ``shutdown_session()`` + a fresh
    ``_get_client`` — the new kernel's /tmp is empty, so the dataset the run staged pre-flight is
    gone. We replay each ``{sandbox_target: local_source}`` from ``_STAGED_FILES`` (recorded by
    ``register_staged_file``) by re-reading the LOCAL source (still on the process fs) and running
    the SAME ``build_write_file_code`` write. This restores the inputs so the retried invoke — and
    every subsequent trainer/FE invoke until the next reprovision (which re-stages again) — can
    ``pd.read_csv`` the dataset and actually fit a model + write y_pred. Re-entrancy-guarded (the
    replay invokes the kernel → back through ``_invoke_with_reprovision``), env-toggled, and
    blanket try/except — it NEVER raises into the reprovision path (a failed re-stage degrades to
    the pre-FIX-22 behavior, i.e. the retried invoke may still FileNotFound, not a crash)."""
    global _restaging
    if _restaging or not _restage_inputs_enabled() or not _STAGED_FILES:
        return
    _restaging = True
    try:
        # Lazy import (cloud.sandbox_staging imports execute_python from here → cycle),
        # mirroring FIX-20's egress import.
        from pathlib import Path

        from ..cloud.sandbox_staging import build_write_file_code

        restaged = 0
        for target_path, source_path in list(_STAGED_FILES.items()):
            try:
                src = Path(source_path)
                if not src.exists():
                    continue
                code = build_write_file_code(target_path, src.read_bytes())
                # Direct client invoke (NOT execute_python) so a re-stage failure cannot itself
                # trigger another reprovision recursion — best-effort passthrough.
                _get_client(region).invoke("executeCode", {"code": code})
                restaged += 1
            except Exception:  # pragma: no cover — per-file best-effort  # nosec B112
                continue
        if restaged:
            print(
                f"RESTAGE_INPUTS: re-staged {restaged}/{len(_STAGED_FILES)} input file(s) "
                f"into the fresh kernel after reprovision (FIX-22)",
                flush=True,
            )
    except Exception:  # pragma: no cover — re-staging must never break the reprovision retry  # nosec B110
        pass
    finally:
        _restaging = False


def _get_client(region: str | None) -> Any:
    region = resolve_aws_region(region)
    global _ci_ctx, _ci_client, _ci_region, _atexit_registered
    if _ci_client is not None:
        if _ci_region != region:
            raise CodeInterpreterError(
                f"Code Interpreter session already open in region {_ci_region!r}; "
                f"requested {region!r}. Call shutdown_session() first to switch regions."
            )
        return _ci_client

    factory = _session_factory or _default_factory
    try:
        ctx = factory(region)
        client = ctx.__enter__()
    except Exception as e:
        _ci_ctx = None
        _ci_client = None
        raise CodeInterpreterError(
            f"Failed to open Code Interpreter session in {region}: {e}"
        ) from e

    _ci_ctx = ctx
    _ci_client = client
    _ci_region = region

    if not _atexit_registered:
        atexit.register(shutdown_session)
        _atexit_registered = True

    return _ci_client


def _parse_result(result: dict[str, Any]) -> CodeExecResult:
    structured = result.get("structuredContent") or {}
    return CodeExecResult(
        session_id=str(result.get("sessionId", "")),
        id=str(result.get("id", "")),
        is_error=bool(result.get("isError", False)),
        stdout=str(structured.get("stdout", "")),
        stderr=str(structured.get("stderr", "")),
        exit_code=int(structured.get("exitCode", 0)),
        execution_time=float(structured.get("executionTime", 0.0)),
        content=list(result.get("content", [])),
    )
