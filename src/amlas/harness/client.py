"""``InvokeHarness`` event-stream client for harness-hosted Leads (ADR-0007 phase 1).

Three measured facts shape this module, and each one is a trap if ignored:

1. **``validationException`` arrives as a stream EVENT, not an API exception.** A reader that
   only guards the ``invoke_harness`` call itself sees a "successful" invocation whose content
   happens to be an error object. It is treated here as a TERMINAL ERROR and funnelled into
   ``ERROR_LOG`` with ``source="harness.<name>"`` (T-HARN-03.2 / T-HARN-10.3).
2. **Consume incrementally and stop at the terminal event.** This is the same failure mode
   FIX-14 hit on ``invoke_agent_runtime``: the body stays open past the last useful event, so
   draining to exhaustion blocks for the whole timeout. See
   [`cloud/dispatch.py`](../cloud/dispatch.py:343) for the precedent (T-HARN-10.1).
3. **A new ``runtimeSessionId`` costs ~55s of microVM cold start; a reused one costs ~1.7s.**
   The cost attaches to the SESSION, not the harness, so a lead invoked with a fresh id every
   orchestration iteration pays it every time — minutes across 7 leads by N iterations.
   :class:`HarnessSessions` makes reuse the default (T-HARN-20).

Per-role model tiers are sent as a per-invoke ``model`` override, which the spike verified
honours three different tiers on ONE harness — so hosting does not multiply harness count
(T-HARN-09).
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config.aws_region import resolve_aws_region
from ..config.guardrails import converse_guardrail_config
from ..config.runtime import Config, load_config
from .spec import HarnessSpecError

#: Minimum botocore that ships the harness operations. Stated ONCE so the error message
#: cannot drift from the ``pyproject.toml`` constraint (a test pins the two together).
HARNESS_BOTOCORE_FLOOR = "1.43.0"

#: A normal end-of-turn. ``messageStop`` is the last event of a healthy stream.
TERMINAL_EVENTS: frozenset[str] = frozenset({"messageStop"})

#: Error events that arrive ON THE STREAM and terminate the turn.
ERROR_EVENTS: frozenset[str] = frozenset(
    {
        "validationException",
        "internalServerException",
        "runtimeClientError",
        "throttlingException",
        "accessDeniedException",
        "serviceQuotaExceededException",
        "conflictException",
        "resourceNotFoundException",
    }
)

#: ``runtimeSessionId`` minimum length, enforced service-side (measured in the spike).
SESSION_ID_MIN_LENGTH = 33
_SESSION_ID_MAX_LENGTH = 120

DEFAULT_DEADLINE_SECONDS = 900.0


class HarnessInvokeError(RuntimeError):
    """Base class for harness invocation failures."""


class HarnessSdkTooOldError(HarnessInvokeError):
    """The installed SDK has no ``InvokeHarness`` operation."""


class HarnessTerminalEventMissingError(HarnessInvokeError):
    """The stream ended, or the deadline elapsed, with no terminal event.

    Raised rather than returning a partial turn: a caller that cannot tell "finished" from
    "truncated" will happily act on half an answer.
    """


@dataclass(frozen=True)
class HarnessToolUse:
    """One ``inline_function`` call the harness paused on, for our code to execute."""

    name: str
    tool_use_id: str
    input: Mapping[str, Any]


@dataclass(frozen=True)
class HarnessTurn:
    """The outcome of one ``InvokeHarness`` turn.

    Attributes:
        text: Concatenated assistant text from the ``contentBlockDelta`` events.
        events: Every decoded event, in arrival order.
        tool_uses: ``inline_function`` calls awaiting execution in our process.
        usage: Token usage from the ``metadata`` event.
        error: The terminal error event, or ``None``.
        lead_name: Owning lead — usage and errors are attributed to it.
        role: Model tier the invocation ran under.
        session_id: The ``runtimeSessionId`` actually sent.
        session_reused: ``False`` when this invocation paid a cold start.
        ttft_s: Seconds to the first text delta.
        total_s: Seconds to the terminal event.
    """

    text: str
    events: list[dict[str, Any]]
    tool_uses: list[HarnessToolUse]
    usage: dict[str, Any]
    error: dict[str, Any] | None
    lead_name: str
    role: str
    session_id: str
    session_reused: bool
    ttft_s: float | None
    total_s: float

    @property
    def is_error(self) -> bool:
        return self.error is not None


def make_harness_session_id(project_id: str) -> str:
    """A fresh ``runtimeSessionId`` of at least :data:`SESSION_ID_MIN_LENGTH` characters."""
    # Imported from its DEFINING module rather than via ``cloud.dispatch`` (which merely
    # re-exports it), so the harness client does not depend on the dispatch surface.
    from ..cloud.spec_builder import sanitize_project_id

    return _pad_session_id(f"amlas-h-{sanitize_project_id(project_id)}-{uuid.uuid4().hex}")


def _pad_session_id(session_id: str) -> str:
    """Pad DETERMINISTICALLY to the service floor (T-HARN-11.2).

    Deterministic matters: a caller that reuses its own session string across iterations must
    get the same padded value back, or reuse silently becomes a new session — and a new
    session is a ~55s cold start.
    """
    sid = session_id[:_SESSION_ID_MAX_LENGTH]
    if len(sid) >= SESSION_ID_MIN_LENGTH:
        return sid
    return sid + "0" * (SESSION_ID_MIN_LENGTH - len(sid))


@dataclass
class HarnessSessions:
    """Session bookkeeping per ``(lead, run)`` — the T-HARN-20 reuse ledger.

    Keyed on the pair rather than on the lead alone so a session stays stable *within* a run
    and distinct *between* runs, which is what keeps the managed-memory scoping in T-HARN-06.3
    intact while still amortising the cold start.
    """

    _ids: dict[tuple[str, str], str] = field(default_factory=dict)
    _uses: dict[tuple[str, str], int] = field(default_factory=dict)
    _cold_starts: dict[tuple[str, str], int] = field(default_factory=dict)

    @staticmethod
    def _key(lead_name: str, run_id: str) -> tuple[str, str]:
        return (lead_name, run_id or "no-run")

    def session_for(self, *, lead_name: str, run_id: str) -> str:
        """The session id for this ``(lead, run)``, creating one on first use."""
        key = self._key(lead_name, run_id)
        if key not in self._ids:
            self._ids[key] = make_harness_session_id(f"{lead_name}-{run_id or 'no-run'}")
            self._cold_starts[key] = self._cold_starts.get(key, 0) + 1
        self._uses[key] = self._uses.get(key, 0) + 1
        return self._ids[key]

    def was_reused(self, *, lead_name: str, run_id: str) -> bool:
        """True once the session has been handed out more than once."""
        return self._uses.get(self._key(lead_name, run_id), 0) > 1

    def renew(self, *, lead_name: str, run_id: str, reason: str = "expired") -> str:
        """Replace an unusable session and record that a cold start was incurred.

        T-HARN-20.5 — recovery must be transparent to the caller but VISIBLE in the trace,
        otherwise a session silently expiring every iteration looks like the platform being
        slow rather than a reuse bug.
        """
        key = self._key(lead_name, run_id)
        self._ids.pop(key, None)
        self._uses[key] = 0
        _ = reason
        return self.session_for(lead_name=lead_name, run_id=run_id)

    def cold_starts(self, *, lead_name: str, run_id: str) -> int:
        """How many times a NEW session was established for this ``(lead, run)``."""
        return self._cold_starts.get(self._key(lead_name, run_id), 0)


def _model_block(
    *, role: str, model_id: str | None, config: Config | None
) -> dict[str, Any]:
    """Per-invoke model override — explicit ``model_id`` beats the role tier.

    Same precedence as [`_build.build_agent`](../_build.py:88), read from the same
    ``cfg.models.for_role`` map, so the harness path cannot drift onto a different tier than
    the in-process path for the same role.

    **The guardrail rides along, and it MUST.** A harness has no guardrail field, so the
    guardrail lives in ``bedrockModelConfig.additionalParams.guardrailConfig`` (T-HARN-02).
    A per-invoke ``model`` override REPLACES that whole block — so sending a bare override
    would silently strip the guardrail from the turn, making the per-role tier feature
    (T-HARN-09) a silent deletion of WA-02 / AGENTSEC04 filtering (T-HARN-02). Re-attaching it
    here is what keeps the two features from cancelling each other out.
    """
    resolved = model_id
    if resolved is None:
        cfg = config or load_config()
        resolved = cfg.models.for_role(role)  # type: ignore[arg-type]
    model_cfg: dict[str, Any] = {
        "modelId": resolved,
        "apiFormat": "converse_stream",
    }
    guardrail = converse_guardrail_config()
    if guardrail:
        model_cfg["additionalParams"] = {"guardrailConfig": dict(guardrail)}
    return {"bedrockModelConfig": model_cfg}


def _user_messages(prompt: str) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [{"text": prompt}]}]


def _capture(error_log: Any, exc: BaseException | str, *, lead_name: str) -> None:
    """Funnel a harness failure, guarded so telemetry can NEVER mask the real error.

    Mirrors [`_build.py`](../_build.py:171): the same swallow-and-continue discipline, because
    an exception raised from the capture path would replace the exception the caller needs.
    """
    try:
        if error_log is not None:
            error_log.capture(exc, source=f"harness.{lead_name}")
        else:
            from ..telemetry.error_log import capture_error

            capture_error(exc, source=f"harness.{lead_name}")
    except Exception:  # nosec B110
        pass


class _StreamReader:
    """Accumulates one turn from the event stream."""

    def __init__(self) -> None:
        self.text: list[str] = []
        self.events: list[dict[str, Any]] = []
        self.tool_uses: list[HarnessToolUse] = []
        self.usage: dict[str, Any] = {}
        self.error: dict[str, Any] | None = None
        self.ttft_s: float | None = None
        self._cur_tool: dict[str, Any] | None = None
        self._cur_input = ""

    def feed(self, event: Mapping[str, Any], *, started: float) -> bool:
        """Absorb one event. Returns True when the turn is terminal."""
        self.events.append(dict(event))

        if "contentBlockStart" in event:
            tu = (event["contentBlockStart"].get("start") or {}).get("toolUse") or {}
            if tu:
                self._cur_tool = {
                    "name": str(tu.get("name") or ""),
                    "toolUseId": str(tu.get("toolUseId") or ""),
                }
                self._cur_input = ""

        if "contentBlockDelta" in event:
            delta = event["contentBlockDelta"].get("delta") or {}
            if delta.get("text"):
                if self.ttft_s is None:
                    self.ttft_s = round(time.monotonic() - started, 3)
                self.text.append(str(delta["text"]))
            tu_delta = delta.get("toolUse") or {}
            if tu_delta.get("input"):
                self._cur_input += str(tu_delta["input"])

        if "contentBlockStop" in event and self._cur_tool is not None:
            self.tool_uses.append(
                HarnessToolUse(
                    name=self._cur_tool["name"],
                    tool_use_id=self._cur_tool["toolUseId"],
                    input=_decode_tool_input(self._cur_input),
                )
            )
            self._cur_tool, self._cur_input = None, ""

        if "metadata" in event:
            self.usage = dict(event["metadata"].get("usage") or {})

        for kind in ERROR_EVENTS:
            if kind in event:
                payload = event[kind]
                message = (
                    payload.get("message")
                    if isinstance(payload, Mapping)
                    else str(payload)
                )
                self.error = {"kind": kind, "message": str(message or payload)}
                return True

        return any(kind in event for kind in TERMINAL_EVENTS)


def _decode_tool_input(raw: str) -> Mapping[str, Any]:
    """Decode accumulated partial-JSON tool input, PRESERVING an undecodable payload.

    Dropping it would hide the reason a round trip failed; ``{"_raw": ...}`` keeps it
    inspectable.
    """
    if not raw.strip():
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {"_raw": raw}
    return decoded if isinstance(decoded, Mapping) else {"_raw": raw}


def _resolve_client(client: Any) -> Any:
    """Return a client that has ``invoke_harness``, or explain what is missing."""
    if client is None:
        import boto3

        from ..config.runtime import load_config as _load

        region = resolve_aws_region(getattr(_load(), "region", None))
        client = boto3.Session(region_name=resolve_aws_region(region)).client("bedrock-agentcore")
    if not hasattr(client, "invoke_harness"):
        raise HarnessSdkTooOldError(
            "the installed AWS SDK has no InvokeHarness operation; AgentCore harness "
            f"support requires botocore>={HARNESS_BOTOCORE_FLOOR}"
        )
    return client


def invoke_harness(
    *,
    harness_arn: str,
    prompt: str | None = None,
    messages: Sequence[Mapping[str, Any]] | None = None,
    lead_name: str,
    role: str = "planner",
    model_id: str | None = None,
    session_id: str | None = None,
    run_id: str = "",
    actor_id: str = "",
    sessions: HarnessSessions | None = None,
    allowed_tools: Sequence[str] | None = None,
    sink: Callable[[dict[str, Any]], None] | None = None,
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
    client: Any | None = None,
    config: Config | None = None,
    error_log: Any | None = None,
) -> HarnessTurn:
    """Run one turn against a harness and return the assembled :class:`HarnessTurn`.

    Args:
        harness_arn: Target harness ARN.
        prompt: Convenience single user turn. Ignored when ``messages`` is given.
        messages: Full Converse-shaped message list (used to replay a ``toolResult``).
        lead_name: Owning lead — errors and usage are attributed to it.
        role: Model tier; resolved through ``cfg.models.for_role``.
        model_id: Explicit model id, winning over ``role``.
        session_id: Explicit session id; padded to the service floor if short.
        run_id: Run identity, used with ``sessions`` to reuse one session per (lead, run).
        actor_id: Managed-memory actor scope — normally the project id. **This is an
            ``InvokeHarness`` parameter, not part of the harness's memory configuration**, so it
            cannot live on the spec: managed memory namespaces facts under
            ``/actors/{actorId}/…``, and omitting it leaves long-term memory unscoped, so
            nothing carries over per project. Sent only when non-empty, keeping the
            no-memory path byte-identical.
        sessions: Reuse ledger. Supplying one is what avoids the ~55s cold start per call.
        allowed_tools: Per-invoke ``allowedTools`` override. ``["*"]`` is refused.
        sink: Called with every event AS IT ARRIVES.
        deadline_seconds: Hard wall clock; exceeding it raises rather than hanging.
        client: Injected client. Callers may pass a fake; ``None`` builds a real one.
        config: Config override.
        error_log: Injected :class:`~amlas.telemetry.error_log.ErrorLog`.

    Raises:
        HarnessSpecError: If ``allowed_tools`` contains ``"*"`` (ADR-0002).
        HarnessSdkTooOldError: If the SDK lacks the operation.
        HarnessTerminalEventMissingError: If no terminal event arrives in time.
    """
    if allowed_tools is not None and "*" in allowed_tools:
        raise HarnessSpecError(
            "allowedTools=['*'] makes the in-VM shell builtin LLM-reachable; use the "
            "@*/<tool> form (ADR-0002, T-HARN-05.8)"
        )

    resolved_client = _resolve_client(client)

    reused = False
    if session_id is None and sessions is not None:
        # Order matters: ``session_for`` counts this hand-out, so ``was_reused`` must be
        # read AFTER it to describe THIS invocation rather than the previous one.
        sid = sessions.session_for(lead_name=lead_name, run_id=run_id)
        reused = sessions.was_reused(lead_name=lead_name, run_id=run_id)
    elif session_id is None:
        sid = make_harness_session_id(f"{lead_name}-{run_id or 'no-run'}")
    else:
        sid = _pad_session_id(session_id)

    kwargs: dict[str, Any] = {
        "harnessArn": harness_arn,
        "runtimeSessionId": sid,
        "messages": list(messages) if messages else _user_messages(prompt or ""),
        "model": _model_block(role=role, model_id=model_id, config=config),
    }
    if allowed_tools is not None:
        kwargs["allowedTools"] = list(allowed_tools)
    if actor_id:
        kwargs["actorId"] = actor_id

    started = time.monotonic()
    try:
        response = resolved_client.invoke_harness(**kwargs)
    except Exception as exc:
        _capture(error_log, exc, lead_name=lead_name)
        raise

    reader = _StreamReader()
    terminal = False
    stream: Iterable[Mapping[str, Any]] = response.get("stream", []) or []
    for event in stream:
        if not isinstance(event, Mapping):
            continue
        terminal = reader.feed(event, started=started)
        if sink is not None:
            try:
                sink(dict(event))
            except Exception:  # nosec B110
                pass  # a sink failure never breaks the live stream
        if terminal:
            break
        if time.monotonic() - started > deadline_seconds:
            raise HarnessTerminalEventMissingError(
                f"harness.{lead_name}: no terminal event within the "
                f"{deadline_seconds}s deadline (session {sid})"
            )

    if not terminal:
        raise HarnessTerminalEventMissingError(
            f"harness.{lead_name}: stream ended with no terminal event "
            f"({len(reader.events)} events, session {sid})"
        )

    if reader.error is not None:
        _capture(
            error_log,
            f"{reader.error['kind']}: {reader.error['message']}",
            lead_name=lead_name,
        )

    return HarnessTurn(
        text="".join(reader.text),
        events=reader.events,
        tool_uses=reader.tool_uses,
        usage=reader.usage,
        error=reader.error,
        lead_name=lead_name,
        role=role,
        session_id=sid,
        session_reused=reused,
        ttft_s=reader.ttft_s,
        total_s=round(time.monotonic() - started, 3),
    )


__all__ = [
    "DEFAULT_DEADLINE_SECONDS",
    "ERROR_EVENTS",
    "HARNESS_BOTOCORE_FLOOR",
    "SESSION_ID_MIN_LENGTH",
    "TERMINAL_EVENTS",
    "HarnessInvokeError",
    "HarnessSdkTooOldError",
    "HarnessSessions",
    "HarnessTerminalEventMissingError",
    "HarnessToolUse",
    "HarnessTurn",
    "invoke_harness",
    "make_harness_session_id",
]
