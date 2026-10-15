"""Approval-gated ``inline_function`` calls on the harness path (T-HARN-18).

This is where the migration's security shape lives. A harness runs the agent loop
**server-side**, so a gated tool executed there would bypass the approval gate entirely and a
server-side loop could effectively self-approve spend or an irreversible action.

It cannot, because there is no Gateway: every tool is an ``inline_function``, so the harness
PAUSES and hands the call back to our process. **That pause is the gate.** The decision is made
here, by the real [`request_approval`](../control_plane/approvals.py:265) policy, and a denial
is returned to the harness as a tool RESULT rather than being executed.

Returning the denial rather than raising is deliberate. The harness is mid-turn waiting on a
result for the ``toolUse`` it emitted; abandoning it leaves the turn dangling, whereas telling
the model it was refused lets it explain or choose another path. Callers that would rather fail
loudly can pass ``raise_on_denial=True``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from ..control_plane.approvals import ApprovalPolicy, request_approval
from .client import HarnessToolUse

#: Executor signature: given the paused call, run the real tool and return its result.
ToolExecutor = Callable[[HarnessToolUse], Any]


class ApprovalDeniedError(RuntimeError):
    """A gated tool call was refused, and the caller asked to fail rather than continue."""


def _content(value: Any) -> list[dict[str, Any]]:
    """Render a tool result as the harness expects — text blocks, JSON when structured."""
    if isinstance(value, str):
        return [{"text": value}]
    try:
        return [{"text": json.dumps(value, default=str)}]
    except (TypeError, ValueError):
        return [{"text": str(value)}]


def _result_block(call: HarnessToolUse, value: Any, *, status: str) -> dict[str, Any]:
    return {
        "toolResult": {
            "toolUseId": call.tool_use_id,
            "content": _content(value),
            "status": status,
        }
    }


def resolve_tool_calls(
    tool_uses: Sequence[HarnessToolUse],
    *,
    executor: ToolExecutor,
    policy: ApprovalPolicy | None = None,
    cost_estimates: dict[str, float] | None = None,
    on_decision: Callable[[dict[str, Any]], None] | None = None,
    raise_on_denial: bool = False,
) -> list[dict[str, Any]]:
    """Gate, then execute, each paused ``inline_function`` call — in order.

    Every call is decided INDEPENDENTLY, so one denial neither suppresses nor implies anything
    about the others. Results come back in the same order and carry the same ``toolUseId``s,
    because the harness matches results to calls by id and a mismatch corrupts the turn.

    Args:
        tool_uses: The calls the harness paused on.
        executor: Runs the real tool. Only ever invoked for an APPROVED call.
        policy: Approval policy; defaults to the same default ``request_approval`` uses, so
            headless callers inherit the safe-by-default non-interactive ASK behaviour (a gated
            action is denied rather than auto-spent).
        cost_estimates: Optional per-tool cost estimate, which is what lets bounded compute
            auto-approve under its cap.
        on_decision: Receives each raw decision dict — the audit trail.
        raise_on_denial: Raise :class:`ApprovalDeniedError` instead of returning a denial result.

    Returns:
        One ``{"toolResult": ...}`` block per call, ready for :func:`build_resume_messages`.
    """
    estimates = cost_estimates or {}
    results: list[dict[str, Any]] = []

    for call in tool_uses:
        decision = request_approval(
            action=call.name,
            cost_estimate=float(estimates.get(call.name, 0.0)),
            impact=f"harness inline_function {call.name}",
            policy=policy,
        )
        if on_decision is not None:
            try:
                on_decision(dict(decision))
            except Exception:  # nosec B110
                pass  # an audit sink must never break the turn

        if decision.get("decision") != "approved":
            if raise_on_denial:
                raise ApprovalDeniedError(
                    f"{call.name}: {decision.get('reason') or 'approval denied'}"
                )
            # The underlying tool is NOT executed. The harness still needs a result, so the
            # refusal is what it gets.
            results.append(
                _result_block(
                    call,
                    {
                        "denied": True,
                        "tool": call.name,
                        "reason": decision.get("reason") or "approval denied",
                        "action_class": decision.get("action_class"),
                    },
                    status="error",
                )
            )
            continue

        try:
            value = executor(call)
        except Exception as exc:
            # A raising tool must still produce a result, or the paused turn never resumes.
            results.append(
                _result_block(
                    call,
                    {"error": f"{type(exc).__name__}: {exc}", "tool": call.name},
                    status="error",
                )
            )
            continue
        results.append(_result_block(call, value, status="success"))

    return results


def build_resume_messages(
    *,
    prompt: str,
    tool_uses: Iterable[HarnessToolUse],
    tool_results: Sequence[dict[str, Any]],
    prior_messages: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Assemble the message list that resumes a paused harness turn.

    The harness resumes only from a well-formed pair: the assistant's ``toolUse`` blocks
    replayed, then a user turn carrying the matching ``toolResult`` blocks. Proven end to end
    by the T-HARN-01 Part B round trip.
    """
    messages: list[dict[str, Any]] = list(prior_messages or [])
    if not messages:
        messages.append({"role": "user", "content": [{"text": prompt}]})
    messages.append(
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "name": call.name,
                        "toolUseId": call.tool_use_id,
                        "input": dict(call.input),
                    }
                }
                for call in tool_uses
            ],
        }
    )
    messages.append({"role": "user", "content": list(tool_results)})
    return messages


__all__ = [
    "ApprovalDeniedError",
    "ToolExecutor",
    "build_resume_messages",
    "resolve_tool_calls",
]
