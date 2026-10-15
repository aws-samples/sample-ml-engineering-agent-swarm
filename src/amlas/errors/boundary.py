"""Tool error boundary (, Phase 2).

A decorator that gives every tool a UNIFORM error contract: on an exception it (1) classifies
via the taxonomy, (2) captures into the centralized dual-sink funnel
(:func:`amlas.telemetry.error_log.capture_error` → local JSONL always + CloudWatch mirror in
cloud modes), and (3) returns the STANDARD shape
``{"success": False, "error": <user_message>, "error_code": ..., "retryable": ...,
"category": ...}`` — replacing the ~310 hand-rolled ``{"success": False, "error": ...}`` dicts
that were never funneled and had no stable schema.

Guardrail/security errors are re-raised (fail-closed) rather than softened into a dict, so a
denial cannot be mistaken for a soft tool failure.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from .messages import format_user_error
from .taxonomy import AmlasError, ErrorCategory, classify

F = TypeVar("F", bound=Callable[..., Any])


def tool_error_boundary(source: str = "", *, reraise_guardrail: bool = True) -> Callable[[F], F]:
    """Wrap a tool fn so any exception becomes the standard error dict + is captured.

    Args:
        source: the logical origin tag for the error funnel (e.g. ``"tools.data_tools"``);
            defaults to the wrapped function's module when empty.
        reraise_guardrail: when True (default) a :class:`~amlas.errors.taxonomy.GuardrailDeniedError`
            (or any ``fail_closed`` guardrail-category error) is RE-RAISED rather than converted
            to a soft dict — a safety denial must never look like a recoverable tool failure.

    The returned dict shape:
        ``{"success": False, "error": <actionable message>, "error_code": <slug>,
           "retryable": <bool>, "category": <category>}``.
    """

    def _decorate(fn: F) -> F:
        tag = source or getattr(fn, "__module__", "") or "tools"

        @functools.wraps(fn)
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except BaseException as exc:  # classified + funneled below (broad by design)
                err = classify(exc, source=tag)
                # Fail-closed: a guardrail denial is re-raised, not softened.
                if reraise_guardrail and err.category is ErrorCategory.GUARDRAIL:
                    _capture(err, tag)
                    raise
                # KeyboardInterrupt / SystemExit must propagate — never swallow.
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                _capture(err, tag)
                # ``error`` carries the RAW underlying message — this matches the ~310
                # hand-rolled ``{"success": False, "error": str(e)}`` dicts the boundary
                # replaces, and (critically) preserves the SPECIFIC, actionable failure text
                # (e.g. sklearn's "least populated class ... has only 1 member") that internal
                # callers surface and the A5 "no hollow success" hardening asserts on. The
                # friendly, category-based guidance lives in ``user_message`` (for UI/chat).
                return {
                    "success": False,
                    "error": str(exc),
                    "user_message": format_user_error(exc, source=tag),
                    "error_code": err.code,
                    "retryable": err.retryable,
                    "category": str(err.category),
                }

        return _wrapped  # type: ignore[return-value]

    return _decorate


def _capture(err: AmlasError, source: str) -> None:
    """Funnel a classified error into the centralized dual-sink surface (best-effort)."""
    try:
        from ..telemetry.error_log import capture_error

        capture_error(
            err,
            source=source,
            error_type=err.code,
            context={"category": str(err.category), "retryable": err.retryable},
        )
    except Exception:  # nosec B110
        pass  # telemetry must never break the tool path


__all__ = ["tool_error_boundary"]
