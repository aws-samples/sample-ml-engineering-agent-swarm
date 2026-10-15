"""Universal error handling.

The *handling* discipline that complements the *capture* surface
(:mod:`amlas.telemetry.error_log`) and the process net (:mod:`amlas._runtime_guard`):

* :mod:`.taxonomy` — the ``AmlasError`` root + category classes + :func:`classify`;
* :mod:`.retry`    — one shared full-jitter exponential backoff (:func:`retry_with_backoff`);
* :mod:`.messages` — :func:`format_user_error`, actionable per-category remediation;
* :mod:`.boundary` — :func:`tool_error_boundary`, the uniform tool error contract + funnel.

Fail-mode policy (see the taxonomy's ``fail_closed`` attribute):
* GUARDRAIL / security / data-integrity errors fail **closed** (deny) — never softened;
* TRANSIENT / observability / best-effort failures fail **open** (surface + retry), never
  blocking the path.

Emission (dual-sink, mode-driven — ):
* **offline** → local JSONL error file ONLY (offline-survivable; the single file the TUI +
  tui-agent share);
* **local / full_aws** → durable **CloudWatch** mirror + the local JSONL file.
The local file is pruned to a bounded tail (``AMLAS_ERROR_LOG_MAX_LINES``).
"""

from __future__ import annotations

from .boundary import tool_error_boundary
from .messages import format_user_error
from .retry import backoff_delay, retry_with_backoff
from .taxonomy import (
    AmlasError,
    AuthError,
    ConfigError,
    DependencyError,
    ErrorCategory,
    GuardrailDeniedError,
    PermanentError,
    TransientError,
    UserInputError,
    classify,
    is_retryable,
)

__all__ = [
    "AmlasError",
    "AuthError",
    "ConfigError",
    "DependencyError",
    "ErrorCategory",
    "GuardrailDeniedError",
    "PermanentError",
    "TransientError",
    "UserInputError",
    "backoff_delay",
    "classify",
    "format_user_error",
    "is_retryable",
    "retry_with_backoff",
    "tool_error_boundary",
]
