"""Universal error taxonomy.

amlas already has a strong error *capture* surface (:mod:`amlas.telemetry.error_log`) and a
process safety net (:mod:`amlas._runtime_guard`). What it lacked was a shared *handling*
discipline: a common exception root every layer can classify, attach retryability +
user-actionable messaging to, and decide fail-closed vs fail-open on. This module is that
root.

Design
------
``AmlasError`` is the common base. Every domain error can subclass a CATEGORY base
(:class:`TransientError`, :class:`PermanentError`, …) so callers can classify by ``isinstance``
or by the ``.category`` / ``.retryable`` attributes rather than string-matching. Each instance
carries:

* ``code``        — a short stable slug (e.g. ``"quota_exceeded"``) for logs/telemetry;
* ``category``    — the :class:`ErrorCategory` (transient/permanent/user_input/config/auth/
                    dependency/guardrail);
* ``retryable``   — whether a retry could plausibly succeed (transient → True);
* ``fail_closed`` — the disposition when unhandled: guardrail/security/data-integrity errors
                    fail CLOSED (deny), observability/best-effort failures fail OPEN;
* ``user_message``— an actionable, human-facing string (no raw trace);
* ``source``      — the emitting layer (``tui.chat.engine`` / ``agents.<name>`` / …), matching
                    the :mod:`amlas.telemetry.error_log` ``source`` taxonomy;
* ``context``     — a small structured dict of extra detail.

:func:`classify` maps ANY raw exception (botocore / stdlib / third-party) onto an
``AmlasError`` category using keyword signatures promoted from the two places that already
did this ad-hoc: the code-interpreter quota tokens (FIX-27) and the Bedrock creds markers.
It is dependency-free and never raises.

Backward compatibility: category bases multiply-inherit their historical stdlib base where
callers already ``except`` it (e.g. :class:`TransientError` is a ``RuntimeError``,
:class:`UserInputError` is a ``ValueError``), so re-parenting an existing exception onto the
taxonomy does not break existing ``except`` clauses.
"""

from __future__ import annotations

from typing import Any

# ``enum.StrEnum`` is py3.11+. This module must ALSO import under an older interpreter, because
# it is not only ours to run: the SageMaker training entrypoint
# (:mod:`amlas.sagemaker_entrypoints.tabular_train`) imports ``tools.training_tools`` → ``errors``
# INSIDE the managed sklearn container, whose Python we do not control. A bare
# ``from enum import StrEnum`` made every dispatched training job die at import with
# ``ImportError: cannot import name 'StrEnum' from 'enum'`` — invisible offline,
# because the repo's own interpreter is 3.11.
#
# The fallback is NOT a plain ``(str, Enum)``: on py<3.11 that inherits ``Enum.__str__``, so
# ``str(ErrorCategory.TRANSIENT)`` would become ``"ErrorCategory.TRANSIENT"`` instead of
# ``"transient"`` — silently corrupting the telemetry ``category`` field and anything that
# formats a category into a message. ``__str__``/``__format__`` are pinned to the VALUE so the
# two paths are behaviourally identical.
try:  # pragma: no cover - exercised by whichever interpreter is running
    from enum import StrEnum
except ImportError:  # pragma: no cover - py<3.11, e.g. the sklearn training container
    from enum import Enum

    # ruff UP042 advises inheriting from ``enum.StrEnum`` — which is the very thing this branch
    # exists because we cannot do. Suppressed, not "fixed".
    class StrEnum(str, Enum):  # type: ignore[no-redef]  # noqa: UP042
        """py<3.11 stand-in for :class:`enum.StrEnum` — value-valued ``str()``/``format()``."""

        def __str__(self) -> str:
            return str(self.value)

        def __format__(self, format_spec: str) -> str:
            return str.__format__(str(self.value), format_spec)


class ErrorCategory(StrEnum):
    """The disposition classes every amlas error falls into.

    A ``StrEnum`` so ``str(ErrorCategory.TRANSIENT) == "transient"`` and it JSON-serializes as
    its value — no custom ``__str__`` needed at the call sites. See the import above for why a
    py<3.11 fallback exists and why it pins ``__str__``.
    """

    TRANSIENT = "transient"      # quota/throttle/capacity/network — retry may succeed
    PERMANENT = "permanent"      # deterministic failure — retry will not help
    USER_INPUT = "user_input"    # bad/ambiguous user input — actionable by the user
    CONFIG = "config"            # missing/invalid configuration — actionable by the operator
    AUTH = "auth"                # credentials/permissions — actionable by re-auth
    DEPENDENCY = "dependency"    # a downstream service/library failed
    GUARDRAIL = "guardrail"      # a safety/security/data-integrity gate denied the action


class AmlasError(Exception):
    """Root of the amlas error taxonomy — carries classification + user-facing metadata.

    Not raised directly; use a category subclass. All fields have safe defaults so a bare
    ``AmlasError("msg")`` is valid.
    """

    # Category defaults (overridden by subclasses). Class-level so ``classify`` and callers
    # can read them off the type as well as the instance.
    category: ErrorCategory = ErrorCategory.PERMANENT
    retryable: bool = False
    fail_closed: bool = True

    def __init__(
        self,
        message: str = "",
        *,
        code: str = "",
        user_message: str = "",
        source: str = "",
        context: dict[str, Any] | None = None,
        retryable: bool | None = None,
        fail_closed: bool | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or self.category.value
        self.user_message = user_message or message
        self.source = source
        self.context: dict[str, Any] = dict(context or {})
        # Instance overrides fall back to the class-level category defaults.
        self.retryable = self.__class__.retryable if retryable is None else retryable
        self.fail_closed = self.__class__.fail_closed if fail_closed is None else fail_closed
        if cause is not None:
            self.__cause__ = cause

    def to_dict(self) -> dict[str, Any]:
        """A telemetry-friendly summary (feeds :func:`amlas.telemetry.error_log.capture_error`)."""
        return {
            "code": self.code,
            "category": str(self.category),
            "retryable": self.retryable,
            "fail_closed": self.fail_closed,
            "source": self.source,
            "message": self.message,
            "user_message": self.user_message,
            "context": self.context,
        }

    def __str__(self) -> str:
        return self.message or self.__class__.__name__


# ── category bases (multiply-inherit the historical stdlib base for compatibility) ──


class TransientError(AmlasError, RuntimeError):
    """A transient condition (quota/throttle/capacity/network) — a retry may succeed."""

    category = ErrorCategory.TRANSIENT
    retryable = True
    fail_closed = False  # a transient failure is safe to surface + retry, not deny-by-default


class PermanentError(AmlasError, RuntimeError):
    """A deterministic failure — retrying will not help."""

    category = ErrorCategory.PERMANENT
    retryable = False
    fail_closed = True


class UserInputError(AmlasError, ValueError):
    """Bad/ambiguous USER input — actionable by the user, not retryable as-is."""

    category = ErrorCategory.USER_INPUT
    retryable = False
    fail_closed = True


class ConfigError(AmlasError, RuntimeError):
    """Missing/invalid configuration — actionable by the operator (not retryable)."""

    category = ErrorCategory.CONFIG
    retryable = False
    fail_closed = True


class AuthError(AmlasError, RuntimeError):
    """Credentials/permissions failure — actionable by re-authenticating."""

    category = ErrorCategory.AUTH
    retryable = False
    fail_closed = True


class DependencyError(AmlasError, RuntimeError):
    """A downstream service/library failed in a way that isn't obviously transient."""

    category = ErrorCategory.DEPENDENCY
    retryable = False
    fail_closed = True


class GuardrailDeniedError(AmlasError, RuntimeError):
    """A safety/security/data-integrity gate DENIED the action — always fail-closed."""

    category = ErrorCategory.GUARDRAIL
    retryable = False
    fail_closed = True


# ── keyword signatures for classifying RAW (non-AmlasError) exceptions ─────────────
# Promoted from the two places that already did this ad-hoc:
#   * transient/quota tokens — amlas.tools.code_interpreter (FIX-27 quota backoff);
#   * auth/credential markers — amlas.config.bedrock_creds.diagnose_aws_error (it lived in
#     amlas.tui.chat.bedrock_backend when these were promoted; that module is now deleted).
# Order matters: the FIRST matching category wins, so AUTH (a specific, actionable class) is
# checked before the broader TRANSIENT/DEPENDENCY nets.

_AUTH_MARKERS = (
    "expiredtoken",
    "expired token",
    "the security token included in the request is expired",
    "unrecognizedclient",
    "invalidclienttokenid",
    "accessdenied",
    "access denied",
    "unable to locate credentials",
    "credentials were not found",
    "sso session",
    "token has expired",
    "no credentials",
    "unauthorized",
    "forbidden",
)

_TRANSIENT_MARKERS = (
    "servicequotaexceeded",
    "throttl",           # Throttling / ThrottlingException
    "toomanyrequests",
    "too many requests",
    "rate exceeded",
    "provisionedthroughputexceeded",
    "capacity",
    "serviceunavailable",
    "service unavailable",
    "503",
    "500",
    "timeout",
    "timed out",
    "connection reset",
    "connection aborted",
    "temporarily unavailable",
    "modelnotready",
    "modeltimeout",
)

_CONFIG_MARKERS = (
    "not configured",
    "missing required",
    "no such profile",
    "could not be found",  # botocore "config profile (X) could not be found"
    "environment variable",
)

_USER_INPUT_MARKERS = (
    "no dataset",
    "file not found",
    "no such file",
    "could not resolve",
    "invalid spec",
    "unknown algorithm",
    "unknown classification algorithm",
    "unknown regression algorithm",
    "target column",
)


def _text_of(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}".lower()


def classify(exc: BaseException, *, source: str = "") -> AmlasError:
    """Map ANY exception onto an ``AmlasError`` category (dependency-free, never raises).

    An ``AmlasError`` is returned as-is (its ``source`` filled in when absent). A raw
    exception is matched against the keyword signatures — AUTH first (specific + actionable),
    then TRANSIENT (retryable), CONFIG, USER_INPUT — falling back to :class:`DependencyError`
    for anything unrecognized (a downstream failure we don't yet classify), preserving the
    original as ``__cause__``.
    """
    if isinstance(exc, AmlasError):
        if source and not exc.source:
            exc.source = source
        return exc

    text = _text_of(exc)
    kind = type(exc).__name__

    def _mk(cls: type[AmlasError], code: str) -> AmlasError:
        return cls(f"{kind}: {exc}", code=code, source=source, cause=exc)

    if any(m in text for m in _AUTH_MARKERS):
        return _mk(AuthError, "auth_failed")
    if any(m in text for m in _TRANSIENT_MARKERS):
        return _mk(TransientError, "transient")
    if any(m in text for m in _CONFIG_MARKERS):
        return _mk(ConfigError, "config_error")
    if any(m in text for m in _USER_INPUT_MARKERS):
        return _mk(UserInputError, "user_input")
    # Built-in input errors map to user-input; everything else is a dependency failure.
    if isinstance(exc, (ValueError, KeyError, FileNotFoundError)):
        return _mk(UserInputError, "user_input")
    return _mk(DependencyError, "dependency_error")


def is_retryable(exc: BaseException) -> bool:
    """True when ``exc`` (raw or AmlasError) is classified as retryable/transient."""
    return classify(exc).retryable
