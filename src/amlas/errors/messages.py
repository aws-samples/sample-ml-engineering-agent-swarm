"""User-actionable error messages (, Phase 3).

Generalizes the credential-diagnosis pattern that lived only in the TUI chat backend
(:func:`amlas.config.bedrock_creds.diagnose_aws_error`, since moved to a neutral config
home) into a taxonomy-driven formatter every
layer can call, so a cryptic botocore/stdlib trace becomes an actionable remediation string
(re-auth, wait-and-retry, fix-config, fix-input) instead of a raw ``type(e): e`` dump.
"""

from __future__ import annotations

from .taxonomy import AmlasError, ErrorCategory, classify

# Per-category actionable remediation templates. ``{detail}`` is the underlying error type.
_REMEDIATION: dict[ErrorCategory, str] = {
    ErrorCategory.AUTH: (
        "🔐 AWS credentials are missing or expired. Re-authenticate (e.g. "
        "`aws sso login --profile <your-profile>`) or refresh your AWS credential chain, "
        "then retry."
    ),
    ErrorCategory.TRANSIENT: (
        "⏳ A transient capacity/throttle/network condition. This usually clears on its own — "
        "retry in a moment (amlas retries transient failures with backoff automatically)."
    ),
    ErrorCategory.CONFIG: (
        "⚙️ A configuration value is missing or invalid. Check your `.env` / SSM `/amlas/*` "
        "settings (runtime ARN, session bucket, region) and re-run."
    ),
    ErrorCategory.USER_INPUT: (
        "✏️ The request couldn't be resolved as given. Check the dataset path / target / spec "
        "and try again."
    ),
    ErrorCategory.GUARDRAIL: (
        "🛑 A safety/policy guardrail denied this action. Review the reason; adjust the request "
        "or approve explicitly if intended."
    ),
    ErrorCategory.DEPENDENCY: (
        "🔌 A downstream service or dependency failed. If it persists, check the service status "
        "and logs."
    ),
    ErrorCategory.PERMANENT: (
        "❌ The operation failed in a way a retry won't fix. See the details and adjust the "
        "request."
    ),
}


def format_user_error(exc: BaseException, *, source: str = "") -> str:
    """Return an actionable, human-facing message for ``exc`` (never a raw trace).

    Classifies ``exc`` via the taxonomy, then renders the category's remediation plus a
    compact underlying-error hint. An :class:`AmlasError` that already carries a bespoke
    ``user_message`` uses that verbatim (the caller knows best).
    """
    err = classify(exc, source=source)
    if isinstance(exc, AmlasError) and exc.user_message and exc.user_message != exc.message:
        return exc.user_message
    remediation = _REMEDIATION.get(err.category, _REMEDIATION[ErrorCategory.PERMANENT])
    kind = type(exc).__name__ if not isinstance(exc, AmlasError) else (exc.code or type(exc).__name__)
    return f"{remediation}\n\n_(Underlying: {kind})_"


__all__ = ["format_user_error"]
