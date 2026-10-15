"""CREDS-01 — Bedrock credential hygiene + credential-failure diagnosis.

The single home for the stale-``AWS_BEARER_TOKEN_BEDROCK`` trap guard and the
"is this exception really a creds failure?" translator. Both used to live in the
TUI chat backend (``tui/chat/bedrock_backend.py``), which made them unreachable
from the agent factory: :mod:`amlas._build` cannot import ``amlas.tui.*`` without
inverting the dependency direction, and ``tool_binding_backend`` already had to
reach *into* the TUI backend to get :func:`diagnose_aws_error`. A neutral config
home removes both problems, so every ``BedrockModel`` construction path — not
just the TUI entrypoint — can apply the protection.

Leaf-ish by design: stdlib only plus :mod:`._env_compat` (itself stdlib-only), so
there is no import cycle with the rest of the config seam and no AWS dependency.

The function bodies moved here **verbatim** from the TUI backend; the only change
is that the public entry point dropped its leading underscore
(:func:`neutralize_stale_bearer_token`) now that it is a cross-package API. The
underscore name survived as a module-level alias for exactly as long as the TUI
backend did — it was a bridge for that module's re-export and the two suites that
imported through it, and both went away with the module (issue #35, 6.2), so the
alias went with them. There is now ONE spelling of the entry point;
``_bearer_token_expired`` keeps its underscore because it is genuinely private and
never had a public rename.
"""

from __future__ import annotations

import base64
import datetime
import logging
import os
import urllib.parse

from ._env_compat import env_with_compat

logger = logging.getLogger(__name__)

# CREDS-01 (stale Bedrock bearer-token trap). A GUI/Mantle session commonly exports
# ``AWS_BEARER_TOKEN_BEDROCK`` into the environment (via a LaunchAgent / shell rc).
# boto3's bedrock-runtime client gives that bearer token PRECEDENCE over the AWS
# profile's SigV4 chain — so once it expires, EVERY Converse call fails with
# ``AccessDeniedException: Bearer Token has expired`` even though the operator's
# ``AWS_PROFILE`` is perfectly valid (this is exactly the "why doesn't the TUI use
# my AWS account?" symptom). The p6_* deploy scripts already `unset` it (FIX-10);
# the TUI must do the same before building its client. We neutralize the inherited
# token when it is EXPIRED/undecodable (always safe — an expired token is useless)
# OR when an explicit AWS profile is configured (the operator's clear intent is
# SigV4 via that profile), so the profile chain is used. Set
# ``AMLAS_KEEP_BEDROCK_BEARER_TOKEN=1`` to opt out and force bearer-token auth.
_BEARER_TOKEN_ENV = "AWS_BEARER_TOKEN_BEDROCK"  # nosec B105 — environment variable *name*, not a credential


def _bearer_token_expired(token: str) -> bool | None:
    """Return True if the Bedrock bearer token is expired, False if still valid, None if undecodable.

    The token format is ``bedrock-api-key-<base64>`` where the base64 decodes to a
    presigned-URL query string carrying ``X-Amz-Date`` (start) and ``X-Amz-Expires``
    (TTL seconds). We reconstruct the expiry and compare to ``now`` (UTC).
    """
    try:
        raw = token
        prefix = "bedrock-api-key-"
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
        # base64 in the token may be missing padding — pad to a multiple of 4.
        raw += "=" * (-len(raw) % 4)
        decoded = base64.b64decode(raw).decode("utf-8", "replace")
        params = dict(urllib.parse.parse_qsl(decoded))
        amz_date = params.get("X-Amz-Date")
        expires = int(params.get("X-Amz-Expires", "0"))
        if not amz_date or expires <= 0:
            return None
        start = datetime.datetime.strptime(amz_date, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=datetime.UTC
        )
        exp = start + datetime.timedelta(seconds=expires)
        return datetime.datetime.now(datetime.UTC) >= exp
    except Exception:
        return None


def neutralize_stale_bearer_token(profile: str | None) -> bool:
    """Drop an inherited ``AWS_BEARER_TOKEN_BEDROCK`` so the profile SigV4 chain is used.

    Returns True if the token was removed from ``os.environ``. No-op (returns False)
    when: the token is absent; the operator opted out via
    ``AMLAS_KEEP_BEDROCK_BEARER_TOKEN``; or the token is still VALID and no explicit
    profile is configured (the operator may be relying on bearer-token auth).
    """
    # CREDS-01c: distinguish "var ABSENT" (nothing to do) from "var PRESENT but EMPTY/whitespace".
    # An empty-but-PRESENT ``AWS_BEARER_TOKEN_BEDROCK`` (some shells export ``VAR=``) is the WORST
    # case: boto3/Strands still honors the var and sends an unsigned bearer request → every Converse
    # fails with IncompleteSignatureException ("Authorization header requires Credential/Signature").
    # The old ``if not token`` skipped it (treated empty as absent) so the var survived and broke the
    # tool-agent. Drop an empty/whitespace token unconditionally; only a genuinely-absent var no-ops.
    if _BEARER_TOKEN_ENV not in os.environ:
        return False
    token = os.environ.get(_BEARER_TOKEN_ENV) or ""
    if not token.strip():
        os.environ.pop(_BEARER_TOKEN_ENV, None)
        logger.info(
            "Neutralized empty/whitespace %s — boto3 would otherwise send an unsigned bearer "
            "request (IncompleteSignatureException); using the AWS profile SigV4 chain.",
            _BEARER_TOKEN_ENV,
        )
        return True
    if env_with_compat("AMLAS_KEEP_BEDROCK_BEARER_TOKEN", "").strip().lower() in (
        "1", "true", "yes", "on"
    ):
        return False
    expired = _bearer_token_expired(token)
    # Expired/undecodable → always drop. Valid but a profile is configured → drop
    # (the explicit profile wins). Valid and no profile → keep.
    if expired in (True, None) or profile:
        os.environ.pop(_BEARER_TOKEN_ENV, None)
        logger.info(
            "Neutralized inherited %s (expired=%s, profile=%s) — using AWS profile "
            "SigV4 chain for Bedrock",
            _BEARER_TOKEN_ENV,
            expired,
            profile or "<default-chain>",
        )
        return True
    return False


# Substrings (lower-cased) that mark an AWS credential / auth failure, as opposed
# to a transient model error. Used to turn a cryptic botocore exception into an
# actionable re-auth instruction for the user (an expired AWS SSO session is the
# most common real cause of "errors in prompting").
_CREDENTIAL_ERROR_MARKERS = (
    "expiredtoken",
    "expired token",
    "the security token included in the request is expired",
    "unrecognizedclient",
    "invalidclienttokenid",
    "accessdenied",
    "unable to locate credentials",
    "credentials were not found",
    "could not be found",  # botocore "The config profile (X) could not be found"
    "sso session",
    "token has expired",
    "no credentials",
)


def diagnose_aws_error(exc: Exception, profile: str | None) -> str | None:
    """Return an actionable re-auth message if ``exc`` looks like a creds failure.

    Returns ``None`` when the error is NOT credential-related (so the caller keeps
    its normal model/runtime error rendering). When it IS, the returned string
    tells the user exactly how to recover — the profile name + the precise
    ``aws sso login`` command (or refreshing the standard boto3 credential chain)
    — instead of a raw botocore trace.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    if not any(marker in text for marker in _CREDENTIAL_ERROR_MARKERS):
        return None
    prof = profile or "your default AWS profile"
    login = (
        f"aws sso login --profile {profile}" if profile else "aws sso login"
    )
    return (
        f"🔐 **AWS credentials for `{prof}` are missing or expired.**\n"
        f"The chat agent could not reach Bedrock. Re-authenticate, then relaunch:\n\n"
        f"  • AWS SSO:            `{login}`\n"
        f"  • or refresh your standard AWS credential chain "
        f"(env vars / shared config / instance role)\n\n"
        f"_(Underlying error: {type(exc).__name__})_"
    )
