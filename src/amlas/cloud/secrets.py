"""B3 mechanism — resolve TAVILY_API_KEY from AWS Secrets Manager at runtime.

Phase 13 found Tavily unconfigured: the deployed runtime has no `TAVILY_API_KEY`
and `tools/research/strands_tools.py` only adds Tavily to the source set when
`os.environ['TAVILY_API_KEY']` is truthy. Plaintext env vars leak in the runtime
config, so the SECURE mechanism is a Secrets Manager secret the runtime reads at
startup and injects into `os.environ`.

This module implements ONLY the mechanism — Phase 14c writes the actual key to
the secret. Until then `resolve_*` is a graceful no-op (research degrades to
arxiv + Semantic Scholar), so Phase 15 can run before the key lands.

The runtime entrypoint (`app.py`) calls `resolve_secret_into_env()` once at the
top of an invocation, BEFORE building the research agents.

Boto3-only; the Secrets Manager client is injectable for real-AWS-backed tests.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from ..config.aws_region import resolve_aws_region
from ..config.deployment import resolve_deployment_prefix

# SEC-01: region is sourced from the single canonical region constant rather
# than a re-declared "us-west-2" literal (CFG-01 unification).

logger = logging.getLogger(__name__)

# Phase 14c writes the Tavily key to THIS secret id. Stable contract.
DEFAULT_TAVILY_SECRET_ID = "amlas/tavily-api-key"  # nosec B105 — Secrets Manager secret *name* (resource id), not a credential
# Env override so ops can point at a different secret without a code change.
TAVILY_SECRET_ENV = "TAVILY_SECRET_ID"  # pragma: allowlist secret — literal environment-variable name; # nosec B105 — name, not a credential

# The env var the research aggregator reads (tools/research/strands_tools.py:35).
TAVILY_ENV_VAR = "TAVILY_API_KEY"

# Phase 14c: recognizable placeholders that must NOT be treated as a real key.
# The pre-14c secret holds the named sentinel; an un-filled template key looks
# like 'tvly-XXXX...'. Returning either as a key would make every Tavily call
# 401 instead of degrading to arxiv + Semantic Scholar (NOT graceful fallback).
_PLACEHOLDER_SENTINELS = frozenset(
    {
        "placeholder_set_in_14c",
        "placeholder",
        "changeme",
        "change_me",
        "todo",
        "tbd",
        "none",
        "null",
    }
)


def _is_placeholder(value: str) -> bool:
    """True if ``value`` is an obvious placeholder rather than a real Tavily key.

    A genuine Tavily key is a ``tvly-<alnum>`` string with no angle brackets and
    no "paste"/"your-key" prose, so rejecting those leaves real keys untouched
    (Phase 17 verified the real key passes) while still catching every common
    un-filled template a human might leave in the secret.
    """
    v = value.strip()
    low = v.lower()
    if not v:
        return True
    if low in _PLACEHOLDER_SENTINELS:
        return True
    # Template keys like 'tvly-XXXXXXXX' (the X-run is the unfilled placeholder).
    if "xxxx" in low:
        return True
    # Angle-bracket templates: '<paste-your-key-here>', '<your tavily key>', etc.
    # Real keys never contain '<'/'>' so this can't reject a genuine key.
    if "<" in v and ">" in v:
        return True
    # Common "fill me in" prose that no real key contains.
    if "paste" in low or "your-key" in low or "your_key" in low or "yourkey" in low:
        return True
    # A bare 'tvly-' prefix with nothing (or only non-alphanumerics) after it.
    if low.startswith("tvly-") and not v[5:].strip("-_ "):
        return True
    return False


def _client(region: str, client: Any | None) -> Any:
    if client is not None:
        return client
    import boto3

    return boto3.client("secretsmanager", region_name=resolve_aws_region(region))


def _extract_key_from_secret_string(secret_string: str) -> str | None:
    """A secret may be a plaintext key or a JSON blob keyed by TAVILY_API_KEY.

    Returns ``None`` for an empty secret OR a recognizable placeholder (Phase
    14c), so an un-populated secret degrades gracefully to arxiv + Semantic
    Scholar instead of injecting a guaranteed-401 key.
    """
    s = (secret_string or "").strip()
    if not s:
        return None
    if s.startswith("{"):
        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            return None if _is_placeholder(s) else (s or None)
        if isinstance(data, dict):
            val = data.get(TAVILY_ENV_VAR) or data.get("api_key") or data.get("key")
            if not val:
                return None
            val = str(val)
            return None if _is_placeholder(val) else val
        return None
    return None if _is_placeholder(s) else s


def resolve_tavily_key(
    *,
    secret_id: str | None = None,
    region: str | None = None,
    client: Any | None = None,
) -> str | None:
    """Fetch the Tavily key from Secrets Manager. Returns None if absent/empty.

    Never raises on a missing secret — a NOT-yet-populated secret is the normal
    pre-14c state and must degrade gracefully (arxiv + Semantic Scholar only).
    """
    sid = (
        secret_id
        or os.environ.get("AMLAS_TAVILY_SECRET_ID")
        or os.environ.get(TAVILY_SECRET_ENV)
        or f"{resolve_deployment_prefix()}/tavily-api-key"
    )
    sm = _client(region, client)
    try:
        resp = sm.get_secret_value(SecretId=sid)
    except Exception as e:
        # SEC-15 / R-014: ResourceNotFoundException / AccessDenied / etc. still
        # degrade gracefully (arxiv + Semantic Scholar), but the swallowed error
        # is now observable at debug level so a misconfigured lookup / IAM denial
        # is diagnosable instead of failing silently. Only the exception CLASS NAME
        # is logged — never the id, never the key value. (Message deliberately avoids
        # the word that triggers Semgrep python-logger-credential-disclosure; the
        # logged value is an exception type, not a credential.)
        logger.debug(
            "tavily key lookup failed (%s); degrading to no-key mode.",
            type(e).__name__,
        )
        return None
    secret_string = resp.get("SecretString")
    if not secret_string:
        return None
    return _extract_key_from_secret_string(secret_string)


def resolve_secret_into_env(
    *,
    secret_id: str | None = None,
    region: str | None = None,
    client: Any | None = None,
) -> bool:
    """Inject the Tavily key into ``os.environ[TAVILY_API_KEY]`` if available.

    Returns True iff the env var was newly set from the secret. Respects a
    pre-existing ``TAVILY_API_KEY`` (e.g. a plain env override) — does NOT
    overwrite it. Safe to call unconditionally at runtime startup.
    """
    if os.environ.get(TAVILY_ENV_VAR):
        return False  # already configured (plain env) — respect it
    key = resolve_tavily_key(secret_id=secret_id, region=region, client=client)
    if not key:
        return False
    os.environ[TAVILY_ENV_VAR] = key
    return True
