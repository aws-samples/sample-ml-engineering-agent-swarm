"""Environment-variable compatibility seam.

This is the SINGLE place the ``AMLAS_*`` env-var rename's one-version back-compat
alias logic lives — do NOT scatter prefix-fallback logic across call sites.

The historical ``AGENTICML_*`` prefix was renamed to the canonical ``AMLAS_*``
prefix. To avoid breaking existing deployments whose ``.env`` still exports the old name, this
module resolves a canonical ``AMLAS_<SUFFIX>`` value with a fixed precedence:

    1. ``AMLAS_<SUFFIX>``        — the canonical name; wins whenever it is set.
    2. ``AGENTICML_<SUFFIX>``    — legacy fallback (one-time DeprecationWarning).
    3. the caller's ``default``  — when neither name is set.

The suffix is identical across prefixes (for example,
``AMLAS_S3_BUCKET`` ⇄ ``AGENTICML_S3_BUCKET``), so the
fallback is derived by prefix-swap — no per-variable table to drift.

Leaf module by design: it imports only the stdlib (``os`` + ``warnings``) and
NOTHING from this package, so the config modules that form the resolution seam
(:mod:`.externalized`, :mod:`.settings`, :mod:`.runtime`, :mod:`.guardrails`) can
all import it without an import cycle.
"""

from __future__ import annotations

import os
import re
import warnings
from typing import overload

# Canonical public environment-variable prefix.
CANONICAL_ENV_PREFIX = "AMLAS_"

# F15 (unfilled-template-placeholder guard, centralized): ``.env.example`` ships
# ``KEY=<PLACEHOLDER>`` tokens that ``cp .env.example .env`` copies verbatim. An
# operator who never fills one leaves a NON-EMPTY literal like
# ``AMLAS_TRUSTED_MODEL_ROOT=<TRUSTED_MODEL_ROOT>`` or ``AMLAS_S3_BUCKET=<LAKE_BUCKET>``
# in the environment. Read naively that placeholder is truthy and silently
# poisons downstream logic (e.g. it shadowed a test's real trusted-root override and
# made an edge-deploy resolve ``repo_root/<TRUSTED_MODEL_ROOT>`` → pkl load refused).
# ``infra/app.py`` already coerces the KMS-ARN placeholder to empty; this is the DRY
# home for that rule so EVERY env consumer (incl. spawned subprocesses reading the
# helper) treats an UNFILLED, purely-``<...>``-shaped value as "unset". A value that
# merely CONTAINS a ``<...>`` substring inside real text (e.g. a bucket name) is left
# untouched — only a token that is ENTIRELY a single angle-bracket placeholder is
# neutralized. Zero brackets in a genuine value ⇒ never matched.
_UNFILLED_PLACEHOLDER = re.compile(r"^<[A-Za-z0-9_]+>$")


def _is_unfilled_placeholder(value: str) -> bool:
    """True iff ``value`` is an UNFILLED ``<PLACEHOLDER>`` token (F15).

    Matches only a value that is ENTIRELY a single ``<UPPER_SNAKE>``-style
    angle-bracket token (optionally surrounded by whitespace) — the exact shape
    ``.env.example`` ships. Real values that merely embed such a token as a
    substring (``arn:...:<AWS_ACCOUNT_ID>:...``, ``amlas-lake-<AWS_ACCOUNT_ID>-...``)
    are NOT matched, so only never-filled templates are neutralized.
    """
    return bool(_UNFILLED_PLACEHOLDER.match(value.strip()))

# Generic legacy prefix still honored for compatibility.
LEGACY_ENV_PREFIXES: tuple[str, ...] = ("AGENTICML_",)

# One-time deprecation bookkeeping: warn ONCE per distinct legacy name resolved,
# so a long-running process / busy test session is not spammed.
_WARNED_LEGACY: set[str] = set()


def warn_legacy_env(legacy_name: str, canonical_name: str) -> None:
    """Emit a one-time ``DeprecationWarning`` that ``legacy_name`` is deprecated.

    Idempotent per ``legacy_name`` (subsequent calls for the same legacy var are
    no-ops) so the notice surfaces once without flooding logs/test output.
    """
    if legacy_name in _WARNED_LEGACY:
        return
    _WARNED_LEGACY.add(legacy_name)
    warnings.warn(
        f"environment variable {legacy_name!r} is deprecated; use {canonical_name!r} "
        "instead (the legacy name is still honored for one release).",
        DeprecationWarning,
        stacklevel=3,
    )


def legacy_names_for(canonical_name: str) -> tuple[str, ...]:
    """Return the legacy env names a canonical ``AMLAS_<SUFFIX>`` falls back to.

    Empty when ``canonical_name`` is not ``AMLAS_``-prefixed (nothing to alias).
    """
    if not canonical_name.startswith(CANONICAL_ENV_PREFIX):
        return
    suffix = canonical_name[len(CANONICAL_ENV_PREFIX):]
    return tuple(prefix + suffix for prefix in LEGACY_ENV_PREFIXES)


@overload
def env_with_compat(name: str, default: str, *, warn: bool = ...) -> str: ...
@overload
def env_with_compat(name: str, default: None = ..., *, warn: bool = ...) -> str | None: ...


def env_with_compat(name: str, default: str | None = None, *, warn: bool = True) -> str | None:
    """Resolve a canonical ``AMLAS_*`` env var with legacy prefix back-compat.

    Precedence: ``AMLAS_<SUFFIX>`` → ``AGENTICML_<SUFFIX>`` → ``default``.
    The canonical name wins whenever it is present (even if set to an
    empty string — an explicit override), mirroring ``os.environ.get`` semantics.
    When a legacy name supplies the value, a one-time ``DeprecationWarning`` fires
    (suppress with ``warn=False``).

    Args:
        name: the canonical env var to read (normally ``AMLAS_`` prefixed).
        default: returned when neither the canonical nor any legacy name is set.
        warn: emit the one-time deprecation notice on a legacy hit (default True).

    Returns:
        The resolved value, or ``default`` when nothing is set.
    """
    # 1. Canonical wins whenever present (matches os.environ.get: empty str counts),
    #    UNLESS it is an unfilled ``<PLACEHOLDER>`` token (F15) — then fall through as
    #    if unset so a never-filled ``.env.example`` copy cannot poison the value.
    if name in os.environ and not _is_unfilled_placeholder(os.environ[name]):
        return os.environ[name]
    # 2. Legacy prefix fallback (derived by suffix swap). The
    #    same unfilled-placeholder guard applies to the legacy names.
    for legacy in legacy_names_for(name):
        if legacy in os.environ and not _is_unfilled_placeholder(os.environ[legacy]):
            if warn:
                warn_legacy_env(legacy, name)
            return os.environ[legacy]
    # 4. Caller default.
    return default


def apply_env_compat(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Normalize legacy env vars into their canonical ``AMLAS_*`` names in-place.

    For every ``AGENTICML_<SUFFIX>`` present whose canonical
    ``AMLAS_<SUFFIX>`` is NOT already set, copy the legacy value onto the canonical
    name (one-time deprecation warned). This is the belt-and-suspenders companion to
    :func:`env_with_compat` for code paths / spawned subprocesses that read
    ``os.environ`` directly without going through the helper. Canonical names that
    are already set are left untouched (canonical wins).

    Args:
        environ: the mapping to normalize (defaults to ``os.environ``).

    Returns:
        A dict of the ``{canonical: value}`` pairs newly written (for inspection).
    """
    env = os.environ if environ is None else environ
    written: dict[str, str] = {}
    # Snapshot keys first — we mutate ``env`` while iterating.
    for key in list(env.keys()):
        for legacy_prefix in LEGACY_ENV_PREFIXES:
            if not key.startswith(legacy_prefix):
                continue
            suffix = key[len(legacy_prefix):]
            canonical = CANONICAL_ENV_PREFIX + suffix
            if canonical in env and not _is_unfilled_placeholder(env[canonical]):
                break  # canonical already set (real value) — wins, leave legacy as-is
            value = env[key]
            if _is_unfilled_placeholder(value):
                break  # legacy value is itself an unfilled placeholder — nothing to promote
            env[canonical] = value
            written[canonical] = value
            warn_legacy_env(key, canonical)
            break
    return written
