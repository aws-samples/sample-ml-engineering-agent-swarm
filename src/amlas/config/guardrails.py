"""WA-02 — in-band Bedrock Guardrail configuration (responsible AI, always-on filtering).

The audit found a real ``BedrockGuardrailBackend`` existed but was only invoked as a
Stage-9 evidence tool — the live ``BedrockModel`` attached **no** ``guardrailConfig``, so
harmful-output / prompt-injection filtering was NOT enforced in-band on every model turn
(GenAI Lens ``AGENTSEC04``/``AGENTSEC08``). This module is the single source the agent
factory (``_build.py``) and the TUI chat backend read to bind a guardrail on EVERY Bedrock
invocation.

Configuration (OSS-portable; secure default ON when a guardrail is provisioned).
OSS-4b P3: the canonical env names are ``AMLAS_GUARDRAIL_*``; the legacy
``AMLAS_GUARDRAIL_*`` names are still honored for one release via the
:func:`env_with_compat` seam (one-time deprecation):
    * ``AMLAS_GUARDRAIL_ID``       — the Bedrock Guardrail identifier. The ``GuardrailStack``
      publishes it to SSM ``/amlas/guardrail-id``; deployers thread it into the runtime
      env. EMPTY (the
      offline/unprovisioned default) → guardrails are a NO-OP, so offline offline validation and a
      fresh checkout build byte-for-byte as before.
    * ``AMLAS_GUARDRAIL_VERSION``  — the guardrail version (default ``DRAFT``).
    * ``AMLAS_GUARDRAIL_ENABLED``  — explicit opt-out seam (default ``true``); set ``false``
      to disable in-band binding even when an id is present.
    * ``AMLAS_GUARDRAIL_TRACE``    — trace mode (default ``enabled`` so interventions are
      observable in traces).

Design: a guardrail is bound IFF an id is configured AND not explicitly disabled. This keeps
the secure default (filter on whenever a guardrail exists) without hardcoding any account-
specific id (the id is supplied entirely by the adopter's environment / SSM).
"""

from __future__ import annotations

from typing import Any

# OSS-4b P3: single env back-compat seam — canonical ``AMLAS_GUARDRAIL_*`` with
# legacy ``AMLAS_GUARDRAIL_*`` fallback (one-time deprecation).
from ._env_compat import env_with_compat

# SSM key the GuardrailStack publishes the id/version under (adopters read these into env).
SSM_GUARDRAIL_ID = "/amlas/guardrail-id"
SSM_GUARDRAIL_VERSION = "/amlas/guardrail-version"

_DEFAULT_VERSION = "DRAFT"
_DEFAULT_TRACE = "enabled"


def _truthy(value: str | None, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def guardrail_enabled() -> bool:
    """True when an in-band guardrail should be bound (id present AND not disabled)."""
    gid = (env_with_compat("AMLAS_GUARDRAIL_ID", "") or "").strip()
    if not gid:
        return False
    return _truthy(env_with_compat("AMLAS_GUARDRAIL_ENABLED"), default=True)


def guardrail_model_kwargs() -> dict[str, Any]:
    """Strands ``BedrockModel`` guardrail kwargs — bound in-band on every model turn (WA-02).

    Returns the ``guardrail_id`` / ``guardrail_version`` / ``guardrail_trace`` kwargs the
    Strands ``BedrockModel`` constructor accepts, OR an empty dict when no guardrail is
    configured (offline-safe). Threading these onto the live model makes input+output
    filtering always-on, not just gate-time.
    """
    if not guardrail_enabled():
        return {}
    return {
        "guardrail_id": (env_with_compat("AMLAS_GUARDRAIL_ID", "") or "").strip(),
        "guardrail_version": env_with_compat("AMLAS_GUARDRAIL_VERSION", _DEFAULT_VERSION).strip(),
        "guardrail_trace": env_with_compat("AMLAS_GUARDRAIL_TRACE", _DEFAULT_TRACE).strip(),
    }


def converse_guardrail_config() -> dict[str, Any] | None:
    """The Bedrock Converse-API ``guardrailConfig`` block (TUI chat path), or ``None``.

    Same source of truth as :func:`guardrail_model_kwargs`, shaped for the raw
    ``bedrock-runtime.converse`` call the TUI uses (``guardrailIdentifier`` /
    ``guardrailVersion`` / ``trace``). ``None`` when unconfigured so the TUI omits the
    block entirely (unchanged offline behavior).
    """
    if not guardrail_enabled():
        return None
    return {
        "guardrailIdentifier": (env_with_compat("AMLAS_GUARDRAIL_ID", "") or "").strip(),
        "guardrailVersion": env_with_compat("AMLAS_GUARDRAIL_VERSION", _DEFAULT_VERSION).strip(),
        "trace": env_with_compat("AMLAS_GUARDRAIL_TRACE", _DEFAULT_TRACE).strip(),
    }
