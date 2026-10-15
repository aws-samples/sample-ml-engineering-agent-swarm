"""Backend-neutral governance tools — Stage-9 governance touchpoints (M4; doc 06 §T-GOV).

The doc 05 §4.4 EVOLVE pattern for the GovernanceBackend port: each tool resolves
:class:`~amlas.backends.ports.GovernanceBackend` (Local markdown card offline,
SageMaker Model Cards managed) and a guardrail check via :class:`GuardrailBackend`. NEW M4
tools, ADDITIVE — they supply the governance/safety evidence the readiness gates consult.

  * ``create_model_card`` (T-GOV-01/03) — compose a model card (intended use / risk / metrics).
  * ``query_lineage`` (T-GOV-04) — surface an artifact/model's upstream/downstream lineage.
  * ``check_data_access`` (T-GOV-05) — the data-access gate (LakeFormation managed).
  * ``run_guardrail_check`` (T-GUARD-01, mode=monitor) — content-safety check for the Safety gate.

All read-only / CHEAP — they inform the readiness review; they never deploy or grant access
(grant-access is privileged + HITL-gated, M6 breadth).
"""

from __future__ import annotations

from typing import Any

from strands import tool

from ..backends.ports import GovernanceBackend, GuardrailBackend
from ..backends.registry import REGISTRY
from ..backends.types import ModelRef
from ..errors import tool_error_boundary

__all__ = ["check_data_access", "create_model_card", "query_lineage", "run_guardrail_check"]


def _to_ref(model: Any) -> ModelRef:
    """Coerce a ModelRef / dict / URI into a domain :class:`ModelRef`."""
    if isinstance(model, ModelRef):
        return model
    if isinstance(model, str):
        return ModelRef(uri=model, backend="local")
    if isinstance(model, dict):
        return ModelRef(uri=str(model.get("uri", model.get("model_path", ""))),
                        algo=str(model.get("algo", model.get("algorithm", ""))),
                        metrics={k: float(v) for k, v in (model.get("metrics") or {}).items()
                                 if isinstance(v, (int, float))})
    return ModelRef(uri=str(model), backend="local")


@tool
@tool_error_boundary(source="tools.governance_neutral")
def create_model_card(
    model: Any,
    intended_use: str = "",
    risk_rating: str = "low",
    metrics: dict[str, Any] | None = None,
    observations: str = "",
) -> dict[str, Any]:
    """Compose a model card for ``model`` via :class:`GovernanceBackend` (T-GOV-01/03).

    Resolves the governance port (local markdown card offline, SageMaker Model Cards managed) and
    builds a card capturing intended use, risk rating, metrics, and observations.
    """
    try:
        resolved = REGISTRY.resolve(GovernanceBackend)
        out: dict[str, Any] = resolved.adapter.model_card(
            model_ref=_to_ref(model),
            content={"intended_use": intended_use, "risk_rating": risk_rating,
                     "metrics": metrics or {}, "observations": observations},
        )
        out.setdefault("adapter", resolved.adapter_name)
        return out
    except Exception as e:
        return {"success": False, "error": f"create_model_card failed: {e}"}


@tool
@tool_error_boundary(source="tools.governance_neutral")
def query_lineage(ref: str) -> dict[str, Any]:
    """Surface the upstream/downstream lineage of ``ref`` via :class:`GovernanceBackend` (T-GOV-04)."""
    try:
        resolved = REGISTRY.resolve(GovernanceBackend)
        out: dict[str, Any] = resolved.adapter.lineage(ref=ref)
        out.setdefault("adapter", resolved.adapter_name)
        return out
    except Exception as e:
        return {"success": False, "error": f"query_lineage failed: {e}"}


@tool
@tool_error_boundary(source="tools.governance_neutral")
def check_data_access(principal: str, resource: str) -> dict[str, Any]:
    """Check ``principal``'s access to ``resource`` via :class:`GovernanceBackend` (T-GOV-05).

    The data-access gate (LakeFormation managed; allow-with-note offline). Read-only — grant is
    privileged + HITL-gated (M6).
    """
    try:
        resolved = REGISTRY.resolve(GovernanceBackend)
        out: dict[str, Any] = resolved.adapter.check_access(principal=principal, resource=resource)
        out.setdefault("adapter", resolved.adapter_name)
        return out
    except Exception as e:
        return {"success": False, "error": f"check_data_access failed: {e}"}


@tool
@tool_error_boundary(source="tools.governance_neutral")
def run_guardrail_check(
    text: str, policy: dict[str, Any] | None = None, untrusted: bool = False
) -> dict[str, Any]:
    """Run a content-safety guardrail check via :class:`GuardrailBackend` (T-GUARD-01, monitor).

    Resolves the guardrail port (in-process content filter offline, Bedrock Guardrails managed)
    and returns ``{"action","passed","violations"}`` — the Safety/Security/Privacy gate evidence.

    ``untrusted`` (SEC-A16): set True when ``text`` comes from an UNTRUSTED research/web/Tavily
    path the agent ingests. On the untrusted path the in-process guardrail runs prompt-injection
    detection by DEFAULT (OWASP LLM01); an explicit ``policy["detect_injection"]`` still wins.
    Forwarded to adapters that accept it (the in-process filter); managed adapters that don't
    accept the kwarg get the call without it (back-compat).
    """
    try:
        resolved = REGISTRY.resolve(GuardrailBackend)
        out: dict[str, Any]
        try:
            out = resolved.adapter.check(text=text, policy=policy, untrusted=untrusted)
        except TypeError:
            # Adapter predates the SEC-A16 untrusted seam (e.g. managed Bedrock path) — call
            # without it; the managed Guardrail PROMPT_ATTACK filter is always-on regardless.
            out = resolved.adapter.check(text=text, policy=policy)
        out.setdefault("adapter", resolved.adapter_name)
        return out
    except Exception as e:
        return {"success": False, "error": f"run_guardrail_check failed: {e}"}
