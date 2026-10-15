"""Deploy/Serving Lead (D3) — NEW in M4 (doc 06 §Leads; ADR-007).

The 6th Lead, owning the Stage-9 → Stage-10 hand-off (production-readiness review → deploy).
Composes three NEW sub-agents as tools — ``readiness_review_agent`` (the 10 readiness gates),
``deployment_agent`` (package/deploy/rollout/rollback/release), and ``governance_agent`` (model
cards/lineage/access) — and owns the ``DeploymentBackend`` port (the priority GAP). It drives
``readiness_flow`` then, on a go, ``deploy_flow`` (both ``plan_approve_execute``-gated).

Mirrors the MLOps Lead (M0) wiring EXACTLY — ``build_agent`` +
``.as_tool(preserve_context=True)`` + ``stack_prompts`` + the ``hooks_factory`` fan-out so each
sub-agent also receives the ``MemoryHookProvider``. This keeps it consistent with the existing
5-Lead structure (the Monitoring Lead M1 arrives in M5).

The composition lives in [`declarations.DEPLOY_SERVING_LEAD`](declarations.py:1); this module
is the signature-preserving builder over it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import DEPLOY_SERVING_LEAD, build_lead

#: Repo-wide convention: the prompt files stacked into this agent's system prompt.
#: Sourced from the declaration so the two can never disagree.
SYSTEM_PROMPT_SOURCES = DEPLOY_SERVING_LEAD.prompt_sources


def build_deploy_serving_lead(
    *,
    model: Any | None = None,
    deployment_model: Any | None = None,
    readiness_model: Any | None = None,
    governance_model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "deploy_serving_lead",
) -> Agent:
    """Build the Deploy/Serving Lead (D3) agent.

    ``hooks_factory`` (when provided by the Orchestrator) is forwarded to ``build_agent``
    for the Lead itself AND to every sub-agent constructor so the ``.as_tool(...)``-wrapped
    sub-agents also receive a ``MemoryHookProvider``. The default ``None`` preserves the existing
    import-time behavior.

    Args:
        model: Lead model; None (default) → real Bedrock. Callers may inject a real BedrockModel.
        deployment_model / readiness_model / governance_model: Optional sub-agent model overrides.
        hooks_factory: Optional zero-arg memory-hook factory fanned out to subs.
        name: Agent name — defaults to "deploy_serving_lead".
    """
    return build_lead(
        DEPLOY_SERVING_LEAD,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
        readiness_model=readiness_model,
        deployment_model=deployment_model,
        governance_model=governance_model,
    )
