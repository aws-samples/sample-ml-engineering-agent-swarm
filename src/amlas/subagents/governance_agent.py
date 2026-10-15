"""governance_agent sub-agent (D3.c / M0.b) — NEW in M4 (doc 06 §Subagents).

Owner of model-governance touchpoints over the ``GovernanceBackend`` port: model cards, lineage,
and data-access checks. Composed by both the Deploy/Serving Lead (D3) and the MLOps Lead (M0).
Mirrors the SageMaker Ops sub-agent pattern.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import GOVERNANCE_AGENT, build_subagent


def build_governance_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "governance_agent",
) -> Agent:
    """Build the governance_agent sub-agent (D3.c / M0.b).

    Composes the GovernanceBackend port tools (model cards / lineage / access checks) so the
    readiness gates have governance evidence and a deployed model is auditable.
    """
    return build_subagent(
        GOVERNANCE_AGENT,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
