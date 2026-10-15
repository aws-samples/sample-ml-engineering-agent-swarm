"""Imputation Strategist sub-agent (D0.b) — pick + prove a per-column plan.

System prompt stack: paranoid_scientist + data_steering (the dial) +
imputation_strategist.md.
Tools (ADR-0002 — DISPATCH, not in-sandbox sklearn imputers): apply_imputation_via_port
(port-neutral → SageMaker Processing). The agent still CHOOSES the per-column strategy.

HARD steering (forbid/pin) is enforced *inside* ``apply_imputation_plan`` — the
agent proposes, the tool enforces. Master plan §6 / §3.1, spec sheet 3.2.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import IMPUTATION_STRATEGIST, build_subagent

# ADR-0002: applying an imputation plan is a data-plane WORKLOAD — dispatch it to SageMaker
# Processing via the port-neutral tool, rather than running sklearn imputers in the AgentCore
# sandbox. The agent orchestrates: choose the per-column strategy (its judgment) → dispatch the
# apply → the Processing job writes the imputed dataset + audit manifest to S3.


def build_imputation_strategist_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "imputation_strategist",
) -> Agent:
    """Build the Imputation Strategist sub-agent (D0.b).

    Args:
        model: Optional pre-built model; None (default) uses real Bedrock. Callers may pass a real BedrockModel (real_bedrock fixture).
        hooks_factory: Optional zero-arg callable forwarded to ``build_agent``
            so the Orchestrator can attach a ``MemoryHookProvider``.
        name: Agent name — defaults to "imputation_strategist".
    """
    return build_subagent(
        IMPUTATION_STRATEGIST,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
