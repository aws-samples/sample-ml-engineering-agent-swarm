"""DQ Assessment sub-agent (D0.a) — pattern scan + missingness diagnosis.

System prompt stack: paranoid_scientist + data_verification_checklist + dq_assessment.md.
Tools (ADR-0002 — DISPATCH, not in-sandbox scan): run_profile, recommend_dq_rules, evaluate_dq
(port-neutral → Glue Data Quality / SageMaker Processing).

The sub-agent returns a DQ-report JSON (patterns_detected / dimensions / mechanism /
verdict). Master plan §3 (D0.a), spec sheet 3.1.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import DQ_ASSESSMENT, build_subagent

# ADR-0002: DQ scanning is a data-plane WORKLOAD — dispatch it to Glue Data Quality / SageMaker
# Processing via the port-neutral tools, rather than running the 25-pattern pandas scan in the
# AgentCore sandbox. The agent orchestrates: profile → recommend rules → evaluate → report.


def build_dq_assessment_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "dq_assessment",
) -> Agent:
    """Build the DQ Assessment sub-agent (D0.a).

    Args:
        model: Optional pre-built model; None (default) uses real Bedrock. Callers may pass a real BedrockModel (real_bedrock fixture).
        hooks_factory: Optional zero-arg callable forwarded to ``build_agent``
            so the Orchestrator can attach a ``MemoryHookProvider`` scoped to
            ``(memory_id, actor_id=project_id, session_id=run_id)``.
        name: Agent name — defaults to "dq_assessment" (used in traces + a2a).
    """
    return build_subagent(
        DQ_ASSESSMENT,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
