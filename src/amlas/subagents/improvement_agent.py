"""improvement_agent sub-agent (M1.b) — NEW in M5 (doc 06 §Subagents; ADR-008/003).

A bounded executor of the Stage-12 continuous-improvement tools — failure triage, AgentCore-Memory
carry-over, and lineage-preserving retirement. Mirrors the deployment_agent (D3.a):
``build_agent`` + ``stack_prompts`` + the ``hooks_factory`` fan-out. The Monitoring Lead (M1)
composes it with ``.as_tool(preserve_context=True)``. The retrain itself RE-ENTERS the unified
exploration retreat loop (ADR-008) under the non-delegated Arbiter (ADR-003) — this agent prepares
the carry-context + retires; it never trains and never grades.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import IMPROVEMENT_AGENT, build_subagent


def build_improvement_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "improvement_agent",
) -> Agent:
    """Build the improvement_agent sub-agent (M1.b).

    Args:
        model: Optional pre-built model; None (default) uses real Bedrock. Callers may pass a real BedrockModel.
        hooks_factory: Optional zero-arg callable forwarded to ``build_agent`` so the
            Orchestrator can attach a ``MemoryHookProvider``.
        name: Agent name — defaults to "improvement_agent".
    """
    return build_subagent(
        IMPROVEMENT_AGENT,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
