"""Research Discovery sub-agent — discovers relevant algorithms for a task.

System prompt stack: paranoid_scientist + research_discovery.
Tools: search_algorithms, recommend_approaches.

The sub-agent takes a SpecModel (as context in the prompt) and uses research tools
to discover the most suitable algorithms and experiment suggestions
for the given ML task.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import RESEARCH_DISCOVERY, build_subagent


def build_research_discovery_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "research_discovery",
) -> Agent:
    """Build the Research Discovery sub-agent.

    Args:
        model: Optional pre-built model; None (default) uses real Bedrock. Callers may pass a real BedrockModel (real_bedrock fixture).
        hooks_factory: Optional zero-arg callable forwarded to ``build_agent``
            so the Orchestrator can attach a ``MemoryHookProvider`` scoped to
            ``(memory_id, actor_id=project_id, session_id=run_id)``.
        name: Agent name — defaults to "research_discovery" (used in traces + a2a).
    """
    return build_subagent(
        RESEARCH_DISCOVERY,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
