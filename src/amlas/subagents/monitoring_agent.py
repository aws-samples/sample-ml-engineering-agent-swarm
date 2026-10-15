"""monitoring_agent sub-agent (M1.a) — NEW in M5 (doc 06 §Subagents; ADR-008).

A bounded executor of the Stage-11 monitoring tools over the ``MonitoringBackend`` port.
Mirrors the deployment_agent (D3.a, [`subagents/deployment_agent.py`](deployment_agent.py:39)):
``build_agent`` + ``stack_prompts`` + the ``hooks_factory`` fan-out. The Monitoring Lead
(M1) composes it with ``.as_tool(preserve_context=True)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import MONITORING_AGENT, build_subagent


def build_monitoring_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "monitoring_agent",
) -> Agent:
    """Build the monitoring_agent sub-agent (M1.a).

    Args:
        model: Optional pre-built model; None (default) uses real Bedrock. Callers may pass a real BedrockModel.
        hooks_factory: Optional zero-arg callable forwarded to ``build_agent`` so the
            Orchestrator can attach a ``MemoryHookProvider``.
        name: Agent name — defaults to "monitoring_agent".
    """
    return build_subagent(
        MONITORING_AGENT,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
