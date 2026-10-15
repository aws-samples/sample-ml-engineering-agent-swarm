"""deployment_agent sub-agent (D3.a) — NEW in M4 (doc 06 §Subagents; ADR-007).

A bounded executor of the Stage-10 deployment tools over the ``DeploymentBackend`` port.
Mirrors the SageMaker Ops sub-agent ([`subagents/sagemaker_ops.py`](sagemaker_ops.py:40)):
``build_agent`` + ``stack_prompts`` + the ``hooks_factory`` fan-out. The Deploy/Serving
Lead (D3) composes it with ``.as_tool(preserve_context=True)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import DEPLOYMENT_AGENT, build_subagent


def build_deployment_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "deployment_agent",
) -> Agent:
    """Build the deployment_agent sub-agent (D3.a).

    Args:
        model: Optional pre-built model; None (default) uses real Bedrock. Callers may pass a real BedrockModel.
        hooks_factory: Optional zero-arg callable forwarded to ``build_agent`` so the
            Orchestrator can attach a ``MemoryHookProvider``.
        name: Agent name — defaults to "deployment_agent".
    """
    return build_subagent(
        DEPLOYMENT_AGENT,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
