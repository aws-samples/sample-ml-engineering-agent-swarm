"""Data Integrity sub-agent — block-or-pass verdict at the start of D1.

System prompt stack: paranoid_scientist + data_verification_checklist + data_integrity.md.
Tools: data_verify_tool, leakage_check_tool (both @tool-wrapped, path-based).

The sub-agent returns JSON with verdict=pass|block. Caller (FE Lead) halts on block.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import DATA_INTEGRITY, build_subagent


def build_data_integrity_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "data_integrity",
) -> Agent:
    """Build the Data Integrity sub-agent.

    Args:
        model: Optional pre-built model; None (default) uses real Bedrock. Callers may pass a real BedrockModel (real_bedrock fixture).
        hooks_factory: Optional zero-arg callable forwarded to ``build_agent``
            so the Orchestrator can attach a ``MemoryHookProvider`` scoped to
            ``(memory_id, actor_id=project_id, session_id=run_id)``.
        name: Agent name — defaults to "data_integrity" (used in traces + a2a).
    """
    return build_subagent(
        DATA_INTEGRITY,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
