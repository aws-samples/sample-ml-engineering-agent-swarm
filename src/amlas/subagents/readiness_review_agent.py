"""readiness_review_agent sub-agent (D3.b) — NEW in M4 (doc 06 §Subagents).

A bounded executor of the Stage-9 readiness tools: the 10 production-readiness gates + risk
register + go/no-go review. Mirrors the SageMaker Ops sub-agent pattern.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import READINESS_REVIEW_AGENT, build_subagent


def build_readiness_review_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "readiness_review_agent",
) -> Agent:
    """Build the readiness_review_agent sub-agent (D3.b).

    Runs the 10 readiness gates → risk register → go/no-go review. The Quality gate defers to the
    deterministic Arbiter (ADR-003); a failing gate yields a ``no_go`` recommendation that blocks
    ``deploy_flow``.
    """
    return build_subagent(
        READINESS_REVIEW_AGENT,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
