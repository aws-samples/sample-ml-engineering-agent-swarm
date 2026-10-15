"""solution_survey_agent sub-agent (R3.a) — NEW in M6 (doc 06 §Subagents; doc 06 Stage 3; T-HUB-01).

The Stage-3 solution-survey executor over the ``ModelHubBackend`` port: it lists JumpStart /
Bedrock FM model cards, gates each candidate against the reproducibility evidence, and builds the
ranked candidate shortlist the prototype stage evaluates. Bounded executor (no planning); the
Research Lead composes it with ``.as_tool(preserve_context=True)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import SOLUTION_SURVEY_AGENT, build_subagent


def build_solution_survey_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "solution_survey_agent",
) -> Agent:
    """Build the solution_survey_agent sub-agent (R3.a) — Stage-3 ModelHub list + research-gate + shortlist."""
    return build_subagent(
        SOLUTION_SURVEY_AGENT,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
