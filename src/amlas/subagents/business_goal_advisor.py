"""business_goal_advisor sub-agent (R1.a) — NEW in M6 (doc 06 §Subagents; doc 06 Stage 1).

The Stage-1 business-goal specialist: it captures the business problem into a structured record,
defines a measurable KPI, and renders the business go/no-go gate. Pure-reasoning (doc 06 §2.7 —
no managed backend). The Research Lead composes it with ``.as_tool(preserve_context=True)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import BUSINESS_GOAL_ADVISOR, build_subagent


def build_business_goal_advisor(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "business_goal_advisor",
    messages: Sequence[Any] | None = None,
    steering: Any | None = None,
    problem: Any | None = None,
) -> Agent:
    """Build the business_goal_advisor sub-agent (R1.a) — Stage-1 problem/KPI/go-no-go reasoning.

    H2-6: intent-reasoning advisor — accepts the same additive session context as the
    framing sub-agent (prior ``messages`` + a ``steering``/``problem`` preamble). All
    default to ``None`` → byte-identical to today.
    """
    return build_subagent(
        BUSINESS_GOAL_ADVISOR,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
        messages=messages,
        steering=steering,
        problem=problem,
    )
