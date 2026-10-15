"""solution_fit_advisor sub-agent (R1.b) — NEW in M6 (doc 06 §Subagents; doc 06 Stage 1/3).

The "is-ML-the-right-fit" specialist: it assesses whether ML is genuinely warranted (vs a
deterministic/rules alternative) and surveys the candidate AWS managed-service options for the
problem/modality. Pure-reasoning. The Research Lead composes it with ``.as_tool(preserve_context=True)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import SOLUTION_FIT_ADVISOR, build_subagent


def build_solution_fit_advisor(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "solution_fit_advisor",
    messages: Sequence[Any] | None = None,
    steering: Any | None = None,
    problem: Any | None = None,
) -> Agent:
    """Build the solution_fit_advisor sub-agent (R1.b) — ML-necessity + managed-service fit reasoning.

    H2-6: intent-reasoning advisor — accepts the same additive session context as the
    framing sub-agent (prior ``messages`` + a ``steering``/``problem`` preamble). All
    default to ``None`` → byte-identical to today.
    """
    return build_subagent(
        SOLUTION_FIT_ADVISOR,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
        messages=messages,
        steering=steering,
        problem=problem,
    )
