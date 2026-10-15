"""problem_framing_advisor sub-agent (R2.a) — NEW in M6 (doc 06 §Subagents; doc 06 Stage 2).

The Stage-2 framing specialist: it selects the canonical ML problem type, defines the success
metric(s) + guardrails, assembles the ``problem_card.json``, and checks the stage-2 exit gate.
Pure-reasoning — turns the Stage-1 business goal into a framed problem card the downstream
Research/FE/ML Leads consume. The Research Lead composes it with ``.as_tool(preserve_context=True)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import PROBLEM_FRAMING_ADVISOR, build_subagent


def build_problem_framing_advisor(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "problem_framing_advisor",
    messages: Sequence[Any] | None = None,
    steering: Any | None = None,
    problem: Any | None = None,
) -> Agent:
    """Build the problem_framing_advisor sub-agent (R2.a) — Stage-2 problem-type/metric/card reasoning.

    H2-6: as an intent-reasoning advisor it accepts the same additive session context as
    the framing sub-agent — prior conversation ``messages`` (threaded into
    ``build_agent(messages=…)``) and a ``steering``/``problem`` preamble composed into the
    system prompt (:func:`compose_system_prompt`). All three default to ``None`` → the
    build is byte-identical to today.
    """
    return build_subagent(
        PROBLEM_FRAMING_ADVISOR,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
        messages=messages,
        steering=steering,
        problem=problem,
    )
