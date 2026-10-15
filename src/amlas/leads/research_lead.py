"""Research Lead (R) — coordinates framing + feature_hypothesis + algo_hypothesis.

Uses Strands agents-as-tools: each sub is wrapped with `.as_tool(preserve_context=True)`
so downstream subs see the earlier sub's output in their context.

The composition itself (sub-agents, tool names, prompt stack, description) lives in
[`declarations.RESEARCH_LEAD`](declarations.py:1) so it is readable without building
an Agent; this module is the signature-preserving builder over it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import RESEARCH_LEAD, build_lead


def build_research_lead(
    *,
    model: Any | None = None,
    framing_model: Any | None = None,
    feature_model: Any | None = None,
    algo_model: Any | None = None,
    research_scout_model: Any | None = None,
    research_curator_model: Any | None = None,
    business_goal_model: Any | None = None,
    solution_fit_model: Any | None = None,
    problem_framing_model: Any | None = None,
    solution_survey_model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "research_lead",
) -> Agent:
    """Build the Research Lead agent.

    Separate model params for each sub let tests script deterministic outputs per sub.
    In production all sub `model=None` → they auto-build a real BedrockModel.

    The optional ``hooks_factory`` kwarg lets the Orchestrator inject a
    ``MemoryHookProvider`` scoped to ``(memory_id, actor_id=project_id,
    session_id=run_id)`` into every Agent built under this Lead — both the
    Lead itself and each sub-agent wrapped via ``.as_tool(...)``. It is
    forwarded to sub-agent builders only when non-None so existing callers
    that do not pass it (every offline validation) are unaffected even before task
    2.4 extends the sub-agent factory signatures.
    """
    return build_lead(
        RESEARCH_LEAD,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
        framing_model=framing_model,
        feature_model=feature_model,
        algo_model=algo_model,
        research_scout_model=research_scout_model,
        research_curator_model=research_curator_model,
        business_goal_model=business_goal_model,
        solution_fit_model=solution_fit_model,
        problem_framing_model=problem_framing_model,
        solution_survey_model=solution_survey_model,
    )
