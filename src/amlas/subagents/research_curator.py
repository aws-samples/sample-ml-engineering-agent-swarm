"""research_curator sub-agent (R0.b) — NEW in M6 (doc 06 §Subagents; ADR-017).

The Stage-0 curation executor: it triages the research_scout's findings by reproducibility,
compiles a structured digest (persisted to ``memory-bank/research/``), and proposes a recurring
scan cadence. Pure-reasoning over the findings (no managed backend). Bounded executor (no
planning); the Research Lead composes it with ``.as_tool(preserve_context=True)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import RESEARCH_CURATOR, build_subagent


def build_research_curator(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "research_curator",
) -> Agent:
    """Build the research_curator sub-agent (R0.b) — Stage-0 triage + digest + scan cadence."""
    return build_subagent(
        RESEARCH_CURATOR,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
