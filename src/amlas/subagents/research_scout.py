"""research_scout sub-agent (R0.a) — NEW in M6 (doc 06 §Subagents; ADR-017).

The Stage-0 research-discovery executor over the ``RetrievalBackend`` port (+ the live research
aggregator). It indexes the research corpus, scans external + indexed sources for a query, and
retrieves the top-k relevant chunks — the raw findings the research_curator then triages. Bounded
executor (no planning); the Research Lead composes it with ``.as_tool(preserve_context=True)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import RESEARCH_SCOUT, build_subagent


def build_research_scout(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "research_scout",
) -> Agent:
    """Build the research_scout sub-agent (R0.a) — Stage-0 scan + RAG retrieval over RetrievalBackend."""
    return build_subagent(
        RESEARCH_SCOUT,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
