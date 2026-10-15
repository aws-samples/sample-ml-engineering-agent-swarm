"""data_collection_agent sub-agent (DE.a) — NEW in M6 (doc 06 §Subagents; doc 06 Stage 4; T-QRY-01).

The Stage-4 data-inventory executor over the ``QueryBackend`` port: it runs NL→SQL over the lake
to inventory what data is available + how much is labeled — the read that informs the labeling
strategy. Bounded executor (CHEAP_READ; no job creation here — labeling is the labeling_strategist's
gated action). The Data Engineering Lead composes it with ``.as_tool(preserve_context=True)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import DATA_COLLECTION_AGENT, build_subagent


def build_data_collection_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "data_collection_agent",
) -> Agent:
    """Build the data_collection_agent sub-agent (DE.a) — Stage-4 lake inventory via QueryBackend (read)."""
    return build_subagent(
        DATA_COLLECTION_AGENT,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
