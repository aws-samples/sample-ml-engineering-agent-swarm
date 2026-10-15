"""labeling_strategist sub-agent (DE.b) — NEW in M6 (doc 06 §Subagents; doc 06 Stage 4; T-LBL-01/02).

The Stage-4 labeling executor over the ``LabelingBackend`` port: it creates a Ground-Truth
labeling job and polls its progress. Job CREATION is IRREVERSIBLE/spend and is approval-gated by
``data_collection_flow`` (the Lead drives the gated flow; this agent executes the tools). Bounded
executor; the Data Engineering Lead composes it with ``.as_tool(preserve_context=True)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import LABELING_STRATEGIST, build_subagent


def build_labeling_strategist(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "labeling_strategist",
) -> Agent:
    """Build the labeling_strategist sub-agent (DE.b) — Stage-4 labeling-job create + progress (gated)."""
    return build_subagent(
        LABELING_STRATEGIST,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
