"""FE Lead (D1) — agents-as-tools over DataIntegrity + TabularFE + VisionFE.

The composition lives in [`declarations.FE_LEAD`](declarations.py:1); this module is
the signature-preserving builder over it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import FE_LEAD, build_lead


def build_fe_lead(
    *,
    model: Any | None = None,
    data_integrity_model: Any | None = None,
    tabular_fe_model: Any | None = None,
    vision_fe_model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "fe_lead",
) -> Agent:
    """Build the FE Lead (D1) agent.

    ``hooks_factory`` (when provided by the Orchestrator) is forwarded to
    ``build_agent`` for the Lead itself AND to every sub-agent
    constructor so the ``.as_tool(...)``-wrapped sub-agents also receive a
    ``MemoryHookProvider``. The default ``None`` preserves every existing
    unit-call site exactly.
    """
    return build_lead(
        FE_LEAD,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
        data_integrity_model=data_integrity_model,
        tabular_fe_model=tabular_fe_model,
        vision_fe_model=vision_fe_model,
    )
