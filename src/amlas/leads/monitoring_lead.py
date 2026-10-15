"""Monitoring Lead (M1) — NEW in M5 (doc 06 §Leads; ADR-008/003/014).

The 7th Lead, owning the Stage-11 → Stage-12 lifecycle CLOSE: production monitoring → drift/failure
detection → the closed-loop retrain trigger that RE-ENTERS the unified exploration retreat loop
(ADR-008). Composes two NEW sub-agents as tools — ``monitoring_agent`` (drift detection + metrics +
alerts + report + the retrain-trigger decision) and ``improvement_agent`` (failure triage +
AgentCore-Memory carry-over + retire) — and owns the ``MonitoringBackend`` port. It drives
``monitor_flow`` then, on a TRIGGER, ``improve_retrain_flow`` (plan_approve_execute-gated).

Mirrors the Deploy/Serving Lead (D3) wiring EXACTLY — ``build_agent`` +
``.as_tool(preserve_context=True)`` + ``stack_prompts`` + the ``hooks_factory`` fan-out so each
sub-agent also receives the ``MemoryHookProvider``. This completes the 7-Lead structure and closes
the lifecycle loop (the linear pipeline becomes a closed monitoring→retreat cycle).

The composition lives in [`declarations.MONITORING_LEAD`](declarations.py:1); this module is the
signature-preserving builder over it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import MONITORING_LEAD, build_lead

#: Repo-wide convention: the prompt files stacked into this agent's system prompt.
#: Sourced from the declaration so the two can never disagree.
SYSTEM_PROMPT_SOURCES = MONITORING_LEAD.prompt_sources


def build_monitoring_lead(
    *,
    model: Any | None = None,
    monitoring_model: Any | None = None,
    improvement_model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "monitoring_lead",
) -> Agent:
    """Build the Monitoring Lead (M1) agent.

    ``hooks_factory`` (when provided by the Orchestrator) is forwarded to ``build_agent``
    for the Lead itself AND to every sub-agent constructor so the ``.as_tool(...)``-wrapped
    sub-agents also receive a ``MemoryHookProvider``. The default ``None`` preserves the existing
    import-time behavior.

    Args:
        model: Lead model; None (default) → real Bedrock. Callers may inject a real BedrockModel.
        monitoring_model / improvement_model: Optional sub-agent model overrides.
        hooks_factory: Optional zero-arg memory-hook factory fanned out to subs.
        name: Agent name — defaults to "monitoring_lead".
    """
    return build_lead(
        MONITORING_LEAD,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
        monitoring_model=monitoring_model,
        improvement_model=improvement_model,
    )
