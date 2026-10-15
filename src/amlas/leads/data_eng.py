"""Data Engineering Lead (D0) — agents-as-tools over DQ Assessment + Imputation
Strategist + Synthetic Data + LLM Extrapolation.

Orchestrates the data-quality lifecycle assess → impute/synthesize → re-validate,
threading the domain-steering ``SteeringProfile`` down to every sub-agent. Mirrors
the FE Lead (D1) wiring including the ``hooks_factory`` fan-out to sub-agents.
Master plan §3 (D0), spec sheet 3.5.

The composition (subs, the directly-bound DQ + AWS-landing control-plane tools, the
prompt stack) lives in [`declarations.DATA_ENGINEERING_LEAD`](declarations.py:1); this
module is the signature-preserving builder over it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import DATA_ENGINEERING_LEAD, build_lead


def build_data_engineering_lead(
    *,
    model: Any | None = None,
    dq_assessment_model: Any | None = None,
    imputation_model: Any | None = None,
    synthetic_model: Any | None = None,
    llm_extrapolation_model: Any | None = None,
    data_collection_model: Any | None = None,
    labeling_model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "data_engineering_lead",
) -> Agent:
    """Build the Data Engineering Lead (D0) agent.

    ``hooks_factory`` (when provided by the Orchestrator) is forwarded to
    ``build_agent`` for the Lead itself AND to every sub-agent constructor
    so the ``.as_tool(...)``-wrapped sub-agents also receive a
    ``MemoryHookProvider``. The default ``None`` preserves every existing
    unit-call site exactly (mirrors the FE / ML Execution Leads).

    Args:
        model: Lead model; None (default) → real Bedrock. Callers may inject a real BedrockModel.
        dq_assessment_model: Optional sub-agent model override.
        imputation_model: Optional sub-agent model override.
        synthetic_model: Optional sub-agent model override.
        llm_extrapolation_model: Optional sub-agent model override.
        data_collection_model: Optional sub-agent model override.
        labeling_model: Optional sub-agent model override.
        hooks_factory: Optional zero-arg callable producing the per-agent hooks list.
        name: Agent name — defaults to "data_engineering_lead".
    """
    return build_lead(
        DATA_ENGINEERING_LEAD,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
        dq_assessment_model=dq_assessment_model,
        imputation_model=imputation_model,
        synthetic_model=synthetic_model,
        llm_extrapolation_model=llm_extrapolation_model,
        data_collection_model=data_collection_model,
        labeling_model=labeling_model,
    )
