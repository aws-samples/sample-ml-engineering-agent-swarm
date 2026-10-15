"""MLOps Lead (M0) — agents-as-tools over SageMaker Ops + the medallion lake.

Orchestrates the agentic MLOps loop the customer has "nobody" for: read a silver
slice → (hand off DQ/impute to the Data Engineering Lead — referenced, not
rebuilt) → train + register + monitor via the SageMaker Ops sub-agent → write the
cleaned/inferred dataset BACK to silver/gold with lineage → report MLOps KPIs.

Mirrors the Data Engineering Lead (D0) wiring EXACTLY — ``build_agent`` +
``.as_tool(preserve_context=True)`` + ``stack_prompts`` + the ``hooks_factory``
fan-out so the sub-agent also receives the ``MemoryHookProvider``. The steering
``data_steering`` preamble is stacked so the operator's dial (soft prefs + the
write-tier pin/forbid the tools enforce) reaches the Lead's reasoning.

Master plan §3 (M0), §8; spec sheet 3.6.

The composition (subs, the directly-bound lake/KPI + AWS control-plane tools, the
prompt stack) lives in [`declarations.MLOPS_LEAD`](declarations.py:1); this module is
the signature-preserving builder over it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import MLOPS_LEAD, build_lead


def build_mlops_lead(
    *,
    model: Any | None = None,
    sagemaker_ops_model: Any | None = None,
    devops_git_model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "mlops_lead",
) -> Agent:
    """Build the MLOps Lead (M0) agent.

    ``hooks_factory`` (when provided by the Orchestrator) is forwarded to
    ``build_agent`` for the Lead itself AND to the SageMaker Ops sub-agent
    constructor so the ``.as_tool(...)``-wrapped sub-agent also receives a
    ``MemoryHookProvider``. The default ``None`` preserves every existing
    unit-call site exactly (mirrors the Data Engineering / FE Leads).

    Args:
        model: Lead model; None (default) → real Bedrock. Callers may inject a real BedrockModel.
        sagemaker_ops_model: Optional sub-agent model override.
        devops_git_model: Optional sub-agent model override.
        hooks_factory: Optional zero-arg callable producing the per-agent hooks list.
        name: Agent name — defaults to "mlops_lead".
    """
    return build_lead(
        MLOPS_LEAD,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
        sagemaker_ops_model=sagemaker_ops_model,
        devops_git_model=devops_git_model,
    )
