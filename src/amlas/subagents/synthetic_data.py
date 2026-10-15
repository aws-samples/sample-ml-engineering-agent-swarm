"""Synthetic Data sub-agent (D0.c) — requirement→spec→generate→validate.

System prompt stack: paranoid_scientist + data_steering (the dial) +
synthetic_data.md.
Tools (ADR-0002 — DISPATCH the generation, keep the decision): build_synthesis_spec (decision),
run_processing_job (→ SageMaker Processing runs SDV/SMOTE off-box), llm_synthesize_rows (LLM).

Acceptance is the deterministic 3-pillar AND-gate inside ``validate_synthetic`` —
the agent chooses a generator, the grader scores it. Master plan §7 / §3.1, spec 3.3.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import SYNTHETIC_DATA, build_subagent

# ADR-0002: fitting SDV generators (GaussianCopula/CTGAN/TVAE) + SMOTE augmentation over the full
# dataset is a data-plane WORKLOAD — dispatch it to SageMaker Processing via run_processing_job (a
# transform_spec carrying the SynthesisSpec), rather than training generators in the AgentCore
# sandbox. The agent keeps build_synthesis_spec (its decision: which generator + constraints) and
# llm_synthesize_rows (an LLM call, not local compute) — the heavy synthesize/augment/validate
# compute is dispatched.


def build_synthetic_data_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "synthetic_data",
) -> Agent:
    """Build the Synthetic Data sub-agent (D0.c).

    Args:
        model: Optional pre-built model; None (default) uses real Bedrock. Callers may pass a real BedrockModel (real_bedrock fixture).
        hooks_factory: Optional zero-arg callable forwarded to ``build_agent``
            so the Orchestrator can attach a ``MemoryHookProvider``.
        name: Agent name — defaults to "synthetic_data".
    """
    return build_subagent(
        SYNTHETIC_DATA,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
