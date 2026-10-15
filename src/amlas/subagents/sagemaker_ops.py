"""SageMaker Ops sub-agent (M0.a) — bounded executor of the MLOps tools.

System prompt stack: paranoid_scientist + thought_guidance + sagemaker_ops.md.
Tools (Phase-10, ``tools/mlops/``): build/run pipeline, register/approve/list
model packages, setup/inspect Model Monitor, run Clarify, Feature Store
ingest/get. Mirrors the Trainer/Tuner bounded sub-agent
([`subagents/trainer_tuner.py`](trainer_tuner.py:14)).

Master plan §3 (M0.a), spec sheet 3.7.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import SAGEMAKER_OPS, build_subagent


def build_sagemaker_ops_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "sagemaker_ops",
) -> Agent:
    """Build the SageMaker Ops sub-agent (M0.a).

    Args:
        model: Optional pre-built model; None (default) uses real Bedrock. Callers may pass a real BedrockModel (real_bedrock fixture).
        hooks_factory: Optional zero-arg callable forwarded to ``build_agent``
            so the Orchestrator can attach a ``MemoryHookProvider``.
        name: Agent name — defaults to "sagemaker_ops".
    """
    return build_subagent(
        SAGEMAKER_OPS,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
