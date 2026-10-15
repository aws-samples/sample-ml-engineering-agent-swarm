"""ML Execution Lead (D2) — agents-as-tools over AlgorithmSpecialist + TrainerTuner + FineTune.

The composition lives in [`declarations.ML_EXECUTION_LEAD`](declarations.py:1); this
module is the signature-preserving builder over it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import ML_EXECUTION_LEAD, build_lead


def build_ml_execution_lead(
    *,
    model: Any | None = None,
    algo_specialist_model: Any | None = None,
    trainer_tuner_model: Any | None = None,
    fine_tune_model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "ml_execution_lead",
) -> Agent:
    """Build the ML Execution Lead (D2) agent.

    ``hooks_factory`` (when provided by the Orchestrator) is forwarded to
    ``build_agent`` for the Lead itself AND to every sub-agent
    constructor so the ``.as_tool(...)``-wrapped sub-agents also receive a
    ``MemoryHookProvider``. The default ``None`` preserves every existing
    unit-call site exactly.
    """
    return build_lead(
        ML_EXECUTION_LEAD,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
        algo_specialist_model=algo_specialist_model,
        trainer_tuner_model=trainer_tuner_model,
        fine_tune_model=fine_tune_model,
    )
