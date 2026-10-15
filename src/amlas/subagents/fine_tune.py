"""Fine-Tune sub-agent (D2.c) — submits REAL async SageMaker training jobs.

De-mock Phase 5: re-pointed off the deleted ``tools/stubs`` fakes onto the REAL
``submit_sagemaker_job`` (real ``CreateTrainingJob`` + DynamoDB) and
``fetch_training_metrics`` (real ``describe_training_job`` metrics — MLflow dropped
per OPEN DECISION 6).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import FINE_TUNE, build_subagent


def build_fine_tune_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "fine_tune",
) -> Agent:
    return build_subagent(
        FINE_TUNE,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
