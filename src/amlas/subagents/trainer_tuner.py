"""Trainer/Tuner sub-agent (D2.b) — Optuna search within compute budget."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import TRAINER_TUNER, build_subagent


def build_trainer_tuner_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "trainer_tuner",
) -> Agent:
    # ADR-0002: the trainer ORCHESTRATES training — it DISPATCHES the deliverable fit to SageMaker
    # (submit_tabular_training → the tabular_train entrypoint writes the graded y_pred to S3) and
    # awaits it (fetch_training_metrics), rather than fitting in the AgentCore sandbox. It KEEPS
    # execute_python_tool for LIGHT decision-support ONLY (quick data inspection / sanity math to
    # choose algo+params) — never to run the training workload; the prompt scopes this.
    return build_subagent(
        TRAINER_TUNER,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
