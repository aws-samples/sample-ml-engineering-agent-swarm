"""Tabular FE sub-agent (D1.b) — CAAFE validator-loop on features."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import TABULAR_FE, build_subagent

# ADR-0002: materializing engineered features over the full dataset is a data-plane WORKLOAD —
# dispatch it to SageMaker Processing via run_processing_job (a transform_spec of the feature
# derivations), rather than running pandas/sklearn FE in the AgentCore sandbox. The agent keeps
# save_feature_manifest (its decision artifact) and execute_python_tool for LIGHT feature
# HYPOTHESIS checks (a quick MI/correlation on a tiny sample to decide which features to keep) —
# never the full-dataset transform, which is dispatched.


def build_tabular_fe_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "tabular_fe",
) -> Agent:
    return build_subagent(
        TABULAR_FE,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
