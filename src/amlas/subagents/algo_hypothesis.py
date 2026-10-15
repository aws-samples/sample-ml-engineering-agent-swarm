"""Algorithm Hypothesis sub-agent (R3) — shortlists 3-5 algorithms + search spaces."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import ALGO_HYPOTHESIS, build_subagent


def build_algo_hypothesis_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "algo_hypothesis",
) -> Agent:
    return build_subagent(
        ALGO_HYPOTHESIS,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
