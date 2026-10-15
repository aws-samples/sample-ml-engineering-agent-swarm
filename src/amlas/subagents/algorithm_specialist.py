"""Algorithm Specialist sub-agent (D2.a) — selects one algo per iteration + config."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import ALGORITHM_SPECIALIST, build_subagent


def build_algorithm_specialist_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "algorithm_specialist",
) -> Agent:
    return build_subagent(
        ALGORITHM_SPECIALIST,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
