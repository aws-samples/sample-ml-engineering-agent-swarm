"""Feature Hypothesis sub-agent (R2) — proposes feature set + justifications."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import FEATURE_HYPOTHESIS, build_subagent


def build_feature_hypothesis_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "feature_hypothesis",
) -> Agent:
    return build_subagent(
        FEATURE_HYPOTHESIS,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
