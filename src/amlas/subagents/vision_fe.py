"""Vision FE sub-agent (D1.c) — augmentation manifest + ROI strategy."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import VISION_FE, build_subagent


def build_vision_fe_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "vision_fe",
) -> Agent:
    return build_subagent(
        VISION_FE,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
