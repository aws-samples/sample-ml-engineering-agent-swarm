"""LLM Extrapolation sub-agent (D0.d) — bounded numeric/text gap fill.

System prompt stack: paranoid_scientist + data_steering (the dial) +
llm_extrapolation.md.
Tools: llm_extrapolate_values (Phase-7) + execute_python_tool (helper math only).

A *bounded* executor: it infers missing values from row context + domain notes,
but never decides acceptance — its output is always handed to
``evaluate_imputation`` by the Strategist. Master plan §3 (D0.d), spec sheet 3.4.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import LLM_EXTRAPOLATION, build_subagent


def build_llm_extrapolation_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "llm_extrapolation",
) -> Agent:
    """Build the LLM Extrapolation sub-agent (D0.d).

    Args:
        model: Optional pre-built model; None (default) uses real Bedrock. Callers may pass a real BedrockModel (real_bedrock fixture).
        hooks_factory: Optional zero-arg callable forwarded to ``build_agent``
            so the Orchestrator can attach a ``MemoryHookProvider``.
        name: Agent name — defaults to "llm_extrapolation".
    """
    return build_subagent(
        LLM_EXTRAPOLATION,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
