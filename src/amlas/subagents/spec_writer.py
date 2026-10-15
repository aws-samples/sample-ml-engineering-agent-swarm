"""Spec Writer sub-agent — generates validated SpecModel from free-text input.

System prompt stack: paranoid_scientist + spec_writer.
Tools: validate_spec, save_spec, list_example_specs.

The sub-agent takes a user's natural-language description of a business challenge
and produces a valid task_spec.md (YAML front-matter + markdown domain notes).
It validates its output against SpecModel before returning.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import SPEC_WRITER, build_subagent


def build_spec_writer_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "spec_writer",
) -> Agent:
    """Build the Spec Writer sub-agent.

    Args:
        model: Optional pre-built model; None (default) uses real Bedrock. Callers may pass a real BedrockModel (real_bedrock fixture).
        hooks_factory: Optional zero-arg callable forwarded to ``build_agent``
            so the Orchestrator can attach a ``MemoryHookProvider`` scoped to
            ``(memory_id, actor_id=project_id, session_id=run_id)``.
        name: Agent name — defaults to "spec_writer" (used in traces + a2a).
    """
    return build_subagent(
        SPEC_WRITER,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
