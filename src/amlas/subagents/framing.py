"""Framing sub-agent (R1) — distills spec into problem_card.json."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import FRAMING, build_subagent


def build_framing_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "framing",
    messages: Sequence[Any] | None = None,
    steering: Any | None = None,
    problem: Any | None = None,
) -> Agent:
    """Build the framing sub-agent (R1) — distills a task spec into a problem_card.json.

    H2-6 (Hop-2 subtask H2-C): the framing sub-agent reasons about the USER'S intent, so
    when a flow/Lead delegates to it for interactive refinement it should carry the session
    context the user already established. The optional ``messages`` (prior conversation
    turns), ``steering`` (the session dial), and ``problem`` (the active problem framing)
    are threaded in ADDITIVELY:

      * ``messages`` → :func:`build_agent(messages=…)` (the canonical Strands multi-turn
        seam) so the sub-agent sees the conversation instead of starting blind;
      * ``steering`` / ``problem`` → a preamble composed into the ``system_prompt`` via
        :func:`compose_system_prompt` so the delegated reasoning respects the session's
        HARD constraints + framing.

    Passing NONE for all three (every existing swarm call site) yields a BYTE-IDENTICAL
    build to before — single-turn, static prompt — so the default path is unchanged.
    """
    return build_subagent(
        FRAMING,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
        messages=messages,
        steering=steering,
        problem=problem,
    )
