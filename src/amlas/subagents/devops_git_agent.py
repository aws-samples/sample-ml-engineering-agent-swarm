"""devops_git_agent sub-agent — NEW in M7 (doc 06 §1.3; US-CC.08; ADR-010).

The one Claude-Code-native capability that warrants its OWN subagent rather than a
control-plane facet (doc 06 §1.3): conversational git/PR authoring for ML artifacts.
It owns **T-GIT-02 ``commit_artifact``** + **T-GIT-01 ``open_pr``** — commit a named
spec / pipeline / model-card and open a GitLab MR with an auto-written title/description/
test-checklist, linking it back to the originating run. ORCHESTRATE-only (git/glab via
``tools/git_neutral`` — no formal AWS port).

Composed from [`declarations.DEVOPS_GIT_AGENT`](declarations.py:1) like every other sub-agent;
the MLOps Lead wraps it with ``.as_tool(preserve_context=True)``.

The push/MR is approval-gated (``open_pr`` classifies ``IRREVERSIBLE`` and the
``devops_git_flow`` it composes routes through ``plan_approve_execute``); it **refuses
force-push to protected branches** by default (the ``git_neutral`` safety rail). The
``@tool`` wrappers default ``dry_run=True`` so an un-approved call never mutates VCS — they
now live in [`tools/git_tools.py`](../tools/git_tools.py:1), because a ``@tool`` referenced by a
declaration cannot sit in the sub-agent module that imports that declaration.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from strands import Agent

from .declarations import DEVOPS_GIT_AGENT, build_subagent


def build_devops_git_agent(
    *,
    model: Any | None = None,
    hooks_factory: Callable[[], Sequence[Any]] | None = None,
    name: str = "devops_git_agent",
) -> Agent:
    """Build the devops_git_agent sub-agent (NEW, cross-cutting; US-CC.08).

    Args:
        model: Optional pre-built model; None (default) uses real Bedrock. Callers may pass a real BedrockModel.
        hooks_factory: Optional zero-arg callable forwarded to ``build_agent`` so the
            Orchestrator can attach a ``MemoryHookProvider`` (+ the approval-gate hook).
        name: Agent name — defaults to "devops_git_agent".
    """
    return build_subagent(
        DEVOPS_GIT_AGENT,
        model=model,
        name=name,
        hooks_factory=hooks_factory,
    )
