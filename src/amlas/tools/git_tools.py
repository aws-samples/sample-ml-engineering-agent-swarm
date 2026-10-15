"""Agent-facing ``@tool`` wrappers over the neutral git operations (T-GIT-01 / T-GIT-02).

[`git_neutral`](git_neutral.py:1) holds the plain implementations; these are the
``@tool``-decorated, agent-facing signatures with the docstrings the model actually reads when
choosing a tool.

They lived inside ``subagents/devops_git_agent.py`` until the sub-agent declaration seam landed
(ADR-0007 phase 1). That placement made the tools unreachable from
[`subagents/declarations.py`](../subagents/declarations.py:1) without a circular import — the
sub-agent module has to import its declaration, and the declaration has to reference the tool
objects. A ``@tool`` belongs in the tools layer regardless, so they moved here rather than the
seam being bent around them.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from .git_neutral import commit_artifact as _commit_artifact
from .git_neutral import open_pr as _open_pr


@tool
def commit_artifact(
    paths: list[str] | str,
    summary: str = "",
    run_ref: str = "",
    repo_dir: str = ".",
    dry_run: bool = True,
) -> dict[str, Any]:
    """Stage + commit named ML artifact(s) with a structured message (T-GIT-02; US-CC.08).

    A local commit (``BOUNDED_COMPUTE`` — bounded, recoverable). ``dry_run`` (default True)
    constructs the git commands without mutating the working tree; set ``dry_run=False``
    only after the approval beat. Returns ``{status, artifacts, message, sha, commands, ...}``.
    """
    return _commit_artifact(
        paths=paths, summary=summary, run_ref=run_ref, repo_dir=repo_dir, dry_run=dry_run
    )


@tool
def open_pr(
    branch: str,
    target_branch: str = "main",
    title: str = "",
    summary: str = "",
    artifacts: list[str] | None = None,
    run_ref: str = "",
    repo_dir: str = ".",
    force: bool = False,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Push the branch + open a GitLab MR with an auto-written description (T-GIT-01; US-CC.08).

    The push/MR is a REAL external side-effect (``IRREVERSIBLE`` → approval-gated upstream);
    refuses force-push to a protected branch. ``dry_run`` (default True) constructs the
    push/MR commands without any network. Returns ``{status, branch, pr_url, body, commands, ...}``.
    """
    return _open_pr(
        branch=branch,
        target_branch=target_branch,
        title=title,
        summary=summary,
        artifacts=artifacts,
        run_ref=run_ref,
        repo_dir=repo_dir,
        force=force,
        dry_run=dry_run,
    )


__all__ = ["commit_artifact", "open_pr"]
