"""``devops_git_flow`` — commit ML artifacts then open a PR/MR (roadmap M7; US-CC.08).

The cross-cutting flow the NEW ``devops_git_agent`` owns (doc 06 §1.3): the
conversational "open a PR for this spec / pipeline / model-card" sequence. It composes
the two backend-neutral VCS tools (``tools/git_neutral``):

    **commit_artifact (T-GIT-02) → open_pr (T-GIT-01)**

commit_artifact stages + commits the named artifact(s) locally (BOUNDED_COMPUTE,
auto-approved under cap); open_pr pushes the branch + opens the GitLab MR (IRREVERSIBLE,
gated). Because the flow's terminal step PUSHES, the flow as a whole is spend/irreversible
in its registry descriptor → the M7 capability funnel routes ``devops_git_flow`` through
``plan_approve_execute`` so a denial ABORTS before any push/MR (the approval-gating the
M7 tests assert).

``dry_run`` threads through to BOTH tools so validation exercises the exact commit + push/MR
command construction with zero VCS mutation or network — the M7 behavioral contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..tools.git_neutral import commit_artifact, open_pr

__all__ = ["DevopsGitFlowResult", "run_devops_git_flow"]


@dataclass
class DevopsGitFlowResult:
    """The output contract of one ``devops_git_flow`` run (commit → open_pr)."""

    status: str  # "opened" | "committed" | "dry_run" | "refused" | "error"
    commit: dict[str, Any] = field(default_factory=dict)
    pr: dict[str, Any] = field(default_factory=dict)
    pr_url: str = ""
    branch: str = ""
    artifacts: list[str] = field(default_factory=list)
    dry_run: bool = True
    error: str | None = None


def run_devops_git_flow(
    *,
    paths: list[str] | str,
    branch: str,
    target_branch: str = "main",
    summary: str = "",
    title: str = "",
    run_ref: str = "",
    repo_dir: str = ".",
    force: bool = False,
    dry_run: bool = True,
) -> DevopsGitFlowResult:
    """Run the commit→PR sequence for named ML artifacts (US-CC.08; devops_git_agent).

    Steps:
      1. **commit_artifact** (T-GIT-02) — stage + commit ``paths`` with a structured
         message linking ``run_ref``;
      2. **open_pr** (T-GIT-01) — push ``branch`` + open the GitLab MR with an
         auto-written title/description/checklist, returning the MR URL.

    The flow is registered spend-incurring + irreversible, so the M7 capability funnel
    wraps it with ``plan_approve_execute`` — a denial aborts BEFORE the push step. Within
    a dry run, both tools only CONSTRUCT their commands (no VCS mutation / no network).

    Args:
        paths: Artifact path(s) to commit.
        branch: The feature branch to push + open the MR from.
        target_branch: MR target (default ``main``).
        summary: Seeds the commit subject + the MR body.
        title: Optional explicit MR title.
        run_ref: Originating run/spec id linked into the commit + MR.
        repo_dir: The git working tree.
        force: Request a force-push (refused on a protected branch by ``open_pr``).
        dry_run: Thread the safe/test no-mutation path through both tools.

    Returns:
        A :class:`DevopsGitFlowResult` (commit + pr sub-results + the MR url/branch).
    """
    artifacts = [paths] if isinstance(paths, str) else list(paths or [])

    commit = commit_artifact(
        paths=artifacts, summary=summary, run_ref=run_ref, repo_dir=repo_dir, dry_run=dry_run
    )
    if commit.get("status") == "error":
        return DevopsGitFlowResult(status="error", commit=commit, artifacts=artifacts,
                                   dry_run=dry_run, error=commit.get("error"))

    pr = open_pr(
        branch=branch, target_branch=target_branch, title=title, summary=summary,
        artifacts=artifacts, run_ref=run_ref, repo_dir=repo_dir, force=force, dry_run=dry_run,
    )
    pr_status = pr.get("status", "error")
    # Map the PR sub-status to the flow status (refused/error propagate; else opened/dry_run).
    if pr_status == "refused":
        status = "refused"
    elif pr_status == "error":
        status = "error"
    elif pr_status == "dry_run":
        status = "dry_run"
    else:
        status = "opened"

    return DevopsGitFlowResult(
        status=status,
        commit=commit,
        pr=pr,
        pr_url=str(pr.get("pr_url", "")),
        branch=branch,
        artifacts=artifacts,
        dry_run=dry_run,
        error=pr.get("error"),
    )
