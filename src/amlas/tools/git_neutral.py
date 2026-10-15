"""Conversational git/PR authoring — T-GIT-02 ``commit_artifact`` + T-GIT-01 ``open_pr``
(roadmap M7; doc 06 §1.3 ``devops_git_agent``; US-CC.08).

The Claude-Code "conversational git/PR" analog for ML artifacts. Two backend-NEUTRAL,
ORCHESTRATE-only tools (no formal AWS port — they shell out to git / GitLab ``glab`` /
the GitLab API, per doc 06 Group-6 reconciliation note):

  * **``commit_artifact`` (T-GIT-02)** — stage + commit a named ML artifact (spec /
    pipeline yaml / model-card / flow output) with a structured commit message. A LOCAL
    commit is a lighter side-effect (it touches the working repo, not a remote), so it is
    classified ``BOUNDED_COMPUTE`` — auto-approved under the bounded cap.

  * **``open_pr`` (T-GIT-01)** — push the branch + open a GitLab merge request (the
    auto-written title/description/test-checklist), returning the MR URL and linking it
    back to the originating run/spec. Push + MR-open are a REAL external side-effect →
    classified ``IRREVERSIBLE`` so ``request_approval`` GATES them (and the
    ``approval_gate_hook`` denies fail-closed). It **refuses force-push to protected
    branches** by default (the doc 06 spec safety rail).

Both run through ``subprocess`` against the local repo's ``git`` (+ ``glab`` when present)
but accept ``dry_run=True`` so TESTS exercise the exact command construction WITHOUT any
real VCS mutation or network — the M7 behavioral contract. Real execution is opt-in
(``dry_run=False``) and gated by the approval beat the ``devops_git_agent`` honors.

No boto3, no AWS — VCS orchestration only.
"""

from __future__ import annotations

import shutil
import subprocess  # nosec B404
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "PROTECTED_BRANCHES",
    "GitResult",
    "PRResult",
    "commit_artifact",
    "open_pr",
]

# Branches we refuse to force-push to by default (doc 06 §1.3 safety rail).
PROTECTED_BRANCHES = ("main", "master", "release", "production", "prod")

# R-B-007 / AMLAS-SEC-B-007 (defense-in-depth, CWE-78 / bandit B603): the tool only ever
# constructs git/glab argv from a fixed set of subcommands (it never interpolates untrusted
# free-text into the executable or the subcommand). This allow-list makes that contract
# explicit + enforced: ``_run_git`` refuses to spawn anything whose (executable, subcommand)
# pair is not registered here, so a future edit cannot accidentally smuggle an unexpected
# subcommand through the same code path. ``shell=False`` (list argv) already prevents shell
# metacharacter injection; this guards the command surface itself.
_ALLOWED_COMMANDS: dict[str, frozenset[str]] = {
    "git": frozenset({"add", "commit", "rev-parse", "push"}),
    "glab": frozenset({"mr"}),
}

# P3 audit H-07 hardening: pin the glab ``mr`` surface to the single action this tool
# authors (``glab mr create``). Any other third token (``merge``, ``approve``, …) is refused.
_ALLOWED_GLAB_MR_ACTIONS: frozenset[str] = frozenset({"create"})


class DisallowedGitCommandError(ValueError):
    """Raised when a subprocess argv is not on the git/glab allow-list (R-B-007)."""


@dataclass
class GitResult:
    """Result of a ``commit_artifact`` call (T-GIT-02)."""

    status: str  # "committed" | "dry_run" | "error" | "noop"
    artifacts: list[str] = field(default_factory=list)
    message: str = ""
    sha: str = ""
    commands: list[list[str]] = field(default_factory=list)
    dry_run: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status, "artifacts": self.artifacts, "message": self.message,
            "sha": self.sha, "commands": [list(c) for c in self.commands],
            "dry_run": self.dry_run, "error": self.error,
        }


@dataclass
class PRResult:
    """Result of an ``open_pr`` call (T-GIT-01)."""

    status: str  # "opened" | "dry_run" | "refused" | "error"
    branch: str = ""
    target_branch: str = "main"
    title: str = ""
    body: str = ""
    pr_url: str = ""
    artifacts: list[str] = field(default_factory=list)
    commands: list[list[str]] = field(default_factory=list)
    dry_run: bool = False
    linked_run: str = ""
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status, "branch": self.branch, "target_branch": self.target_branch,
            "title": self.title, "body": self.body, "pr_url": self.pr_url,
            "artifacts": self.artifacts, "commands": [list(c) for c in self.commands],
            "dry_run": self.dry_run, "linked_run": self.linked_run, "error": self.error,
        }


def _run_git(args: list[str], *, cwd: str | Path, dry_run: bool) -> tuple[int, str]:
    """Run a ``git``/CLI command (or, when ``dry_run``, record it without executing).

    Returns ``(returncode, stdout_or_dryrun_marker)``. A dry run NEVER spawns a process —
    so tests assert the exact argv the tool would run with zero side-effects.
    """
    # R-B-007: enforce the (executable, subcommand) allow-list BEFORE spawning. ``args`` is a
    # tool-built list, but this guard makes the command surface explicit so a crafted/unexpected
    # subcommand cannot ride the same code path (defense-in-depth; raised before any dry-run too).
    _assert_allowed_command(args)
    if dry_run:
        return 0, "<dry-run>"
    try:
        # P3 audit H-07: list argv + shell=False (default) + the R-B-007 allow-list above —
        # caller-controlled values occupy single argv slots where shell metacharacters are inert.
        proc = subprocess.run(  # nosemgrep: dangerous-subprocess-use-audit  # nosec B603
            args, cwd=str(cwd), capture_output=True, text=True, timeout=60, check=False
        )
        return proc.returncode, (proc.stdout or proc.stderr or "").strip()
    except (OSError, subprocess.SubprocessError) as e:
        return 1, str(e)


def _assert_allowed_command(args: list[str]) -> None:
    """Refuse any argv whose (executable, subcommand) pair is not on ``_ALLOWED_COMMANDS`` (R-B-007).

    Raises :class:`DisallowedGitCommandError` for an empty argv, an unknown executable, or a
    subcommand outside the executable's allow-list — so the git tool can only ever run the
    fixed set of git/glab subcommands it was designed to author.
    """
    if not args:
        raise DisallowedGitCommandError("refusing to run an empty command (R-B-007)")
    exe = args[0]
    allowed = _ALLOWED_COMMANDS.get(exe)
    if allowed is None:
        raise DisallowedGitCommandError(
            f"refusing to run disallowed executable {exe!r}; allowed: {sorted(_ALLOWED_COMMANDS)} (R-B-007)"
        )
    subcommand = args[1] if len(args) > 1 else ""
    if subcommand not in allowed:
        raise DisallowedGitCommandError(
            f"refusing {exe!r} subcommand {subcommand!r}; allowed for {exe!r}: {sorted(allowed)} (R-B-007)"
        )
    # P3 audit H-07 hardening: ``glab mr`` is pinned to the ``create`` action this tool authors.
    if exe == "glab" and subcommand == "mr":
        action = args[2] if len(args) > 2 else ""
        if action not in _ALLOWED_GLAB_MR_ACTIONS:
            raise DisallowedGitCommandError(
                f"refusing 'glab mr' action {action!r}; allowed: {sorted(_ALLOWED_GLAB_MR_ACTIONS)} (R-B-007)"
            )


def _structured_commit_message(*, summary: str, artifacts: list[str], run_ref: str) -> str:
    """Compose a structured commit message (subject + artifact list + run linkage)."""
    subject = summary.strip() or f"chore(ml): commit {len(artifacts)} artifact(s)"
    lines = [subject, ""]
    if artifacts:
        lines.append("Artifacts:")
        lines.extend(f"  - {a}" for a in artifacts)
        lines.append("")
    if run_ref:
        lines.append(f"Linked-Run: {run_ref}")
    return "\n".join(lines).rstrip() + "\n"


def commit_artifact(
    *,
    paths: list[str] | str,
    summary: str = "",
    run_ref: str = "",
    repo_dir: str | Path = ".",
    dry_run: bool = True,
) -> dict[str, Any]:
    """``T-GIT-02`` — stage + commit named ML artifact(s) with a structured message (US-CC.08).

    Stages the given artifact path(s) and commits them with an auto-composed structured
    message (subject + artifact list + ``Linked-Run`` trailer). A local commit is a
    bounded, recoverable side-effect — ``classify_action("commit_artifact")`` →
    ``BOUNDED_COMPUTE`` (auto-approved under cap). The PUSH that publishes it is the gated
    step (see :func:`open_pr`).

    Args:
        paths: One path or a list of artifact paths (relative to ``repo_dir``) to stage.
        summary: The commit subject (a default is composed when empty).
        run_ref: Optional originating run/spec id recorded as a ``Linked-Run`` trailer.
        repo_dir: The git working tree.
        dry_run: When True (default — the test/safe path), construct + RECORD the git
            commands without executing them (no working-tree mutation).

    Returns:
        :class:`GitResult` as a dict (``status/artifacts/message/sha/commands/dry_run/error``).
    """
    artifacts = [paths] if isinstance(paths, str) else list(paths or [])
    if not artifacts:
        return GitResult(status="noop", error="no artifact paths supplied", dry_run=dry_run).to_dict()

    message = _structured_commit_message(summary=summary, artifacts=artifacts, run_ref=run_ref)
    add_cmd = ["git", "add", "--", *artifacts]
    commit_cmd = ["git", "commit", "-m", message]
    commands = [add_cmd, commit_cmd]

    if dry_run:
        return GitResult(status="dry_run", artifacts=artifacts, message=message,
                         commands=commands, dry_run=True).to_dict()

    rc, _ = _run_git(add_cmd, cwd=repo_dir, dry_run=False)
    if rc != 0:
        return GitResult(status="error", artifacts=artifacts, message=message,
                         commands=commands, error="git add failed").to_dict()
    rc, out = _run_git(commit_cmd, cwd=repo_dir, dry_run=False)
    if rc != 0:
        return GitResult(status="error", artifacts=artifacts, message=message,
                         commands=commands, error=f"git commit failed: {out}").to_dict()
    _sha_rc, sha = _run_git(["git", "rev-parse", "HEAD"], cwd=repo_dir, dry_run=False)
    return GitResult(status="committed", artifacts=artifacts, message=message, sha=sha,
                     commands=commands, dry_run=False).to_dict()


def _auto_pr_body(*, summary: str, artifacts: list[str], run_ref: str) -> str:
    """Auto-write the MR description + a test-checklist (the doc 06 §1.3 affordance)."""
    lines = [summary.strip() or "Automated ML artifact merge request.", ""]
    if artifacts:
        lines.append("## Artifacts")
        lines.extend(f"- `{a}`" for a in artifacts)
        lines.append("")
    lines += [
        "## Test checklist",
        "- [ ] Offline offline validation pass",
        "- [ ] No live AWS spend introduced unguarded",
        "- [ ] Approval gates respected for spend/irreversible actions",
        "",
    ]
    if run_ref:
        lines.append(f"Linked-Run: {run_ref}")
    return "\n".join(lines).rstrip() + "\n"


def open_pr(
    *,
    branch: str,
    target_branch: str = "main",
    title: str = "",
    summary: str = "",
    artifacts: list[str] | None = None,
    run_ref: str = "",
    repo_dir: str | Path = ".",
    force: bool = False,
    dry_run: bool = True,
) -> dict[str, Any]:
    """``T-GIT-01`` — push ``branch`` + open a GitLab merge request (US-CC.08).

    Constructs the ``git push`` + ``glab mr create`` (GitLab CLI) commands with an
    auto-written title/description/test-checklist, links the MR back to ``run_ref``, and
    returns the MR URL. Push + MR-open are a REAL external side-effect →
    ``classify_action("open_pr")`` is ``IRREVERSIBLE`` so the control plane GATES it
    before this runs. **Refuses to force-push a protected branch** (``PROTECTED_BRANCHES``)
    unless that is overridden out-of-band — the doc 06 §1.3 safety rail.

    Args:
        branch: The feature branch to push + open the MR from.
        target_branch: The MR target (default ``main``).
        title: MR title (a default is composed when empty).
        summary: One-line summary that seeds the auto-written MR body.
        artifacts: Artifact paths listed in the MR body.
        run_ref: Originating run/spec id linked in the MR body (+ returned).
        repo_dir: The git working tree.
        force: Request a force-push (refused for a protected ``branch``).
        dry_run: When True (default — the test/safe path), construct + RECORD the push/MR
            commands WITHOUT executing them (no push, no network).

    Returns:
        :class:`PRResult` as a dict (``status/branch/target_branch/title/body/pr_url/
        commands/dry_run/linked_run/error``). ``status == "refused"`` when a protected
        force-push is blocked.
    """
    artifacts = list(artifacts or [])
    branch = (branch or "").strip()
    if not branch:
        return PRResult(status="error", target_branch=target_branch,
                        error="no branch supplied", dry_run=dry_run).to_dict()

    # Safety rail: never force-push a protected branch (doc 06 §1.3).
    if force and branch in PROTECTED_BRANCHES:
        return PRResult(
            status="refused", branch=branch, target_branch=target_branch,
            error=f"refusing force-push to protected branch {branch!r}", dry_run=dry_run,
        ).to_dict()

    pr_title = title.strip() or (summary.strip() or f"ML artifacts: {branch}")
    body = _auto_pr_body(summary=summary, artifacts=artifacts, run_ref=run_ref)

    push_cmd = ["git", "push", "--set-upstream", "origin", branch]
    if force:
        push_cmd.insert(2, "--force-with-lease")
    # GitLab CLI (``glab``) is the thin MR-open orchestration; fall back gracefully when absent.
    mr_cmd = [
        "glab", "mr", "create",
        "--source-branch", branch,
        "--target-branch", target_branch,
        "--title", pr_title,
        "--description", body,
    ]
    commands = [push_cmd, mr_cmd]

    if dry_run:
        return PRResult(
            status="dry_run", branch=branch, target_branch=target_branch, title=pr_title,
            body=body, artifacts=artifacts, commands=commands, dry_run=True,
            linked_run=run_ref,
            pr_url=f"<dry-run: would open MR {branch} -> {target_branch}>",
        ).to_dict()

    if shutil.which("glab") is None:
        return PRResult(
            status="error", branch=branch, target_branch=target_branch, title=pr_title,
            body=body, artifacts=artifacts, commands=commands, linked_run=run_ref,
            error="glab (GitLab CLI) not found on PATH; cannot open MR",
        ).to_dict()

    rc, out = _run_git(push_cmd, cwd=repo_dir, dry_run=False)
    if rc != 0:
        return PRResult(status="error", branch=branch, target_branch=target_branch,
                        title=pr_title, body=body, artifacts=artifacts, commands=commands,
                        linked_run=run_ref, error=f"git push failed: {out}").to_dict()
    rc, out = _run_git(mr_cmd, cwd=repo_dir, dry_run=False)
    if rc != 0:
        return PRResult(status="error", branch=branch, target_branch=target_branch,
                        title=pr_title, body=body, artifacts=artifacts, commands=commands,
                        linked_run=run_ref, error=f"glab mr create failed: {out}").to_dict()
    return PRResult(
        status="opened", branch=branch, target_branch=target_branch, title=pr_title,
        body=body, artifacts=artifacts, commands=commands, dry_run=False,
        linked_run=run_ref, pr_url=out.strip(),
    ).to_dict()
