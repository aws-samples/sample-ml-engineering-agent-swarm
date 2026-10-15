"""Headless run + session resume — T-RUN-05 / T-RUN-06 (roadmap M7; doc 05 §8.9 / §5.3).

Two Claude-Code-native runtime affordances over the M7 capability funnel:

  * **``run_headless`` (T-RUN-05, US-CC.09)** — the ``claude -p``/Agent-SDK print-mode
    analog. Runs a capability NON-INTERACTIVELY (CLI ``-p`` / Python / CI) and returns
    STRUCTURED output. Approvals resolve by POLICY only (never a prompt): safe-by-default
    a gated (expensive/irreversible) capability DENIES unless it is explicitly allowlisted
    via ``allowed_tools`` (the ``--allowedTools`` analog) or the trusted ``bypass`` mode
    (the ``--dangerously-skip-permissions`` analog). A ``max_budget`` caps allowlisted
    spend. Cheap-read capabilities stream straight through.

  * **``resume_session`` (T-RUN-06, US-CC.07)** — the ``--continue``/``--resume`` analog.
    Persists conversation/run/orchestration state to a local session store and reattaches
    to a prior session by ref — the checkpoint/rewind analog (doc 05 §5.3): resuming a
    prior exploration/improvement iteration. Reuses the M1 project-memory directory
    convention (``.agenticml/sessions/<ref>.json``) and the M5 Memory carry-over shape, so
    a resumed session restores prior project memory, the kanban/run state, and the Memory
    context the next iteration builds on.

Pure-Python + filesystem — no AWS, no LLM. The capability funnel
(``control_plane.capability.invoke_capability``) does the actual flow dispatch + gating;
this module only frames the headless policy + persists/reattaches session state.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .approvals import ApprovalMode, ApprovalPolicy
from .capability import CapabilityResult, invoke_capability

__all__ = [
    "SESSIONS_DIR",
    "Session",
    "list_sessions",
    "resume_session",
    "run_headless",
    "save_session",
]

SESSIONS_DIR = ".agenticml/sessions"


# ── T-RUN-05 run_headless (the ``claude -p`` analog) ─────────────────────────
def _headless_policy(
    *,
    allowed_tools: list[str] | None,
    max_budget: float | None,
    bypass: bool,
) -> ApprovalPolicy:
    """Build the non-interactive approval policy for a headless run (safe-by-default).

    * ``bypass=True`` → ``AUTO`` mode (the explicit ``--dangerously-skip-permissions``
      trusted-automation analog): everything pre-satisfied.
    * ``allowed_tools`` non-empty → ``ALLOWLIST`` mode capped at ``max_budget``: only the
      named capabilities/actions auto-approve (the ``--allowedTools`` analog); everything
      else gates → DENIES (non-interactive).
    * otherwise → ``ASK`` mode, ``interactive=False`` → gated capabilities DENY (never a
      prompt, never an unattended spend). Cheap reads still stream through in all modes.
    """
    if bypass:
        return ApprovalPolicy(mode=ApprovalMode.AUTO, interactive=False)
    if allowed_tools:
        return ApprovalPolicy(
            mode=ApprovalMode.ALLOWLIST,
            allowlist=list(allowed_tools),
            max_auto_cost=max_budget,
            interactive=False,
        )
    return ApprovalPolicy(mode=ApprovalMode.ASK, interactive=False)


def run_headless(
    *,
    capability_id: str,
    params: dict[str, Any] | None = None,
    allowed_tools: list[str] | None = None,
    max_budget: float | None = None,
    bypass: bool = False,
    steering: Any | None = None,
) -> dict[str, Any]:
    """``T-RUN-05`` — run a capability non-interactively, returning structured output (US-CC.09).

    The ``claude -p`` analog for CI/background: builds a non-interactive policy (see
    :func:`_headless_policy` — safe-by-default), invokes the capability through the SAME
    M7 funnel every interactive surface uses (``invoke_capability``), and returns a
    JSON-able result dict. A gated (expensive/irreversible) capability is DENIED unless
    its ``capability_id`` is in ``allowed_tools`` (under ``max_budget``) or ``bypass`` is
    set — so an unattended CI run never silently spends.

    Args:
        capability_id: The flow/capability to run (e.g. ``"data_quality_flow"``).
        params: Keyword params forwarded to the flow executor.
        allowed_tools: Capability/action names to allowlist (the ``--allowedTools`` analog).
        max_budget: Optional spend cap applied to allowlisted gated actions.
        bypass: Trusted-automation bypass (``--dangerously-skip-permissions`` analog).
        steering: Optional steering profile forwarded to the funnel.

    Returns:
        ``{"capability_id","surface":"headless","status","approved","gated","executed",
           "plan","decision","result","error","allowed_tools","max_budget","bypass"}``
        — the structured, machine-readable headless contract.
    """
    policy = _headless_policy(allowed_tools=allowed_tools, max_budget=max_budget, bypass=bypass)
    outcome: CapabilityResult = invoke_capability(
        capability_id=capability_id,
        params=params,
        policy=policy,
        steering=steering,
        surface="headless",
    )
    out = outcome.to_dict()
    out["allowed_tools"] = list(allowed_tools or [])
    out["max_budget"] = max_budget
    out["bypass"] = bypass
    return out


# ── T-RUN-06 resume_session (the ``--continue``/``--resume`` analog) ─────────
@dataclass
class Session:
    """A persisted conversation/run session (the checkpoint/rewind state; doc 05 §5.3).

    Args:
        session_ref: Stable id used to resume (``--resume <ref>``).
        created_at / updated_at: Unix timestamps.
        project_memory: Snapshot of the merged ``.agenticml.yaml`` selectors/steering.
        history: The conversation/turn history (list of ``{role, content}`` dicts).
        runs: Orchestration/run state (the kanban + retreat-edge iterations — the M1/M5
            run records the next iteration resumes from).
        memory: The M5 AgentCore-Memory carry-over context (findings carried forward).
        metadata: Free-form extra state.
    """

    session_ref: str
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    project_memory: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    runs: list[dict[str, Any]] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_ref": self.session_ref,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "project_memory": self.project_memory,
            "history": self.history,
            "runs": self.runs,
            "memory": self.memory,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        return cls(
            session_ref=str(data.get("session_ref", "")),
            created_at=float(data.get("created_at", time.time())),
            updated_at=float(data.get("updated_at", time.time())),
            project_memory=dict(data.get("project_memory", {})),
            history=list(data.get("history", [])),
            runs=list(data.get("runs", [])),
            memory=dict(data.get("memory", {})),
            metadata=dict(data.get("metadata", {})),
        )


def _sessions_dir(project_dir: str | Path) -> Path:
    return Path(project_dir).expanduser() / SESSIONS_DIR


def _session_path(session_ref: str, project_dir: str | Path) -> Path:
    safe = (session_ref or "").strip().replace("/", "_").replace("..", "_")
    return _sessions_dir(project_dir) / f"{safe}.json"


def save_session(session: Session, *, project_dir: str | Path = ".") -> dict[str, Any]:
    """Persist ``session`` to ``<project>/.agenticml/sessions/<ref>.json`` (atomic-ish).

    Updates ``updated_at`` and writes via a temp file + rename so a concurrent reader
    never sees a half-written session. Returns ``{"session_ref","path","saved": True}``.
    """
    session.updated_at = time.time()
    target = _session_path(session.session_ref, project_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(session.to_dict(), indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(target)
    return {"session_ref": session.session_ref, "path": str(target), "saved": True}


def list_sessions(*, project_dir: str | Path = ".") -> list[str]:
    """Return the session refs available to resume (sorted)."""
    d = _sessions_dir(project_dir)
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def resume_session(*, session_ref: str, project_dir: str | Path = ".") -> dict[str, Any]:
    """``T-RUN-06`` — reattach to a prior session by ref (US-CC.07; the ``--resume`` analog).

    Loads the persisted :class:`Session` and returns the restored context — prior project
    memory, conversation history, orchestration/run state (the kanban + retreat-edge
    iterations), and the M5 Memory carry-over — so the control plane can re-enter exactly
    where the prior exploration/improvement iteration left off (the checkpoint/rewind
    analog, doc 05 §5.3).

    Args:
        session_ref: The session id to resume (as persisted by :func:`save_session`).
        project_dir: The project root holding the sessions dir.

    Returns:
        ``{"found": bool, "session_ref", "context": {...}}`` where ``context`` carries the
        restored ``project_memory``/``history``/``runs``/``memory``/``metadata``. ``found``
        is False (with an ``error``) when no session matches the ref.
    """
    path = _session_path(session_ref, project_dir)
    if not path.is_file():
        return {
            "found": False,
            "session_ref": session_ref,
            "context": {},
            "error": f"no session named {session_ref!r} under {_sessions_dir(project_dir)}",
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return {"found": False, "session_ref": session_ref, "context": {}, "error": str(e)}
    session = Session.from_dict(data)
    return {
        "found": True,
        "session_ref": session.session_ref,
        "context": {
            "project_memory": session.project_memory,
            "history": session.history,
            "runs": session.runs,
            "memory": session.memory,
            "metadata": session.metadata,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
        },
    }
