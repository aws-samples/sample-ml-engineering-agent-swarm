"""The Claude-Code-native **control plane** primitives (roadmap M1; doc 05 §3.1).

This package hosts the session/control-plane mechanisms the control agent owns
(doc 05 §3.1; doc 06 ``control_plane_chat_agent`` EVOLVE facets):

  * ``project_memory`` — the ``.agenticml.yaml`` CLAUDE.md-analog project memory
    (load/merge/hierarchy) + ``init_project_memory`` (T-CTL-02, the ``/init`` analog)
    + ``project_init_flow`` (ADR-016; US-CC.01).
  * ``approvals`` — the safe-by-default permission/approval gate (``ask``/``allowlist``/
    ``auto`` modes, action classification, ``request_approval`` T-CTL-03) wired as a
    ``PreToolUse`` guardrail hook (ADR-014; US-CC.03).
  * ``plan_mode`` — read-only ``preview_flow_plan`` (T-CTL-04) + the
    ``plan_approve_execute`` wrapper (ADR-013; US-CC.04).

These are deterministic, pure-Python, boto3-free (the gate consults a cost estimate
passed in; the registry/ports do the actual AWS work). They are the M1 safe-by-default
beat that the MVP demo (doc 07 §7.2) shows before each spend-incurring step.
"""

from __future__ import annotations

__all__ = [
    "ActionClass",
    "ApprovalMode",
    "ApprovalPolicy",
    # M7 — one-capability dispatch (ADR-010) + the runtime/session/mention facets.
    "CapabilityResult",
    "ProjectMemory",
    "Session",
    "approval_gate_hook",
    "classify_action",
    "describe_capability",
    "init_project_memory",
    "invoke_capability",
    "list_capabilities",
    "list_sessions",
    "mention_context",
    "plan_approve_execute",
    "preview_flow_plan",
    "project_init_flow",
    "request_approval",
    "resume_session",
    "run_headless",
    "save_session",
]


def __getattr__(name: str) -> object:  # lazy re-export to keep import cheap + avoid cycles
    if name in ("ProjectMemory", "init_project_memory", "project_init_flow"):
        from . import project_memory as _pm

        return getattr(_pm, name)
    if name in ("ApprovalMode", "ActionClass", "ApprovalPolicy", "request_approval",
                "classify_action", "approval_gate_hook"):
        from . import approvals as _ap

        return getattr(_ap, name)
    if name in ("preview_flow_plan", "plan_approve_execute"):
        from . import plan_mode as _plan

        return getattr(_plan, name)
    # M7 control-plane facets (the ADR-010 funnel + runtime/session/mention).
    if name in ("CapabilityResult", "invoke_capability", "list_capabilities",
                "describe_capability"):
        from . import capability as _cap

        return getattr(_cap, name)
    if name in ("run_headless", "resume_session", "Session", "save_session", "list_sessions"):
        from . import runtime_session as _rs

        return getattr(_rs, name)
    if name == "mention_context":
        from . import mention as _m

        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
