"""Approval / permission gates — safe-by-default (doc 05 §3.1a/§8.8; ADR-014; T-CTL-03).

The Claude-Code permission analog: our control plane **never spends money or changes
production without explicit approval** ("never modifies your files without approval").
This module is the policy + the ``PreToolUse`` guardrail that ENFORCES it (doc 05 §8.9
"the approval gate sits in the hooks layer, invoked by the control plane" — Q9 lean).

**Action classification (doc 05 §8.8 table):**
  * ``CHEAP_READ`` — profile/read-slice/catalog/describe/research/small-sandbox →
    **auto-approved** (streams like any read; never prompts).
  * ``BOUNDED_COMPUTE`` — small in-proc train/tune, generate-only AutoML →
    auto-approved up to a budget, else gated.
  * ``EXPENSIVE_JOB`` — submit_training_run/AMT/Autopilot, processing/Glue/DataBrew,
    batch-transform → **gated** with a cost/runtime estimate.
  * ``IRREVERSIBLE`` — deploy/canary/delete/retire/grant-access → **gated** (always
    ask, even with allowlist unless explicitly allowlisted per action).

**Three modes (doc 05 §8.8, mirroring Claude Code ``ask``/``allowlist``/``auto``):**
  * ``ASK`` (default, safe-by-default) — prompt before each spend/state-change. With
    an interactive ``responder`` callback the user decides; **non-interactive ASK
    denies** (never auto-spends headlessly).
  * ``ALLOWLIST`` — persisted "always allow" decisions per action (optionally under a
    ``max_auto_cost`` cap); non-allowlisted actions still gate.
  * ``AUTO`` — the deliberate trusted-automation bypass (headless/CI) — everything
    pre-satisfied (the explicit ``--dangerously-skip-permissions`` analog).

``request_approval`` (T-CTL-03) returns a structured decision; ``approval_gate_hook``
adapts the policy into a deterministic ``PreToolUse`` :class:`~...backends.hooks.Hook`
whose deny the model cannot override (ADR-015). Pure-Python; no AWS — the gate
consults a cost estimate the caller passes in (the registry/ports do the AWS work).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..backends.hooks import Hook, HookContext, HookEvent
from ..backends.types import Verdict

# The env seam for the APPROVED capped live-run per-run ceiling (gap-closure PREP). Defined in
# cost_governance so the cost surface + the approval policy read the SAME var (one source of truth).
from .cost_governance import ENV_MAX_AUTO_COST, _safe_env_float

__all__ = [
    "ActionClass",
    "ApprovalMode",
    "ApprovalPolicy",
    "approval_gate_hook",
    "classify_action",
    "request_approval",
]


class ApprovalMode(StrEnum):
    """The three permission modes (+ AUTO bypass) — doc 05 §8.8."""

    ASK = "ask"
    ALLOWLIST = "allowlist"
    AUTO = "auto"


class ActionClass(StrEnum):
    """Disposition class of a tool/port action (doc 05 §8.8 table)."""

    CHEAP_READ = "cheap_read"
    BOUNDED_COMPUTE = "bounded_compute"
    EXPENSIVE_JOB = "expensive_job"
    IRREVERSIBLE = "irreversible"


# ── classification tables (doc 05 §8.8) ─────────────────────────────────────
_CHEAP_READS = {
    "run_profile", "read_lake_slice", "list_lake_catalog", "describe_dataset",
    "list_catalog", "read_slice", "research", "web_retrieve", "preview_flow_plan",
    # M4 Stage-9 readiness tools are pure-reasoning / read-only assessments — they
    # produce a go/no-go review, they do NOT spend or change production (doc 06 §5.3
    # readiness_flow). The deploy actions they precede are what gate (below).
    "run_readiness_gate", "build_risk_register", "compile_readiness_review",
    "design_production_architecture", "draft_operational_runbook",
    "select_deployment_strategy", "recommend_inference_config", "readiness_flow",
    # M5 Stage-11 monitoring is read/observe-heavy — dashboards, metrics, drift analysis,
    # alerts, reports, and the retrain-trigger DECISION provision nothing irreversible
    # (the managed schedule-setup sub-step is gated within the flow). doc 06 §5.3 monitor_flow.
    "monitor_flow", "analyze_drift", "get_metrics", "get_recent_logs",
    "compile_monitoring_report", "evaluate_retrain_trigger", "analyze_production_failures",
    "build_monitoring_dashboard", "track_inference_spend",
    # M6 Stage 0-3 research / framing / solution-survey are read + pure-reasoning — they
    # scan/index research, structure problem cards, and survey model options; they provision
    # nothing irreversible (the RetrievalBackend is a local TF-IDF read offline ↔ Bedrock KB
    # managed; ModelHub list is a catalog read). Only Stage-4 labeling-job creation gates
    # (below). doc 06 §5.3 research_intel_flow / business_framing_flow / solution_survey_flow.
    "research_intel_flow", "business_framing_flow", "solution_survey_flow",
    "scan_research", "triage_reproducibility", "generate_research_digest", "schedule_research_scan",
    "capture_business_problem", "define_business_kpi", "assess_ml_necessity", "business_go_no_go",
    "select_problem_type", "define_success_metrics", "build_problem_card", "check_stage_exit",
    "list_jumpstart_models", "survey_managed_services", "apply_research_gate",
    "build_candidate_shortlist", "get_labeling_progress", "run_sql",
    "index_research_corpus", "chunk_documents", "query_research_corpus", "build_retrieval_index",
    "build_metadata_filters", "tune_retrieval",
    # Shift-left AWS control plane (plan §A2/§A3): readiness reports the allowlist and reads
    # selected configuration, and the preflight remediation PLANNER only maps
    # failed checks to the tool that fixes each one — applying is delegated to those tools,
    # which gate individually as IRREVERSIBLE. Neither may demand an approval, or the agent
    # could not even diagnose a blocked dispatch. ``remediate_…`` does not start with a
    # read-ish prefix, so it must be listed explicitly.
    "describe_aws_readiness", "remediate_dispatch_preflight",
    # Plan §B3: rendering a dashboard chart reads existing artifacts and publishes a view
    # request — no spend, no provisioning, nothing irreversible.
    "render_dashboard_view",
}
_BOUNDED_COMPUTE = {
    "train_model", "tune_model", "run_automl_generate_only", "score_data_quality_6dim",
    # M7: a LOCAL git commit of an ML artifact (T-GIT-02 commit_artifact) touches the
    # working tree but not a remote — a bounded, recoverable side-effect → auto-approved
    # under the bounded cap. The PUSH that publishes it (open_pr) is what gates (below).
    "commit_artifact",
}
_EXPENSIVE_JOBS = {
    "submit_training_run", "submit_training_job", "run_hpo", "run_automl",
    "run_processing_job", "run_glue_job", "run_crawler", "apply_recipe",
    "run_batch_transform",
    # M3: the spend-incurring train/tune flow IDs are gated as EXPENSIVE_JOB so the
    # plan_approve_execute wrapper denies them under non-interactive ASK (never
    # auto-approving training/HPO spend as cheap "bounded compute"). doc 05 §8.8.
    "baseline_train_flow", "hpo_flow",
}
_IRREVERSIBLE = {
    "deploy_endpoint", "update_endpoint_canary", "delete_endpoint", "retire_model",
    "grant_data_access", "rollback_deployment", "run_shadow_deployment", "run_ab_test",
    # G3: deploying a compiled model to an edge/embedded device (Greengrass V2 component +
    # fleet deployment) provisions an external, not-freely-reversible artifact → IRREVERSIBLE.
    # The ``deploy`` substring hint already catches it, but list it explicitly for clarity.
    "deploy_edge",
    # M5: the closed-loop improvement flow retrains (spend) then redeploys/retires to
    # PRODUCTION — expensive + IRREVERSIBLE → gated by plan_approve_execute before any
    # spend (ADR-008/014; "retrain" is not a deploy/training substring, so list it). The
    # monitor_flow that DECIDES the trigger stays a CHEAP_READ; only the action gates.
    "improve_retrain_flow",
    # M6: creating a Ground-Truth data-labeling job incurs cost + provisions a managed job
    # (not freely reversible) → gated. The data_collection_flow inventory/strategy steps stay
    # CHEAP_READ; only the labeling-job creation gates (ADR-014). doc 06 §5.3 / doc 05 §8.8.
    "create_labeling_job", "data_collection_flow",
    # M7: opening a PR/MR PUSHES the branch + creates a merge request — a REAL external
    # side-effect that triggers the PR-based AWS security review → gated (US-CC.08; doc 06
    # §1.3 devops_git_agent). The LOCAL commit_artifact stays BOUNDED_COMPUTE (above); only
    # the push/MR gates. devops_git_flow composes them, so the flow gates too.
    "open_pr", "devops_git_flow",
    # Shift-left AWS control plane (plan §A2/§A4, tools/aws_control.py): the agents now
    # CONFIGURE AWS — create lake prefixes, catalog databases, IAM job roles, scoped
    # grants, quota requests. Each provisions or re-permissions real account state that is
    # not freely reversible, so every mutator gates here. The read-only
    # ``describe_aws_readiness`` probe and the plan-only ``remediate_dispatch_preflight``
    # planner stay CHEAP_READ (they are covered by the read-prefix rule below).
    # NOTE: ``grant_role_policy`` is also caught by the "grant" substring hint, and
    # ``ensure_iam_service_role`` by none — hence the explicit list.
    "ensure_s3_prefix", "ensure_glue_database", "ensure_iam_service_role",
    "grant_role_policy", "request_quota_increase",
}

# substring fallbacks so unseen variants still classify safely (deploy/delete → gated)
_IRREVERSIBLE_HINTS = ("deploy", "delete", "retire", "canary", "grant", "rollback")
_EXPENSIVE_HINTS = ("training", "processing", "glue", "databrew", "batch_transform", "tuning")


def classify_action(tool: str) -> ActionClass:
    """Classify ``tool`` into a disposition class (doc 05 §8.8 table; safe-by-default).

    Exact-name tables first, then substring hints (so an unseen ``deploy_*`` still
    gates as irreversible rather than falling through to a cheap read).
    """
    name = (tool or "").strip().lower()
    if name in _CHEAP_READS:
        return ActionClass.CHEAP_READ
    if name in _IRREVERSIBLE:
        return ActionClass.IRREVERSIBLE
    if name in _EXPENSIVE_JOBS:
        return ActionClass.EXPENSIVE_JOB
    if name in _BOUNDED_COMPUTE:
        return ActionClass.BOUNDED_COMPUTE
    # substring fallbacks (gate-leaning — safe-by-default)
    if any(h in name for h in _IRREVERSIBLE_HINTS):
        return ActionClass.IRREVERSIBLE
    if any(h in name for h in _EXPENSIVE_HINTS):
        return ActionClass.EXPENSIVE_JOB
    # unknown → treat as cheap read ONLY if it looks read-only; else bounded compute.
    if name.startswith(("read", "list", "get", "describe", "preview", "profile")):
        return ActionClass.CHEAP_READ
    return ActionClass.BOUNDED_COMPUTE


@dataclass
class ApprovalPolicy:
    """The configured approval policy (mode + allowlist + budget cap + responder).

    Args:
        mode: ``ask`` (default, safe), ``allowlist``, or ``auto`` (bypass).
        allowlist: Action names persisted as "always allow" (allowlist mode).
        max_auto_cost: Optional "always allow under $X" cap for allowlisted /
            bounded-compute actions; over-cap falls back to a gate.
        interactive: Whether an interactive responder is available (ask mode). When
            False (headless), ask-mode gated actions DENY (never auto-spend).
        responder: Optional ``(request) -> bool`` callback the ask-mode prompt calls.
        bounded_compute_cap: Auto-approve bounded compute up to this cost (else gate).
    """

    mode: ApprovalMode = ApprovalMode.ASK
    allowlist: list[str] = field(default_factory=list)
    max_auto_cost: float | None = None
    interactive: bool = False
    responder: Callable[[dict[str, Any]], bool] | None = None
    bounded_compute_cap: float = 1.0

    @classmethod
    def from_memory(cls, memory: dict[str, Any] | None) -> ApprovalPolicy:
        """Build a policy from the project-memory ``approval`` block (safe-by-default).

        Missing/empty config → ``ask`` mode (the safe default). Recognized keys:
        ``mode``, ``allowlist`` (list), ``max_auto_cost`` (float), ``interactive`` (bool).
        """
        block: dict[str, Any] = {}
        if isinstance(memory, dict):
            approval = memory.get("approval")
            block = approval if isinstance(approval, dict) else {}
        mode_raw = str((block or {}).get("mode", "ask")).strip().lower()
        try:
            mode = ApprovalMode(mode_raw)
        except ValueError:
            mode = ApprovalMode.ASK
        allow = (block or {}).get("allowlist") or []
        max_cost = (block or {}).get("max_auto_cost")
        return cls(
            mode=mode,
            allowlist=[str(a) for a in allow] if isinstance(allow, list) else [],
            max_auto_cost=float(max_cost) if max_cost is not None else None,
            interactive=bool((block or {}).get("interactive", False)),
        )

    @classmethod
    def from_env(cls, *, allowlist: list[str] | None = None) -> ApprovalPolicy:
        """Build a policy whose per-run ceiling is the env-sourced ``AMLAS_MAX_AUTO_COST_USD``.

        The APPROVED capped-live-run wiring (gap-closure PREP). The confirmed **$10 per-run
        max-auto-cost ceiling** is read from ``AMLAS_MAX_AUTO_COST_USD`` (safe blocking default
        ``0.0`` when unset — never ``None``/unbounded). When an ``allowlist`` of flow/action ids is
        supplied the policy is ``ALLOWLIST`` mode capped at that ceiling (allowlisted actions UNDER
        the ceiling auto-approve, OVER gate); with NO allowlist it is non-interactive ``ASK`` —
        safe-by-default, so a gated spend DENIES headlessly. This reuses the EXISTING ``max_auto_cost``
        field + allowlist-cap path; it does NOT introduce a parallel mechanism.
        """
        ceiling = _safe_env_float(ENV_MAX_AUTO_COST, 0.0)
        names = [str(a) for a in (allowlist or [])]
        if names:
            return cls(
                mode=ApprovalMode.ALLOWLIST,
                allowlist=names,
                max_auto_cost=ceiling,
                interactive=False,
            )
        return cls(mode=ApprovalMode.ASK, max_auto_cost=ceiling, interactive=False)


def request_approval(
    *,
    action: str,
    cost_estimate: float = 0.0,
    impact: str = "",
    policy: ApprovalPolicy | None = None,
) -> dict[str, Any]:
    """``T-CTL-03`` — decide whether ``action`` may proceed (the approval primitive).

    Returns a structured decision dict::

        {"decision": "approved"|"denied", "action", "cost_estimate", "impact",
         "action_class", "mode", "gated": bool, "auto": bool, "allowlisted": bool,
         "reason"}

    Disposition logic (doc 05 §8.8):
      * cheap reads → always approved, never gated (stream like a read);
      * AUTO mode → approved (the trusted bypass);
      * ALLOWLIST mode → approved iff the action is allowlisted AND under any cap;
      * ASK mode → the interactive responder decides; non-interactive ASK DENIES a
        gated action (safe-by-default — never auto-spends headlessly).
    """
    policy = policy or ApprovalPolicy()
    cls = classify_action(action)

    def _decision(decision: str, *, gated: bool, reason: str, **flags: Any) -> dict[str, Any]:
        out = {
            "decision": decision,
            "action": action,
            "cost_estimate": cost_estimate,
            "impact": impact,
            "action_class": cls.value,
            "mode": policy.mode.value,
            "gated": gated,
            "auto": False,
            "allowlisted": False,
            "reason": reason,
        }
        out.update(flags)
        return out

    # Cheap reads stream like any read — never gated (even in ask mode).
    if cls == ActionClass.CHEAP_READ:
        return _decision("approved", gated=False, reason="cheap read — auto-approved")

    # Bounded compute auto-approves under the bounded cap, else is treated as gated.
    is_gated = True
    if cls == ActionClass.BOUNDED_COMPUTE and cost_estimate <= policy.bounded_compute_cap:
        return _decision("approved", gated=False, reason="bounded compute under cap — auto-approved")

    # AUTO/bypass — trusted headless/CI: everything pre-satisfied.
    if policy.mode == ApprovalMode.AUTO:
        return _decision("approved", gated=is_gated, reason="auto/bypass mode (trusted automation)", auto=True)

    # ALLOWLIST — approve iff allowlisted AND under any cost cap.
    if policy.mode == ApprovalMode.ALLOWLIST:
        if action in policy.allowlist:
            if policy.max_auto_cost is not None and cost_estimate > policy.max_auto_cost:
                return _decision(
                    "denied", gated=True,
                    reason=f"allowlisted but over max_auto_cost ${policy.max_auto_cost} (gated)",
                    allowlisted=True,
                )
            return _decision("approved", gated=is_gated, reason="allowlisted (always-allow)", allowlisted=True)
        return _decision("denied", gated=True, reason="not on allowlist — requires approval")

    # ASK (default, safe) — interactive responder decides; headless ASK denies.
    if policy.interactive and policy.responder is not None:
        req = {"action": action, "cost_estimate": cost_estimate, "impact": impact, "action_class": cls.value}
        try:
            approved = bool(policy.responder(req))
        except Exception:
            approved = False
        return _decision(
            "approved" if approved else "denied",
            gated=True,
            reason="approved by responder" if approved else "denied by responder",
        )
    # non-interactive ask mode → safe-by-default DENY (never auto-spend headlessly)
    return _decision("denied", gated=True, reason="ask mode + non-interactive — gated (safe-by-default)")


# ── the PreToolUse guardrail (doc 05 §8.9 — the gate lives in the hooks layer) ──
def approval_gate_hook(policy: ApprovalPolicy | None = None) -> Hook:
    """Adapt an :class:`ApprovalPolicy` into a deterministic ``PreToolUse`` guardrail.

    The hook MATCHES spend-incurring / irreversible actions (cheap reads do not match
    — they stream); its ACTION runs :func:`request_approval` and returns a denying
    :class:`Verdict` for any non-approved action, which :class:`GuardrailHooks` honors
    fail-closed (the model cannot override — ADR-015). The cost estimate is read from
    ``ctx.args['cost_estimate']`` when the control plane supplies it.
    """
    policy = policy or ApprovalPolicy()

    def _matches(ctx: HookContext) -> bool:
        # Only gate the spend/irreversible classes — cheap reads stream past the gate.
        return classify_action(ctx.tool or ctx.action) in (
            ActionClass.EXPENSIVE_JOB,
            ActionClass.IRREVERSIBLE,
            ActionClass.BOUNDED_COMPUTE,
        )

    def _action(ctx: HookContext) -> Verdict:
        cost = float(ctx.args.get("cost_estimate", 0.0) or 0.0)
        impact = str(ctx.args.get("impact", ""))
        decision = request_approval(
            action=ctx.tool or ctx.action, cost_estimate=cost, impact=impact, policy=policy
        )
        if decision["decision"] == "approved":
            return Verdict(allowed=True, reason=decision["reason"], detail={"approval": decision})
        return Verdict(
            allowed=False,
            reason=f"approval gate: {decision['reason']}",
            detail={"approval": decision},
        )

    return Hook(name="approval_gate", event=HookEvent.PRE_TOOL_USE, matcher=_matches, action=_action)
