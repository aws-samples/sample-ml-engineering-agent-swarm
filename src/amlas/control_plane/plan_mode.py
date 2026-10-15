"""Plan Mode — ``preview_flow_plan`` (T-CTL-04) + ``plan_approve_execute`` (doc 05 §3.1b; ADR-013).

The Claude Code Plan-Mode analog: **think first, build second.** Before any spend, the
control plane produces a READ-ONLY plan — the numbered steps, the ports/tools each will
call, an estimated spend, and irreversibility flags — and provisions/executes NOTHING
(US-CC.04). ``plan_approve_execute`` is the cross-cutting WRAPPER (doc 06 flow row) that
every spend-incurring flow is wrapped with: **preview → approval gate → execute**, and
**aborts on denial** so no money is spent / production changed until the user approves.

For M1 it wraps the orchestration entrypoint (``exploration_loop``) so an exploration
run can be plan-previewed + approved before it spends; later milestones wrap
``imputation_flow`` apply-at-scale, ``baseline_train_flow``, ``deploy_flow``, etc.

Pure-Python: the plan is descriptive (it consults the static flow template + a cheap
cost estimate); ``preview`` NEVER calls the executor (the canary test asserts this).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .approvals import ApprovalPolicy, request_approval

__all__ = ["plan_approve_execute", "preview_flow_plan"]


# ── flow templates (the read-only plan shapes) ──────────────────────────────
# Each entry: ordered steps (step label + the port/tool it will call), an estimated
# spend, and the spend/irreversibility flags. These describe the M1+ flows the wrapper
# guards; the catalog (doc 06 §5.3) is the source of the step→port mapping.
_FLOW_TEMPLATES: dict[str, dict[str, Any]] = {
    "exploration_loop": {
        "steps": [
            {"step": "research", "port": "ModelBackend", "tool": "research_lead"},
            {"step": "feature_engineering", "port": "ETLBackend", "tool": "fe_lead"},
            {"step": "ml_execution", "port": "TrainingBackend", "tool": "submit_training_run"},
            {"step": "arbiter_grade", "port": "EvaluationBackend", "tool": "grade"},
        ],
        "estimated_cost": 0.40,
        "spend_incurring": True,
        "irreversible": False,
    },
    "ml_pipeline": {
        "steps": [
            {"step": "research", "tool": "recommend_approaches"},
            {"step": "experiment_plan", "tool": "create_plan"},
            {"step": "train", "port": "TrainingBackend", "tool": "train_model"},
            {"step": "evaluate", "port": "EvaluationBackend", "tool": "grade"},
            {"step": "decision", "tool": "DecisionEngine.analyze"},
        ],
        "estimated_cost": 0.10,
        "spend_incurring": True,
        "irreversible": False,
    },
    "baseline_train_flow": {
        "steps": [
            {"step": "submit_training_run", "port": "TrainingBackend", "tool": "submit_training_run"},
            {"step": "grade", "port": "EvaluationBackend", "tool": "grade"},
            {"step": "register", "port": "ModelRegistryBackend", "tool": "register_model"},
        ],
        "estimated_cost": 0.40,
        "spend_incurring": True,
        "irreversible": False,
    },
    "deploy_flow": {
        "steps": [
            {"step": "readiness_pass", "port": "DeploymentBackend", "tool": "compile_readiness_review"},
            {"step": "package_model", "port": "DeploymentBackend", "tool": "package_model"},
            {"step": "select_strategy", "port": "DeploymentBackend", "tool": "select_deployment_strategy"},
            {"step": "deploy_endpoint", "port": "DeploymentBackend", "tool": "deploy_endpoint"},
            {"step": "smoke_invoke", "port": "DeploymentBackend", "tool": "invoke_endpoint"},
            {"step": "cut_release", "port": "DeploymentBackend", "tool": "cut_release"},
            {"step": "wire_monitoring", "port": "MonitoringBackend", "tool": "setup_model_monitor"},
        ],
        "estimated_cost": 0.12,
        "spend_incurring": True,
        "irreversible": True,
    },
    # readiness_flow is the Stage-9 go/no-go gate: it runs the 10 readiness gates +
    # risk register + review. It is READ-ONLY/pure-reasoning (it provisions nothing),
    # so it is NOT spend-incurring or irreversible — it PRECEDES (and gates) deploy_flow.
    "readiness_flow": {
        "steps": [
            {"step": "run_readiness_gates", "port": "DeploymentBackend", "tool": "run_readiness_gate"},
            {"step": "build_risk_register", "port": "DeploymentBackend", "tool": "build_risk_register"},
            {"step": "compile_readiness_review", "port": "DeploymentBackend", "tool": "compile_readiness_review"},
        ],
        "estimated_cost": 0.0,
        "spend_incurring": False,
        "irreversible": False,
    },
    # monitor_flow (M5, Stage-11) is read/observe-heavy → CHEAP_READ overall (it provisions
    # nothing irreversible; the managed schedule-setup sub-step is the only spend, gated
    # within the flow). Its terminal output is the closed-loop trigger decision (ADR-008).
    "monitor_flow": {
        "steps": [
            {"step": "setup_baseline_schedule", "port": "MonitoringBackend", "tool": "setup_model_monitor"},
            {"step": "collect_metrics", "port": "MonitoringBackend", "tool": "get_metrics"},
            {"step": "analyze_drift", "port": "MonitoringBackend", "tool": "analyze_drift"},
            {"step": "configure_alert", "port": "MonitoringBackend", "tool": "configure_alert"},
            {"step": "compile_report", "port": "MonitoringBackend", "tool": "compile_monitoring_report"},
            {"step": "evaluate_retrain_trigger", "port": "OrchestrationBackend", "tool": "evaluate_retrain_trigger"},
        ],
        "estimated_cost": 0.0,
        "spend_incurring": False,
        "irreversible": False,
    },
    # improve_retrain_flow (M5, Stage-12) is the loop-closer: failures → memory → retrain
    # (re-enters the unified retreat loop) → re-gate → redeploy/retire. Retrain + redeploy are
    # EXPENSIVE + IRREVERSIBLE → gated by plan_approve_execute BEFORE any spend (ADR-008/014).
    "improve_retrain_flow": {
        "steps": [
            {"step": "analyze_production_failures", "tool": "analyze_production_failures"},
            {"step": "carry_forward_learnings", "port": "MemoryBackend", "tool": "carry_forward_learnings"},
            {"step": "retrain_via_retreat_loop", "port": "OrchestrationBackend", "tool": "run_flow"},
            {"step": "arbiter_gate", "port": "EvaluationBackend", "tool": "grade"},
            {"step": "readiness_regate", "port": "DeploymentBackend", "tool": "run_readiness_gate"},
            {"step": "redeploy", "port": "DeploymentBackend", "tool": "deploy_endpoint"},
        ],
        "estimated_cost": 0.50,
        "spend_incurring": True,
        "irreversible": True,
    },
    # data_collection_flow (M6, Stage-4) inventories the lake + decides a labeling strategy
    # (CHEAP_READ), then CREATES a Ground-Truth labeling job — the IRREVERSIBLE/spend step. Only
    # the labeling-job creation is gated by plan_approve_execute; the plan reflects that (the
    # inventory/strategy steps run ungated before the wrapper). doc 06 §5.3 data_collection_flow.
    "data_collection_flow": {
        "steps": [
            {"step": "create_labeling_job", "port": "LabelingBackend", "tool": "create_labeling_job"},
            {"step": "get_labeling_progress", "port": "LabelingBackend", "tool": "get_labeling_progress"},
        ],
        "estimated_cost": 0.08,
        "spend_incurring": True,
        "irreversible": True,
    },
    # devops_git_flow (M7, US-CC.08) commits ML artifacts locally then PUSHES the branch +
    # opens the GitLab MR. The local commit_artifact is bounded; the push/open_pr is the
    # IRREVERSIBLE external side-effect → the flow is gated by plan_approve_execute so a
    # denial aborts BEFORE the push (the M7 git-authoring approval gate; doc 06 §1.3).
    "devops_git_flow": {
        "steps": [
            {"step": "commit_artifact", "port": "VCS", "tool": "commit_artifact"},
            {"step": "open_pr", "port": "VCS", "tool": "open_pr"},
        ],
        "estimated_cost": 0.0,
        "spend_incurring": True,
        "irreversible": True,
    },
}

# A safe default template for an unknown flow: treat as spend-incurring + irreversible
# (safe-by-default — an unrecognized flow gates).
_DEFAULT_TEMPLATE = {
    "steps": [{"step": "run_flow", "port": "OrchestrationBackend", "tool": "run_flow"}],
    "estimated_cost": 0.0,
    "spend_incurring": True,
    "irreversible": True,
}


def preview_flow_plan(
    *,
    flow_id: str,
    params: dict[str, Any] | None = None,
    steering: Any | None = None,
    executor: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """``T-CTL-04`` — produce a READ-ONLY plan for ``flow_id`` WITHOUT executing it.

    Returns the numbered steps (each naming the port/tool it will call), an estimated
    spend, and irreversibility/spend flags. Provisions nothing; the optional
    ``executor`` is accepted for signature symmetry with ``plan_approve_execute`` but
    is **never invoked** here (the read-only contract — doc 05 §3.1b).

    Args:
        flow_id: The flow to preview (e.g. ``"exploration_loop"``, ``"deploy_flow"``).
        params: The params the flow would run with (echoed into the plan for review).
        steering: Optional steering profile (echoed; reserved for cost-aware previews).
        executor: Accepted but NEVER called (preview is read-only).

    Returns:
        ``{"flow_id","steps","estimated_cost","cost_band","spend_incurring",
           "irreversible","read_only": True, "executed": False, "params"}``.

    The ``estimated_cost`` is the RAW price-book point estimate — it is what the approval
    gate consumes (see :func:`plan_approve_execute`), so it is left byte-identical. The
    ADDITIVE ``cost_band`` is the human-facing view: a caveated, upward-biased range +
    confidence/source label (:func:`~.cost_confidence.band_for`) so a surface never presents a
    bare confident dollar figure for a price-book guess ("be cautious with cost estimates").
    """
    # NOTE: ``executor`` is intentionally ignored — preview must not execute anything.
    template = _FLOW_TEMPLATES.get((flow_id or "").strip().lower(), _DEFAULT_TEMPLATE)
    steps = [dict(s) for s in template["steps"]]
    estimated_cost = template["estimated_cost"]
    # Additive, presentation-only caveated band (never feeds the gate — see the docstring).
    from .cost_confidence import band_for

    cost_band = band_for(estimated_cost, source="estimate").to_dict()
    return {
        "flow_id": flow_id,
        "steps": steps,
        "estimated_cost": estimated_cost,
        "cost_band": cost_band,
        "spend_incurring": template["spend_incurring"],
        "irreversible": template["irreversible"],
        "read_only": True,
        "executed": False,
        "params": dict(params or {}),
        "steering": steering,
    }


def plan_approve_execute(
    *,
    flow_id: str,
    params: dict[str, Any] | None = None,
    executor: Callable[..., Any],
    policy: ApprovalPolicy | None = None,
    steering: Any | None = None,
) -> dict[str, Any]:
    """The cross-cutting wrapper: **preview → approval gate → execute** (doc 05 §3.1b).

    1. ``preview_flow_plan`` builds the read-only plan (steps + cost + flags);
    2. the approval gate (:func:`request_approval`, T-CTL-03) decides using the plan's
       estimated cost + the flow's disposition class;
    3. ONLY on approval is ``executor(**params)`` invoked; on denial the wrapper
       ABORTS without executing (no spend, no state change).

    Args:
        flow_id: The spend-incurring flow being wrapped (M1: ``exploration_loop``).
        params: The params forwarded to ``executor`` on approval.
        executor: The callable that actually runs the flow (e.g. the LocalDriver
            ``run_flow`` bound entrypoint). Invoked as ``executor(**params)``.
        policy: The approval policy (default safe ``ask``).
        steering: Optional steering profile (forwarded to the preview).

    Returns:
        ``{"flow_id","plan","approved": bool,"executed": bool,"decision","result"}``.
        ``result`` is ``None`` when denied (the flow never ran).
    """
    params = dict(params or {})
    plan = preview_flow_plan(flow_id=flow_id, params=params, steering=steering)

    decision = request_approval(
        action=flow_id,
        cost_estimate=float(plan["estimated_cost"]),
        impact="irreversible" if plan["irreversible"] else "spend",
        policy=policy,
    )
    approved = decision["decision"] == "approved"

    if not approved:
        return {
            "flow_id": flow_id,
            "plan": plan,
            "approved": False,
            "executed": False,
            "decision": decision,
            "result": None,
        }

    result = executor(**params)
    return {
        "flow_id": flow_id,
        "plan": plan,
        "approved": True,
        "executed": True,
        "decision": decision,
        "result": result,
    }
