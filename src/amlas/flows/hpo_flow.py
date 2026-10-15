"""``hpo_flow`` — Stage 8 hyperparameter optimization over Tuning+Eval+Tracking ports.

The invariant Stage-8 sequence (doc 06 §5.3 ``hpo_flow`` row):

    **define search space → run HPO (TuningBackend: Optuna local / AMT managed) →
    evaluate best → log to ExperimentTracking → build comparison table**

Safe-by-default (ADR-013/014): HPO incurs spend (managed AMT especially), so the public
entrypoint :func:`run_hpo_flow` is WRAPPED with the M1 ``plan_approve_execute`` —
plan-preview → approval gate → execute, ABORTING before any HPO launch on denial.

The compare step ranks the logged runs via ``ExperimentTrackingBackend`` (the EXISTING
leaderboard comparison IP); the Arbiter remains the non-delegated decision when a
held-out grade is supplied (ADR-003).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..backends.ports import ExperimentTrackingBackend, TuningBackend
from ..backends.registry import REGISTRY

# hpo_flow shares the baseline_train_flow plan template disposition (spend-incurring) —
# register a dedicated template entry so the preview names the right steps.
from ..control_plane import plan_mode as _plan_mode
from ..control_plane.approvals import ApprovalPolicy
from ..control_plane.plan_mode import plan_approve_execute

__all__ = ["HpoFlowResult", "execute_hpo_flow", "run_hpo_flow"]


# Register an hpo_flow plan template (spend-incurring; mirrors the AMT step sequence).
_plan_mode._FLOW_TEMPLATES.setdefault(
    "hpo_flow",
    {
        "steps": [
            {"step": "define_search_space", "tool": "define_ranges"},
            {"step": "run_hpo", "port": "TuningBackend", "tool": "run_hpo"},
            {"step": "evaluate_best", "port": "EvaluationBackend", "tool": "grade"},
            {"step": "log_and_compare", "port": "ExperimentTrackingBackend", "tool": "build_model_comparison_table"},
        ],
        "estimated_cost": 0.80,
        "spend_incurring": True,
        "irreversible": False,
    },
)


@dataclass
class HpoFlowResult:
    """The invariant output contract of one ``hpo_flow`` run."""

    status: str  # "completed" | "failed" | "denied"
    approved: bool = True
    backend: str = ""
    best_params: dict[str, Any] = field(default_factory=dict)
    best_score: float | None = None
    n_trials: int = 0
    objective: str = ""
    direction: str = ""
    run_id: str = ""
    comparison_table: list[dict[str, Any]] = field(default_factory=list)
    plan: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)


def execute_hpo_flow(
    *,
    base_training: dict[str, Any],
    ranges: dict[str, Any],
    objective: str,
    strategy: str = "bayesian",
    max_jobs: int = 8,
    max_parallel: int = 1,
    warm_start_from: str | None = None,
    experiment: str = "hpo",
    steering: Any | None = None,
    scale: int | str | None = None,
) -> HpoFlowResult:
    """The INNER executor (post-approval): run HPO → log best → build comparison.

    Called by :func:`run_hpo_flow` only AFTER the approval gate passes.
    """
    # ── 1. run HPO via TuningBackend (Optuna local offline / AMT managed) ───────
    try:
        tu = REGISTRY.resolve(TuningBackend, steering=steering, scale=scale)
        out = tu.adapter.search(
            base_training=base_training, ranges=ranges, objective=objective,
            strategy=strategy, max_jobs=max_jobs, max_parallel=max_parallel,
            warm_start_from=warm_start_from,
        )
    except Exception as e:
        return HpoFlowResult(status="failed", detail={"stage": "hpo", "error": str(e)})
    if not out.get("success", True) and not out.get("best_params"):
        return HpoFlowResult(status="failed", backend=tu.adapter_name,
                             detail={"stage": "hpo", "result": out})

    best_params = dict(out.get("best_params", {}))
    best_score = out.get("best_score")

    # ── 2. log the best config to ExperimentTracking ────────────────────────────
    run_id = ""
    try:
        trk = REGISTRY.resolve(ExperimentTrackingBackend)
        metrics = {objective: float(best_score)} if isinstance(best_score, (int, float)) else {}
        run_id = trk.adapter.log_run(
            experiment=experiment,
            params={"algorithm": base_training.get("algo", base_training.get("algorithm", "hpo")), **best_params},
            metrics=metrics,
        )
    except Exception:
        run_id = ""

    # ── 3. build the comparison table (ranked runs) ─────────────────────────────
    comparison: list[dict[str, Any]] = []
    try:
        trk = REGISTRY.resolve(ExperimentTrackingBackend)
        if hasattr(trk.adapter, "build_comparison_table"):
            direction = out.get("direction", "maximize")
            comparison = trk.adapter.build_comparison_table(experiment=experiment, direction=direction)
    except Exception:
        comparison = []

    return HpoFlowResult(
        status="completed",
        approved=True,
        backend=tu.adapter_name,
        best_params=best_params,
        best_score=best_score,
        n_trials=int(out.get("n_trials", 0)),
        objective=objective,
        direction=str(out.get("direction", "")),
        run_id=run_id,
        comparison_table=comparison,
        detail={"experiment": experiment, "strategy": out.get("strategy", strategy)},
    )


def run_hpo_flow(
    *,
    base_training: dict[str, Any],
    ranges: dict[str, Any],
    objective: str,
    strategy: str = "bayesian",
    max_jobs: int = 8,
    max_parallel: int = 1,
    warm_start_from: str | None = None,
    experiment: str = "hpo",
    steering: Any | None = None,
    scale: int | str | None = None,
    policy: ApprovalPolicy | None = None,
) -> HpoFlowResult:
    """Run ``hpo_flow`` GATED by ``plan_approve_execute`` (ADR-014).

    HPO is spend-incurring → the flow is plan-previewed + approval-gated FIRST; only on
    approval does :func:`execute_hpo_flow` run. On denial the wrapper aborts BEFORE any
    HPO launch and returns ``status="denied"`` with ``approved=False`` (no spend).
    """
    outcome = plan_approve_execute(
        flow_id="hpo_flow",
        params={
            "base_training": base_training, "ranges": ranges, "objective": objective,
            "strategy": strategy, "max_jobs": max_jobs, "max_parallel": max_parallel,
            "warm_start_from": warm_start_from, "experiment": experiment,
            "steering": steering, "scale": scale,
        },
        executor=execute_hpo_flow,
        policy=policy,
        steering=steering,
    )
    if not outcome["approved"]:
        return HpoFlowResult(
            status="denied", approved=False, plan=outcome["plan"],
            detail={"decision": outcome["decision"], "reason": "approval gate denied — no HPO launched"},
        )
    result: HpoFlowResult = outcome["result"]
    result.plan = outcome["plan"]
    return result
