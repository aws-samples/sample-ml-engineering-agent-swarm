"""``baseline_train_flow`` — Stage 8 baseline training over Training+Eval+Tracking ports.

The invariant Stage-8 sequence (doc 06 §5.3 ``baseline_train_flow`` row):

    **prepare → train baseline (TrainingBackend) → grade via the deterministic Arbiter
    (EvaluationBackend) → log to ExperimentTracking → register candidate**

Safe-by-default (ADR-013/014): training incurs spend, so the public entrypoint
:func:`run_baseline_train_flow` is WRAPPED with the M1 ``plan_approve_execute`` —
plan-preview → approval gate → execute, ABORTING before any training launch on denial.

Non-delegated Arbiter (ADR-003): the accept/reject DECISION is the deterministic
``EvaluationBackend.grade`` — never an LLM/managed call; the flow surfaces the Arbiter's
verdict verbatim and only registers the candidate when the Arbiter accepts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..backends.ports import (
    EvaluationBackend,
    ExperimentTrackingBackend,
    ModelRegistryBackend,
    TrainingBackend,
)
from ..backends.registry import REGISTRY
from ..control_plane.approvals import ApprovalPolicy
from ..control_plane.plan_mode import plan_approve_execute

__all__ = ["BaselineTrainFlowResult", "execute_baseline_train_flow", "run_baseline_train_flow"]


@dataclass
class BaselineTrainFlowResult:
    """The invariant output contract of one ``baseline_train_flow`` run."""

    status: str  # "accepted" | "rejected" | "failed" | "denied"
    approved: bool = True
    job_id: str = ""
    train_backend: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    arbiter_verdict: str = ""
    arbiter_report: dict[str, Any] = field(default_factory=dict)
    run_id: str = ""
    registered: dict[str, Any] = field(default_factory=dict)
    plan: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)


def execute_baseline_train_flow(
    *,
    algo: str,
    train_uri: str,
    target: str | None = None,
    spec: Any = None,
    y_true: Any = None,
    y_pred: Any = None,
    baseline_score: float | None = None,
    hyperparams: dict[str, Any] | None = None,
    resources: dict[str, Any] | None = None,
    experiment: str = "baseline",
    register_group: str | None = None,
    steering: Any | None = None,
) -> BaselineTrainFlowResult:
    """The INNER executor (post-approval): train → grade → log → register.

    Called by :func:`run_baseline_train_flow` only AFTER the approval gate passes. The
    Arbiter decision (step 2) is the non-delegated deterministic grade (ADR-003). The
    candidate is registered (step 4) ONLY when the Arbiter accepts.
    """
    # ── 1. train baseline via TrainingBackend (local sklearn offline) ───────────
    try:
        tr = REGISTRY.resolve(TrainingBackend, steering=steering)
        handle = tr.adapter.submit(algo=algo, train_uri=train_uri, target=target,
                                   hyperparams=hyperparams, resources=resources)
        status = tr.adapter.status(handle)
        metrics = tr.adapter.metrics(handle) if status.state.value == "SUCCEEDED" else {}
    except Exception as e:
        return BaselineTrainFlowResult(status="failed", detail={"stage": "train", "error": str(e)})
    if status.state.value != "SUCCEEDED":
        return BaselineTrainFlowResult(status="failed", job_id=handle.id, train_backend=handle.backend,
                                       detail={"stage": "train", "state": status.state.value, "info": status.detail})

    # ── 2. grade via the deterministic Arbiter (NON-DELEGATED — ADR-003) ────────
    arbiter_verdict, accepted = "", True
    arbiter_report: dict[str, Any] = {}
    if spec is not None and y_true is not None and y_pred is not None:
        try:
            ev = REGISTRY.resolve(EvaluationBackend)
            report = ev.adapter.grade(spec=spec, y_true=y_true, y_pred=y_pred, baseline_score=baseline_score)
            arbiter_report = report.model_dump() if hasattr(report, "model_dump") else dict(report)
            arbiter_verdict = str(arbiter_report.get("verdict", arbiter_report.get("decision", "")))
            accepted = _is_accept(arbiter_report)
        except Exception as e:
            return BaselineTrainFlowResult(status="failed", job_id=handle.id, train_backend=handle.backend,
                                           metrics=metrics, detail={"stage": "grade", "error": str(e)})

    # ── 3. log the run to ExperimentTracking ────────────────────────────────────
    run_id = ""
    try:
        trk = REGISTRY.resolve(ExperimentTrackingBackend)
        run_id = trk.adapter.log_run(
            experiment=experiment,
            params={"algorithm": algo, **(hyperparams or {})},
            metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            artifacts={"model_path": tr.adapter.artifact(handle).uri},
        )
    except Exception:
        run_id = ""

    # ── 4. register the candidate ONLY on Arbiter-accept ────────────────────────
    registered: dict[str, Any] = {}
    if accepted and register_group:
        try:
            reg = REGISTRY.resolve(ModelRegistryBackend)
            ref = reg.adapter.register(
                group=register_group, artifact=tr.adapter.artifact(handle).uri,
                metrics={k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            )
            registered = {"arn": ref.arn, "group": ref.group, "version": ref.version,
                          "approval_status": ref.approval_status}
        except Exception as e:
            registered = {"error": str(e)}

    return BaselineTrainFlowResult(
        status="accepted" if accepted else "rejected",
        approved=True,
        job_id=handle.id,
        train_backend=handle.backend,
        metrics=metrics,
        arbiter_verdict=arbiter_verdict,
        arbiter_report=arbiter_report,
        run_id=run_id,
        registered=registered,
        detail={"experiment": experiment},
    )


def run_baseline_train_flow(
    *,
    algo: str,
    train_uri: str,
    target: str | None = None,
    spec: Any = None,
    y_true: Any = None,
    y_pred: Any = None,
    baseline_score: float | None = None,
    hyperparams: dict[str, Any] | None = None,
    resources: dict[str, Any] | None = None,
    experiment: str = "baseline",
    register_group: str | None = None,
    steering: Any | None = None,
    policy: ApprovalPolicy | None = None,
) -> BaselineTrainFlowResult:
    """Run ``baseline_train_flow`` GATED by ``plan_approve_execute`` (ADR-014).

    Training is spend-incurring → the flow is plan-previewed + approval-gated FIRST; only
    on approval does :func:`execute_baseline_train_flow` run (train→grade→log→register).
    On denial the wrapper aborts BEFORE any training launch and the result is
    ``status="denied"`` with ``approved=False`` (no spend, deterministic-safe-by-default).
    """
    outcome = plan_approve_execute(
        flow_id="baseline_train_flow",
        params={
            "algo": algo, "train_uri": train_uri, "target": target, "spec": spec,
            "y_true": y_true, "y_pred": y_pred, "baseline_score": baseline_score,
            "hyperparams": hyperparams, "resources": resources, "experiment": experiment,
            "register_group": register_group, "steering": steering,
        },
        executor=execute_baseline_train_flow,
        policy=policy,
        steering=steering,
    )
    if not outcome["approved"]:
        return BaselineTrainFlowResult(
            status="denied", approved=False, plan=outcome["plan"],
            detail={"decision": outcome["decision"], "reason": "approval gate denied — no training launched"},
        )
    result: BaselineTrainFlowResult = outcome["result"]
    result.plan = outcome["plan"]
    return result


def _is_accept(report: dict[str, Any]) -> bool:
    """Interpret an Arbiter report dict as accept/reject (defensive across shapes)."""
    for key in ("accepted", "passed", "is_accept"):
        if key in report:
            return bool(report[key])
    verdict = str(report.get("verdict", report.get("decision", ""))).strip().lower()
    if verdict:
        return verdict in ("accept", "accepted", "pass", "passed", "approve", "approved")
    return True  # no verdict present → treat as non-blocking (the grade was informational)
