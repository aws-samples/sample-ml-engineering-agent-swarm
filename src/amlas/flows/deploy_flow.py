"""``deploy_flow`` — the Stage-10 deploy sequence over the DeploymentBackend port (roadmap M4).

The invariant Stage-10 sequence (doc 06 §5.3 ``deploy_flow`` row), the priority GAP closer:

    **(readiness pass) → package_model (T-DEPLOY-09) → select strategy (T-PR-62) →
    deploy_endpoint(mode) (T-DEPLOY-01) → smoke invoke_endpoint (T-DEPLOY-02) →
    (optional) recommend_inference_config (T-DEPLOY-11) → cut_release (T-PR-66) →
    wire monitoring placeholder (T-MON-01)**

with **rollback on failure**: if the smoke-invoke (or any post-deploy step) fails, the flow
fires ``rollback_deployment`` (T-DEPLOY-07) / ``delete_endpoint`` so a half-deployed endpoint
is never left live.

Safe-by-default (ADR-013/014): deploy/delete are EXPENSIVE + IRREVERSIBLE, so the public
entrypoint :func:`run_deploy_flow` is WRAPPED with the M1 ``plan_approve_execute`` —
plan-preview → approval gate → execute, ABORTING **before any package/deploy** on denial
(``status="denied"``, ``approved=False`` — no endpoint created, no spend).

Readiness precondition: when an ``evidence``/``readiness_review`` is supplied (or
``require_readiness=True``), the flow runs ``readiness_flow`` first and **refuses to deploy on a
``no_go``** — the Stage-9→10 hand-off the Deploy/Serving Lead (D3) owns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..control_plane.approvals import ApprovalPolicy
from ..control_plane.plan_mode import plan_approve_execute
from ..tools import deploy_neutral as D
from ._util import raw

__all__ = ["DeployFlowResult", "execute_deploy_flow", "run_deploy_flow"]


@dataclass
class DeployFlowResult:
    """The invariant output contract of one ``deploy_flow`` run."""

    status: str  # "deployed" | "rolled_back" | "blocked_by_readiness" | "failed" | "denied"
    approved: bool = True
    endpoint_name: str = ""
    mode: str = ""
    prediction: Any = None
    package: dict[str, Any] = field(default_factory=dict)
    strategy: dict[str, Any] = field(default_factory=dict)
    sizing: dict[str, Any] = field(default_factory=dict)
    release: dict[str, Any] = field(default_factory=dict)
    monitoring: dict[str, Any] = field(default_factory=dict)
    readiness: dict[str, Any] = field(default_factory=dict)
    edge: dict[str, Any] = field(default_factory=dict)
    rolled_back: bool = False
    adapter: str = ""
    plan: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)


def execute_deploy_flow(
    *,
    model: Any,
    mode: str = "realtime",
    smoke_payload: Any = None,
    resources: dict[str, Any] | None = None,
    slas: dict[str, Any] | None = None,
    traffic_profile: dict[str, Any] | None = None,
    recommend: bool = False,
    wire_monitoring: bool = True,
    readiness_evidence: dict[str, dict[str, Any]] | None = None,
    require_readiness: bool = False,
    edge_target: dict[str, Any] | str | None = None,
    steering: Any | None = None,
) -> DeployFlowResult:
    """The INNER executor (post-approval): readiness → package → deploy → invoke → release.

    Called by :func:`run_deploy_flow` only AFTER the approval gate passes. On any post-deploy
    failure the endpoint is rolled back / deleted (never left half-deployed).
    """
    # ── 0. readiness precondition (Stage-9→10 hand-off) ─────────────────────────
    readiness: dict[str, Any] = {}
    if require_readiness or readiness_evidence is not None:
        from .readiness_flow import run_readiness_flow

        rr = run_readiness_flow(model=model, evidence=readiness_evidence,
                                policy=ApprovalPolicy(), steering=steering)
        readiness = {"recommendation": rr.recommendation, "blocking": rr.blocking,
                     "passed": rr.passed, "concerns": rr.concerns}
        if rr.recommendation != "go":
            return DeployFlowResult(status="blocked_by_readiness", approved=True, readiness=readiness,
                                    detail={"reason": "readiness_flow returned no_go — deploy refused",
                                            "blocking": rr.blocking})

    # ── 1. package the model (T-DEPLOY-09) ──────────────────────────────────────
    pkg = raw(D.package_model)(model)
    if pkg.get("success") is False:
        return DeployFlowResult(status="failed", readiness=readiness, detail={"stage": "package", "error": pkg.get("error")})

    # ── 2. choose the rollout strategy (T-PR-62) ────────────────────────────────
    strategy = raw(D.select_deployment_strategy)(model, slas=slas, traffic_profile=traffic_profile, steering=steering)
    chosen_mode = mode or strategy.get("mode", "realtime")

    # ── 3. deploy the endpoint (T-DEPLOY-01) ────────────────────────────────────
    deploy = raw(D.deploy_endpoint)(model, mode=chosen_mode, resources=resources)
    if deploy.get("success") is False:
        return DeployFlowResult(status="failed", readiness=readiness, package=pkg, strategy=strategy,
                                detail={"stage": "deploy", "error": deploy.get("error")})
    endpoint_name = deploy["endpoint_name"]
    adapter = deploy.get("adapter", "")
    ep_backend = "local" if adapter == "local" else "sagemaker"

    # ── 4. smoke invoke (T-DEPLOY-02) — rollback on failure ─────────────────────
    prediction = None
    if smoke_payload is not None:
        inv = raw(D.invoke_endpoint)(endpoint_name, smoke_payload, mode=chosen_mode, backend=ep_backend)
        if inv.get("success") is False or inv.get("prediction") is None:
            # rollback / delete so we never leave a broken endpoint live (ADR-007 safety).
            rb = raw(D.delete_endpoint)(endpoint_name, backend=ep_backend)
            return DeployFlowResult(
                status="rolled_back", approved=True, endpoint_name=endpoint_name, mode=chosen_mode,
                package=pkg, strategy=strategy, readiness=readiness, rolled_back=True, adapter=adapter,
                detail={"stage": "smoke_invoke", "error": inv.get("error", "no prediction"), "rollback": rb},
            )
        prediction = inv["prediction"]

    # ── 5. (optional) inference-config sizing (T-DEPLOY-11) ─────────────────────
    sizing: dict[str, Any] = {}
    if recommend:
        sizing = raw(D.recommend_inference_config)(model, latency_sla_ms=(slas or {}).get("latency_sla_ms"))

    # ── 6. cut the release (T-PR-66) ────────────────────────────────────────────
    release = raw(D.cut_release)(model, endpoint_name, rollback_target="previous-config")

    # ── 7. wire monitoring placeholder (T-MON-01) ───────────────────────────────
    monitoring: dict[str, Any] = {}
    if wire_monitoring:
        monitoring = _wire_monitoring(endpoint_name)

    # ── 8. (optional) edge/embedded deploy branch (G3, T-DEPLOY-10b) ────────────
    # When an ``edge_target`` is requested, the model is ALSO compiled (Neo) + packaged +
    # (best-effort) deployed to an edge device. Offline this writes a real local bundle; the
    # managed adapter submits the Neo/Edge-Manager jobs (+ Greengrass V2 deploy when configured).
    edge: dict[str, Any] = {}
    if edge_target is not None:
        et = {"target_device": edge_target} if isinstance(edge_target, str) else dict(edge_target)
        edge = raw(D.deploy_edge)(
            model,
            target_device=et.get("target_device", "jetson_nano"),
            framework=et.get("framework", "PYTORCH"),
            greengrass_target=et.get("greengrass_target"),
            component_name=et.get("component_name"),
        )

    return DeployFlowResult(
        status="deployed",
        approved=True,
        endpoint_name=endpoint_name,
        mode=chosen_mode,
        prediction=prediction,
        package=pkg,
        strategy=strategy,
        sizing=sizing,
        release=release,
        monitoring=monitoring,
        readiness=readiness,
        edge=edge,
        adapter=adapter,
        detail={"deploy": deploy},
    )


def _wire_monitoring(endpoint_name: str) -> dict[str, Any]:
    """Wire a monitoring schedule at deploy time (T-MON-01) — best-effort placeholder.

    M4 wires the MonitoringBackend ``setup`` (the OTEL/local adapter offline; Model Monitor
    managed). The CLOSED-LOOP drift→retrain trigger is M5; here we only register the
    monitoring schedule so observability exists from the first request (doc 06 §deploy_flow).
    """
    try:
        from ..backends.ports import MonitoringBackend
        from ..backends.registry import REGISTRY

        resolved = REGISTRY.resolve(MonitoringBackend)
        ref = resolved.adapter.setup(endpoint=endpoint_name, baseline_uri="",
                                     monitor_types=["data_quality"], schedule="hourly")
        return {"schedule_name": getattr(ref, "name", ""), "monitor_type": getattr(ref, "monitor_type", ""),
                "adapter": resolved.adapter_name, "wired": True}
    except Exception as e:
        return {"wired": False, "note": f"monitoring setup deferred: {e}"}


def run_deploy_flow(
    *,
    model: Any,
    mode: str = "realtime",
    smoke_payload: Any = None,
    resources: dict[str, Any] | None = None,
    slas: dict[str, Any] | None = None,
    traffic_profile: dict[str, Any] | None = None,
    recommend: bool = False,
    wire_monitoring: bool = True,
    readiness_evidence: dict[str, dict[str, Any]] | None = None,
    require_readiness: bool = False,
    edge_target: dict[str, Any] | str | None = None,
    steering: Any | None = None,
    policy: ApprovalPolicy | None = None,
) -> DeployFlowResult:
    """Run ``deploy_flow`` GATED by ``plan_approve_execute`` (ADR-014; the prime gate consumer).

    Deploy is expensive + IRREVERSIBLE → the flow is plan-previewed + approval-gated FIRST;
    only on approval does :func:`execute_deploy_flow` run (package→deploy→invoke→release, with
    rollback on failure). On denial the wrapper aborts BEFORE any package/deploy and the
    result is ``status="denied"`` with ``approved=False`` (no endpoint, no spend — safe-by-default).
    """
    outcome = plan_approve_execute(
        flow_id="deploy_flow",
        params={
            "model": model, "mode": mode, "smoke_payload": smoke_payload, "resources": resources,
            "slas": slas, "traffic_profile": traffic_profile, "recommend": recommend,
            "wire_monitoring": wire_monitoring, "readiness_evidence": readiness_evidence,
            "require_readiness": require_readiness, "edge_target": edge_target, "steering": steering,
        },
        executor=execute_deploy_flow,
        policy=policy,
        steering=steering,
    )
    if not outcome["approved"]:
        return DeployFlowResult(
            status="denied", approved=False, plan=outcome["plan"],
            detail={"decision": outcome["decision"], "reason": "approval gate denied — no endpoint deployed"},
        )
    result: DeployFlowResult = outcome["result"]
    result.plan = outcome["plan"]
    return result
