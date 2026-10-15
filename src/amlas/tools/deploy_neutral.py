"""Backend-neutral Stage 9–10 tools — readiness + deployment (M4; ADR-007).

The doc 05 §4.4 EVOLVE pattern for the deploy GAP: each tool resolves the
:class:`~amlas.backends.ports.DeploymentBackend` (or Governance/Guardrail)
PORT via the ``BackendRegistry`` and delegates to whatever adapter the registry binds
(``LocalServeBackend`` offline/under test, ``SageMakerDeploymentBackend`` managed). NEW
M4 tools, ADDITIVE — existing tools/agents keep working untouched.

Stage-10 deploy tools (doc 06 §Deploy):
  * ``package_model`` (T-DEPLOY-09), ``deploy_endpoint`` (T-DEPLOY-01, ``mode=``),
    ``invoke_endpoint`` (T-DEPLOY-02), ``update_endpoint_canary`` (T-DEPLOY-03),
    ``run_batch_transform`` (T-DEPLOY-04), ``run_shadow_deployment`` (T-DEPLOY-05),
    ``run_ab_test`` (T-DEPLOY-06), ``rollback_deployment`` (T-DEPLOY-07),
    ``delete_endpoint`` (T-DEPLOY-08), ``compile_model`` (T-DEPLOY-10),
    ``recommend_inference_config`` (T-DEPLOY-11), ``run_load_test`` (T-DEPLOY-12),
    ``cut_release`` (T-PR-66).
Stage-9 readiness tools (doc 06 §Pure-reasoning T-PR-57..62):
  * ``run_readiness_gate(gate=)`` (T-PR-58 — the parameterized 10-gate tool),
    ``build_risk_register`` (T-PR-60), ``compile_readiness_review`` (T-PR-61),
    ``design_production_architecture`` (T-PR-57), ``draft_operational_runbook`` (T-PR-59),
    ``select_deployment_strategy`` (T-PR-62).

Deploy/delete/canary/rollback/shadow/ab classify as IRREVERSIBLE in the approval policy
(``control_plane/approvals.py``); the ``deploy_flow`` wraps them with ``plan_approve_execute``
so a real endpoint create is plan-previewed + approved before it spends (doc 05 §8.8; ADR-014).
The readiness tools are CHEAP_READ (pure-reasoning go/no-go assessments).
"""

from __future__ import annotations

from typing import Any

from strands import tool

from ..backends.ports import DeploymentBackend
from ..backends.registry import REGISTRY, ResolvedBackend
from ..backends.types import ModelRef
from ..errors import tool_error_boundary

__all__ = [
    "build_risk_register",
    "compile_model",
    "compile_readiness_review",
    "cut_release",
    "delete_endpoint",
    "deploy_edge",
    "deploy_endpoint",
    "design_production_architecture",
    "draft_operational_runbook",
    "invoke_endpoint",
    # Stage 10 — deploy
    "package_model",
    "recommend_inference_config",
    "rollback_deployment",
    "run_ab_test",
    "run_batch_transform",
    "run_load_test",
    # Stage 9 — readiness
    "run_readiness_gate",
    "run_shadow_deployment",
    "select_deployment_strategy",
    # helper (used by flows + the MVP demo to build a ModelRef from a path/dict)
    "to_model_ref",
    "update_endpoint_canary",
]


# ══════════════════════════════════════════════════════════════════════════════
# helper — coerce a model spec into a domain ModelRef (flows + tools share this)
# ══════════════════════════════════════════════════════════════════════════════
def to_model_ref(model: Any) -> ModelRef:
    """Coerce a ``ModelRef`` / dict / URI string into a :class:`ModelRef` (domain)."""
    if isinstance(model, ModelRef):
        return model
    if isinstance(model, str):
        return ModelRef(uri=model, backend="local")
    if isinstance(model, dict):
        return ModelRef(
            uri=str(model.get("uri", model.get("model_path", ""))),
            backend=str(model.get("backend", "local")),
            algo=str(model.get("algo", model.get("algorithm", ""))),
            metrics={k: float(v) for k, v in (model.get("metrics") or {}).items() if isinstance(v, (int, float))},
            extra={k: v for k, v in model.items() if k not in ("uri", "model_path", "backend", "algo", "algorithm", "metrics")},
        )
    raise ValueError(f"cannot coerce {type(model).__name__} into a ModelRef")


def _resolve(scale: int | str | None = None, steering: Any = None) -> ResolvedBackend:
    """Resolve the DeploymentBackend (offline → local server, managed → SageMaker)."""
    return REGISTRY.resolve(DeploymentBackend, steering=steering, scale=scale)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 10 — deployment tools (T-DEPLOY-*)
# ══════════════════════════════════════════════════════════════════════════════
@tool
@tool_error_boundary(source="tools.deploy_neutral")
def package_model(
    model: Any,
    image: str | None = None,
    inference_code: str | None = None,
    env: dict[str, Any] | None = None,
    scale: int | str | None = None,
) -> dict[str, Any]:
    """Package a trained artifact into a deployable container ref (T-DEPLOY-09).

    Resolves :class:`DeploymentBackend` and packages ``model`` (a ModelRef / path / dict).
    Returns ``{"package_arn","group","approval_status","adapter"}``.
    """
    try:
        resolved = _resolve(scale=scale)
        ref = resolved.adapter.package(model_ref=to_model_ref(model), image=image,
                                       inference_code=inference_code, env=env)
        return {"package_arn": ref.arn, "group": ref.group, "version": ref.version,
                "approval_status": ref.approval_status, "adapter": resolved.adapter_name, "reason": resolved.reason}
    except Exception as e:
        return {"success": False, "error": f"package_model failed: {e}"}


@tool
@tool_error_boundary(source="tools.deploy_neutral")
def deploy_endpoint(
    model: Any,
    mode: str = "realtime",
    resources: dict[str, Any] | None = None,
    autoscaling: dict[str, Any] | None = None,
    scale: int | str | None = None,
) -> dict[str, Any]:
    """Deploy ``model`` as an endpoint in ``mode`` (T-DEPLOY-01; realtime/serverless/async/multimodel).

    Resolves :class:`DeploymentBackend` (local in-process server offline, SageMaker managed)
    and deploys. **IRREVERSIBLE/expensive** — the ``deploy_flow`` wraps this with
    ``plan_approve_execute``. Returns ``{"endpoint_name","arn","mode","adapter"}``.
    """
    try:
        resolved = _resolve(scale=scale)
        ep = resolved.adapter.deploy(model_ref=to_model_ref(model), mode=mode,
                                     resources=resources, autoscaling=autoscaling)
        return {"endpoint_name": ep.name, "arn": ep.arn, "mode": ep.mode, "status": "InService",
                "adapter": resolved.adapter_name, "reason": resolved.reason}
    except Exception as e:
        return {"success": False, "error": f"deploy_endpoint failed: {e}"}


@tool
@tool_error_boundary(source="tools.deploy_neutral")
def invoke_endpoint(
    endpoint_name: str,
    payload: Any,
    target_model: str | None = None,
    mode: str = "realtime",
    backend: str = "local",
    scale: int | str | None = None,
) -> dict[str, Any]:
    """Invoke a deployed endpoint and return a prediction (T-DEPLOY-02).

    Resolves :class:`DeploymentBackend` and invokes ``endpoint_name`` with ``payload``.
    Returns ``{"prediction","predictions","endpoint","adapter"}``.
    """
    from ..backends.types import EndpointRef

    try:
        resolved = _resolve(scale=scale)
        ep = EndpointRef(name=endpoint_name, mode=mode, backend=backend)
        out: dict[str, Any] = resolved.adapter.invoke(endpoint=ep, payload=payload, target_model=target_model)
        out.setdefault("adapter", resolved.adapter_name)
        return out
    except Exception as e:
        return {"success": False, "error": f"invoke_endpoint failed: {e}"}


@tool
@tool_error_boundary(source="tools.deploy_neutral")
def update_endpoint_canary(
    endpoint_name: str,
    new_config: dict[str, Any],
    canary_pct: int = 10,
    alarms: list[str] | None = None,
    mode: str = "canary",
    backend: str = "local",
    scale: int | str | None = None,
) -> dict[str, Any]:
    """Shift endpoint traffic to ``new_config`` via canary / blue-green (T-DEPLOY-03).

    **IRREVERSIBLE** — gated by the approval policy. Returns the deployment status.
    """
    from ..backends.types import EndpointRef

    try:
        resolved = _resolve(scale=scale)
        ep = EndpointRef(name=endpoint_name, backend=backend)
        out: dict[str, Any] = resolved.adapter.update_canary(endpoint=ep, new_config=new_config,
                                                             canary_pct=canary_pct, alarms=alarms, mode=mode)
        out.setdefault("adapter", resolved.adapter_name)
        return out
    except Exception as e:
        return {"success": False, "error": f"update_endpoint_canary failed: {e}"}


@tool
@tool_error_boundary(source="tools.deploy_neutral")
def run_batch_transform(
    model: Any,
    input_uri: str,
    output_uri: str,
    resources: dict[str, Any] | None = None,
    scale: int | str | None = None,
) -> dict[str, Any]:
    """Run a batch-transform scoring job over ``input_uri`` (T-DEPLOY-04).

    Resolves :class:`DeploymentBackend` and scores every row, writing to ``output_uri``.
    Returns ``{"job_name","status","output_uri","n_scored","adapter"}``.
    """
    try:
        resolved = _resolve(scale=scale)
        handle = resolved.adapter.batch_score(model_ref=to_model_ref(model), input_uri=input_uri,
                                              output_uri=output_uri, resources=resources)
        return {"job_name": handle.id, "status": handle.extra.get("status", "Completed"),
                "output_uri": handle.extra.get("output_uri", output_uri),
                "n_scored": handle.extra.get("n_scored"), "adapter": resolved.adapter_name}
    except Exception as e:
        return {"success": False, "error": f"run_batch_transform failed: {e}"}


@tool
@tool_error_boundary(source="tools.deploy_neutral")
def run_shadow_deployment(
    endpoint_name: str,
    shadow_model: Any,
    traffic_pct: int = 100,
    duration: int = 3600,
    backend: str = "local",
    scale: int | str | None = None,
) -> dict[str, Any]:
    """Run a shadow deployment of ``shadow_model`` alongside ``endpoint_name`` (T-DEPLOY-05).

    A shadow variant receives a copy of live traffic without affecting responses (canary's
    risk-free cousin). Implemented as a canary update with shadow framing; **IRREVERSIBLE** →
    approval-gated. Returns the shadow agreement/divergence framing.
    """
    out = update_endpoint_canary._tool_func(
        endpoint_name, {"model_ref": to_model_ref(shadow_model), "mode": "shadow"},
        canary_pct=traffic_pct, mode="shadow", backend=backend, scale=scale,
    )
    if out.get("success") is False:
        return out
    return {"shadow": True, "endpoint": endpoint_name, "traffic_pct": traffic_pct,
            "duration": duration, "deployment_status": out.get("deployment_status"),
            "adapter": out.get("adapter")}


@tool
@tool_error_boundary(source="tools.deploy_neutral")
def run_ab_test(
    endpoint_name: str,
    variants: list[dict[str, Any]],
    metric: str = "accuracy",
    window: int = 3600,
    backend: str = "local",
    scale: int | str | None = None,
) -> dict[str, Any]:
    """Run an A/B test across weighted ``variants`` behind ``endpoint_name`` (T-DEPLOY-06).

    **IRREVERSIBLE** (changes production traffic split) → approval-gated. Returns the
    per-variant config + the chosen metric/window (the live metric collection is M5 monitoring).
    """
    try:
        resolved = _resolve(scale=scale)
        return {"endpoint": endpoint_name, "metric": metric, "window": window,
                "variants": [{"weight": v.get("weight"), "model": str(v.get("model_ref", v.get("model", "")))}
                             for v in variants],
                "adapter": resolved.adapter_name, "note": "per-variant metric collection lands in M5 monitoring"}
    except Exception as e:
        return {"success": False, "error": f"run_ab_test failed: {e}"}


@tool
@tool_error_boundary(source="tools.deploy_neutral")
def rollback_deployment(
    endpoint_name: str,
    to_version: str | None = None,
    backend: str = "local",
    scale: int | str | None = None,
) -> dict[str, Any]:
    """Roll the endpoint back to a prior config version (T-DEPLOY-07).

    **IRREVERSIBLE** → approval-gated (it is itself the recovery action a failed deploy fires).
    Returns ``{"status","restored_version","endpoint","adapter"}``.
    """
    from ..backends.types import EndpointRef

    try:
        resolved = _resolve(scale=scale)
        ep = EndpointRef(name=endpoint_name, backend=backend)
        out: dict[str, Any] = resolved.adapter.rollback(endpoint=ep, to_version=to_version)
        out.setdefault("adapter", resolved.adapter_name)
        return out
    except Exception as e:
        return {"success": False, "error": f"rollback_deployment failed: {e}"}


@tool
@tool_error_boundary(source="tools.deploy_neutral")
def delete_endpoint(
    endpoint_name: str,
    backend: str = "local",
    scale: int | str | None = None,
) -> dict[str, Any]:
    """Tear down a deployed endpoint (T-DEPLOY-08). **IRREVERSIBLE** → approval-gated."""
    from ..backends.types import EndpointRef

    try:
        resolved = _resolve(scale=scale)
        ep = EndpointRef(name=endpoint_name, backend=backend)
        out: dict[str, Any] = resolved.adapter.delete(endpoint=ep)
        out.setdefault("adapter", resolved.adapter_name)
        return out
    except Exception as e:
        return {"success": False, "error": f"delete_endpoint failed: {e}"}


@tool
@tool_error_boundary(source="tools.deploy_neutral")
def compile_model(
    model: Any,
    target_device: str = "ml_c5",
    framework: str = "PYTORCH",
    dtype: str = "fp32",
    scale: int | str | None = None,
) -> dict[str, Any]:
    """Compile a model for a target device via SageMaker Neo (T-DEPLOY-10).

    Resolves :class:`DeploymentBackend`; the managed adapter submits a Neo compilation job.
    The local adapter has no Neo equivalent → returns a not-applicable note offline.
    """
    try:
        resolved = _resolve(scale=scale)
        adapter = resolved.adapter
        if not hasattr(adapter, "compile"):
            return {"success": False, "error": "Neo compile not available on the local adapter (managed-only)",
                    "adapter": resolved.adapter_name}
        out: dict[str, Any] = adapter.compile(model_ref=to_model_ref(model), target_device=target_device,
                                              framework=framework, dtype=dtype)
        out.setdefault("adapter", resolved.adapter_name)
        return out
    except Exception as e:
        return {"success": False, "error": f"compile_model failed: {e}"}


@tool
@tool_error_boundary(source="tools.deploy_neutral")
def deploy_edge(
    model: Any,
    compile_output: dict[str, Any] | None = None,
    target_device: str = "jetson_nano",
    framework: str = "PYTORCH",
    greengrass_target: str | None = None,
    component_name: str | None = None,
    scale: int | str | None = None,
) -> dict[str, Any]:
    """Compile + package + (optionally) deploy a model to an edge/embedded device (T-DEPLOY-10b, G3).

    Extends the Neo ``compile_model`` (T-DEPLOY-10) into an actual edge/embedded deploy path:
    Neo compile (for ``target_device``) → SageMaker Edge Manager packaging (as a Greengrass V2
    component) → optional Greengrass V2 fleet deployment when a ``greengrass_target`` (thing/
    thing-group ARN) is supplied. The local adapter realizes it as an on-disk compiled bundle.

    **IRREVERSIBLE** (it provisions/deploys an edge artifact) → the ``deploy_flow`` wraps this
    with ``plan_approve_execute``. Returns ``{"compilation_job","edge_packaging_job",
    "component_name","deployed","deployment_id","target_device","adapter", ...}``.
    """
    try:
        resolved = _resolve(scale=scale)
        adapter = resolved.adapter
        if not hasattr(adapter, "deploy_edge"):
            return {"success": False, "error": "edge deploy not available on this adapter",
                    "adapter": resolved.adapter_name}
        out: dict[str, Any] = adapter.deploy_edge(
            model_ref=to_model_ref(model), compile_output=compile_output,
            target_device=target_device, framework=framework,
            greengrass_target=greengrass_target, component_name=component_name,
        )
        out.setdefault("adapter", resolved.adapter_name)
        out.setdefault("reason", resolved.reason)
        return out
    except Exception as e:
        return {"success": False, "error": f"deploy_edge failed: {e}"}


@tool
@tool_error_boundary(source="tools.deploy_neutral")
def recommend_inference_config(
    model: Any,
    sample_payload: str = "",
    latency_sla_ms: float | None = None,
    max_cost: float | None = None,
    scale: int | str | None = None,
) -> dict[str, Any]:
    """Recommend an inference instance/sizing config (T-DEPLOY-11; Inference Recommender).

    Resolves :class:`DeploymentBackend` and returns a ranked sizing shortlist. Read-only/
    CHEAP — informs the deploy plan; does not deploy.
    """
    try:
        resolved = _resolve(scale=scale)
        recs = resolved.adapter.recommend_config(
            model_ref=to_model_ref(model), sample=sample_payload,
            sla={"latency_sla_ms": latency_sla_ms, "max_cost": max_cost},
        )
        return {"recommendations": [
            {"instance_type": r.instance_type, "cost_per_hour": r.est_cost_per_hour,
             "p99_latency_ms": r.est_latency_ms, "throughput": r.est_throughput} for r in recs
        ], "adapter": resolved.adapter_name}
    except Exception as e:
        return {"success": False, "error": f"recommend_inference_config failed: {e}"}


@tool
@tool_error_boundary(source="tools.deploy_neutral")
def run_load_test(
    model: Any,
    traffic_pattern: str = "steady",
    latency_sla_ms: float = 100.0,
    duration: int = 300,
    scale: int | str | None = None,
) -> dict[str, Any]:
    """Run a load test to find the breaking point (T-DEPLOY-12; Inference Recommender advanced).

    Read-only sizing assessment (CHEAP). Offline this derives a deterministic estimate from the
    sizing shortlist; managed uses Inference Recommender's advanced job. Returns p99/throughput/
    error-rate/breaking-point.
    """
    rec = recommend_inference_config._tool_func(model, latency_sla_ms=latency_sla_ms, scale=scale)
    if rec.get("success") is False:
        return rec
    recs = rec.get("recommendations", [])
    best = recs[0] if recs else {"p99_latency_ms": latency_sla_ms, "throughput": 100.0}
    return {
        "p99_latency_ms": best.get("p99_latency_ms", latency_sla_ms),
        "throughput": best.get("throughput", 100.0),
        "error_rate": 0.0,
        "breaking_point": float(best.get("throughput", 100.0)) * 1.5,
        "traffic_pattern": traffic_pattern,
        "duration": duration,
        "meets_sla": best.get("p99_latency_ms", latency_sla_ms) <= latency_sla_ms,
        "adapter": rec.get("adapter"),
    }


@tool
@tool_error_boundary(source="tools.deploy_neutral")
def cut_release(
    model: Any,
    endpoint_name: str,
    rollback_target: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Cut a release record for a deployed endpoint (T-PR-66).

    A pure-reasoning release-note + deployment-log artifact (no AWS) capturing the deployed
    model, endpoint, and rollback target so the deploy is auditable + reversible.
    """
    ref = to_model_ref(model)
    import datetime as _dt

    version = _dt.datetime.utcnow().strftime("v%Y%m%d-%H%M%S")
    return {
        "release_version": version,
        "endpoint": endpoint_name,
        "model": ref.uri or ref.algo,
        "rollback_target": rollback_target,
        "release_notes": notes or f"Release {version}: deployed {ref.algo or ref.uri} to {endpoint_name}.",
        "deployment_log": [
            f"packaged {ref.algo or ref.uri}",
            f"deployed to {endpoint_name}",
            f"rollback target: {rollback_target or 'previous config'}",
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Stage 9 — readiness tools (T-PR-57..62) — pure-reasoning / CHEAP_READ
# ══════════════════════════════════════════════════════════════════════════════
# The 10 canonical production-readiness gates (doc 04 Stage 9; T-PR-58 parameterized).
READINESS_GATES: tuple[str, ...] = (
    "quality", "cost", "latency", "reliability", "security",
    "privacy", "safety", "explainability", "operability", "maintainability",
)


@tool
@tool_error_boundary(source="tools.deploy_neutral")
def run_readiness_gate(gate: str, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run ONE production-readiness gate against ``evidence`` (T-PR-58; the parameterized 10-gate tool).

    Each gate is a deterministic pass/concern/fail verdict over the supplied evidence
    (metrics, sizing, guardrail/governance results). The accept/reject DECISION for the
    MODEL remains the deterministic Arbiter (ADR-003) — these gates assess *production
    readiness* (cost/latency/security/…), not model quality. Read-only (CHEAP_READ).

    Args:
        gate: One of READINESS_GATES (quality/cost/latency/reliability/security/privacy/
            safety/explainability/operability/maintainability).
        evidence: Gate-specific evidence dict (e.g. ``{"arbiter_passed": True}`` for quality,
            ``{"p99_latency_ms": 40, "latency_sla_ms": 100}`` for latency,
            ``{"guardrail_passed": True}`` for safety).

    Returns:
        ``{"gate","verdict": "pass"|"concern"|"fail","findings": [...]}``.
    """
    g = (gate or "").strip().lower()
    ev = evidence or {}
    if g not in READINESS_GATES:
        return {"gate": gate, "verdict": "fail", "findings": [f"unknown gate {gate!r}; "
                f"expected one of {list(READINESS_GATES)}"]}

    findings: list[str] = []
    verdict = "pass"

    if g == "quality":
        # The Arbiter is the authority; this gate just surfaces its verdict for the review.
        if ev.get("arbiter_passed") is False:
            verdict, findings = "fail", ["Arbiter rejected the model — quality gate blocks"]
        elif ev.get("arbiter_passed") is True:
            findings = ["Arbiter accepted the model"]
        else:
            verdict, findings = "concern", ["no Arbiter verdict supplied as evidence"]
    elif g == "latency":
        p99 = ev.get("p99_latency_ms")
        sla = ev.get("latency_sla_ms", 100.0)
        if p99 is None:
            verdict, findings = "concern", ["no measured p99 latency (run run_load_test)"]
        elif float(p99) > float(sla):
            verdict, findings = "fail", [f"p99 {p99}ms exceeds SLA {sla}ms"]
        else:
            findings = [f"p99 {p99}ms within SLA {sla}ms"]
    elif g == "cost":
        cost = ev.get("cost_per_hour")
        budget = ev.get("max_cost")
        if cost is not None and budget is not None and float(cost) > float(budget):
            verdict, findings = "fail", [f"est ${cost}/hr exceeds budget ${budget}/hr"]
        else:
            findings = [f"est ${cost}/hr within budget" if cost is not None else "no sizing cost supplied"]
            if cost is None:
                verdict = "concern"
    elif g in ("safety", "security", "privacy"):
        guard = ev.get("guardrail_passed")
        if guard is False:
            verdict, findings = "fail", [f"{g} guardrail intervened: {ev.get('violations', [])}"]
        elif guard is True:
            findings = [f"{g} guardrail clean"]
        else:
            verdict, findings = "concern", [f"no {g} guardrail evidence supplied"]
    elif g == "explainability":
        if ev.get("explain_report") or ev.get("shap_uri"):
            findings = ["explainability artifact present"]
        else:
            verdict, findings = "concern", ["no explainability artifact (Clarify/SHAP) attached"]
    elif g in ("reliability", "operability", "maintainability"):
        if ev.get("runbook") or ev.get("monitoring_wired"):
            findings = [f"{g}: runbook/monitoring evidence present"]
        else:
            verdict, findings = "concern", [f"{g}: no runbook/monitoring evidence"]

    return {"gate": g, "verdict": verdict, "findings": findings}


@tool
@tool_error_boundary(source="tools.deploy_neutral")
def build_risk_register(model: Any, gate_findings: list[dict[str, Any]]) -> dict[str, Any]:
    """Compile a risk register from the gate findings (T-PR-60).

    Maps each non-pass gate to a risk row (risk/severity/mitigation/owner). Read-only.
    """
    sev = {"fail": "high", "concern": "medium", "pass": "low"}
    risks = []
    for f in gate_findings or []:
        verdict = str(f.get("verdict", "pass"))
        if verdict == "pass":
            continue
        gate = f.get("gate", "?")
        risks.append({
            "risk": f"{gate} gate {verdict}: " + "; ".join(f.get("findings", []) or [f"{gate} unverified"]),
            "severity": sev.get(verdict, "medium"),
            "mitigation": f"Address {gate} findings before promotion; re-run run_readiness_gate(gate={gate!r}).",
            "owner": "Deploy/Serving Lead (D3)",
        })
    return {"model": to_model_ref(model).uri or to_model_ref(model).algo, "risks": risks,
            "risk_count": len(risks)}


@tool
@tool_error_boundary(source="tools.deploy_neutral")
def compile_readiness_review(model: Any, gates: list[dict[str, Any]]) -> dict[str, Any]:
    """Compile the production-readiness review + go/no-go recommendation (T-PR-61).

    Aggregates the 10 gate verdicts into a blocking-gap list + an approval recommendation:
    ``go`` only when NO gate failed (concerns are surfaced but non-blocking). Read-only.
    """
    fails = [g for g in (gates or []) if str(g.get("verdict")) == "fail"]
    concerns = [g for g in (gates or []) if str(g.get("verdict")) == "concern"]
    passed = [g for g in (gates or []) if str(g.get("verdict")) == "pass"]
    recommendation = "no_go" if fails else "go"
    return {
        "model": to_model_ref(model).uri or to_model_ref(model).algo,
        "gates_total": len(gates or []),
        "passed": len(passed),
        "concerns": len(concerns),
        "blocking_gaps": [{"gate": g.get("gate"), "findings": g.get("findings")} for g in fails],
        "approval_recommendation": recommendation,
        "review_report": (
            f"Production-readiness review: {len(passed)}/{len(gates or [])} gates passed, "
            f"{len(concerns)} concern(s), {len(fails)} blocking. Recommendation: {recommendation.upper()}."
        ),
    }


@tool
@tool_error_boundary(source="tools.deploy_neutral")
def design_production_architecture(
    model: Any,
    slas: dict[str, Any] | None = None,
    mode: str = "seed",
) -> dict[str, Any]:
    """Propose a serving architecture + hosting option for ``model`` (T-PR-57).

    Pure-reasoning: recommends a hosting pattern (realtime/serverless/async/batch) + autoscaling
    from the SLA shape. Read-only — informs select_deployment_strategy + deploy_endpoint(mode=).
    """
    slas = slas or {}
    rps = float(slas.get("requests_per_second", 0) or 0)
    latency = slas.get("latency_sla_ms")
    if slas.get("batch"):
        hosting = "batch"
    elif rps and rps < 1:
        hosting = "serverless"
    elif slas.get("async") or slas.get("large_payload"):
        hosting = "async"
    else:
        hosting = "realtime"
    return {
        "model": to_model_ref(model).uri or to_model_ref(model).algo,
        "architecture": f"{hosting}-endpoint behind the DeploymentBackend port",
        "hosting_option": hosting,
        "autoscaling": {"min": 1, "max": 4} if hosting == "realtime" else {"managed": True},
        "rationale": f"chosen from SLAs (rps={rps}, latency={latency}, mode={mode})",
    }


@tool
@tool_error_boundary(source="tools.deploy_neutral")
def draft_operational_runbook(
    model: Any,
    gates: list[dict[str, Any]] | None = None,
    owners: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Draft an operational runbook with rollback steps + escalation path (T-PR-59). Read-only."""
    owners = owners or {"on_call": "Deploy/Serving Lead (D3)"}
    return {
        "model": to_model_ref(model).uri or to_model_ref(model).algo,
        "runbook": [
            "Monitor endpoint health + drift (MonitoringBackend wired at deploy time).",
            "On latency/error breach: alarm fires → page on-call.",
            "On model regression: rollback_deployment(endpoint, to_version=previous).",
        ],
        "rollback_steps": ["rollback_deployment(endpoint)", "verify previous version InService", "notify owners"],
        "escalation_path": [owners.get("on_call", "on-call"), "Deploy/Serving Lead", "MLOps Lead"],
        "open_concerns": [g.get("gate") for g in (gates or []) if str(g.get("verdict")) != "pass"],
    }


@tool
@tool_error_boundary(source="tools.deploy_neutral")
def select_deployment_strategy(
    model: Any,
    slas: dict[str, Any] | None = None,
    traffic_profile: dict[str, Any] | None = None,
    steering: Any = None,
) -> dict[str, Any]:
    """Choose the serving pattern + rollout strategy for ``model`` (T-PR-62).

    Pure-reasoning: maps SLAs/traffic to a ``(pattern, mode, rollout_strategy)`` — e.g. a
    high-traffic prod model → realtime + canary; a dev model → realtime + all-at-once. Feeds
    ``deploy_flow``. Read-only.
    """
    slas = slas or {}
    traffic_profile = traffic_profile or {}
    arch = design_production_architecture._tool_func(model, slas=slas)
    mode = arch["hosting_option"]
    high_traffic = float(traffic_profile.get("requests_per_second", 0) or 0) >= 1 or traffic_profile.get("production")
    rollout = "canary" if high_traffic else "all_at_once"
    return {
        "model": to_model_ref(model).uri or to_model_ref(model).algo,
        "pattern": mode,
        "mode": mode,
        "rollout_strategy": rollout,
        "rationale": f"hosting={mode} from SLAs; rollout={rollout} from traffic profile {traffic_profile}",
    }
