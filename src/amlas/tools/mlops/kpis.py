"""MLOps KPI scorecard (spec sheet / master plan §8 D7, research KB §4).

The transcript's "how do we benchmark the MLOps side / what are the KPIs"
(14:23–14:31). ``compute_mlops_kpis`` rolls a list of run/registry/monitor/deploy
events into the §8 KPI scorecard:

  * pipeline success rate + mean wall-clock,
  * model-registry time-to-register + approval lead time,
  * monitoring drift-detection latency + violations caught + baseline coverage,
  * deploy success % / rollback rate,
  * lineage/governance: % models with complete lineage + model card,
  * Feature Store reuse count + training/serving skew incidents,
  * an overall L1→L2→L3 maturity placement.

Pure Python (no AWS) so it is fast + deterministic; the SageMaker Ops sub-agent /
MLOps Lead feed it the events they collected from the (real AWS/stubbed) reads.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from ...errors import tool_error_boundary

__all__ = ["compute_mlops_kpis", "maturity_level", "mlops_kpi_scorecard"]


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def maturity_level(kpis: dict[str, Any]) -> str:
    """Place the platform on the Google MLOps L0→L3 ladder (research KB §4).

    L2 = CI/CD + registry + monitoring active; L3 adds an automated drift→retrain
    trigger. We score conservatively from observed coverage.
    """
    has_pipeline = kpis.get("pipeline", {}).get("total", 0) > 0
    has_registry = kpis.get("registry", {}).get("registered", 0) > 0
    has_monitor = kpis.get("monitoring", {}).get("schedules", 0) > 0
    auto_retrain = kpis.get("monitoring", {}).get("auto_retrain_enabled", False)
    if has_pipeline and has_registry and has_monitor and auto_retrain:
        return "L3"
    if has_pipeline and has_registry and has_monitor:
        return "L2"
    if has_pipeline or has_registry:
        return "L1"
    return "L0"


def compute_mlops_kpis(events: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute the MLOps KPI scorecard from collected events (master plan §8 D7).

    Args:
        events: A dict with optional keys:
            ``pipeline_runs``: ``[{"status":"Succeeded|Failed","wall_clock_seconds":N}]``
            ``registrations``: ``[{"time_to_register_seconds":N,"approval_lead_seconds":N,
                                    "has_lineage":bool,"has_model_card":bool}]``
            ``monitors``: ``[{"baseline":bool,"drift_latency_seconds":N,
                              "violations_caught":N,"auto_retrain":bool}]``
            ``deployments``: ``[{"status":"deployed|rolled_back"}]``
            ``feature_groups``: ``[{"reuse_count":N,"skew_incidents":N}]``

    Returns:
        A JSON-serializable scorecard dict with per-category KPIs + an
        ``overall`` block (composite health 0..1 + maturity level). Never raises —
        empty/missing inputs yield zeroed KPIs so the Lead can always report.
    """
    ev = dict(events or {})

    # ── Pipelines ────────────────────────────────────────────────────────────
    runs = ev.get("pipeline_runs", []) or []
    succeeded = sum(1 for r in runs if str(r.get("status", "")).lower() in ("succeeded", "success"))
    wall = [float(r.get("wall_clock_seconds", 0)) for r in runs if r.get("wall_clock_seconds") is not None]
    pipeline = {
        "total": len(runs),
        "succeeded": succeeded,
        "success_rate": round(_safe_div(succeeded, len(runs)), 4),
        "mean_wall_clock_seconds": round(_mean(wall), 2),
    }

    # ── Registry ─────────────────────────────────────────────────────────────
    regs = ev.get("registrations", []) or []
    ttr = [float(r.get("time_to_register_seconds", 0)) for r in regs if r.get("time_to_register_seconds") is not None]
    lead = [float(r.get("approval_lead_seconds", 0)) for r in regs if r.get("approval_lead_seconds") is not None]
    with_lineage = sum(1 for r in regs if r.get("has_lineage"))
    with_card = sum(1 for r in regs if r.get("has_model_card"))
    registry = {
        "registered": len(regs),
        "mean_time_to_register_seconds": round(_mean(ttr), 2),
        "mean_approval_lead_seconds": round(_mean(lead), 2),
    }

    # ── Monitoring ───────────────────────────────────────────────────────────
    mons = ev.get("monitors", []) or []
    with_baseline = sum(1 for m in mons if m.get("baseline"))
    drift_lat = [float(m.get("drift_latency_seconds", 0)) for m in mons if m.get("drift_latency_seconds") is not None]
    violations = sum(int(m.get("violations_caught", 0)) for m in mons)
    auto_retrain = any(m.get("auto_retrain") for m in mons)
    monitoring = {
        "schedules": len(mons),
        "baseline_coverage": round(_safe_div(with_baseline, len(mons)), 4),
        "mean_drift_detection_latency_seconds": round(_mean(drift_lat), 2),
        "violations_caught": violations,
        "auto_retrain_enabled": auto_retrain,
    }

    # ── Deployment ───────────────────────────────────────────────────────────
    deps = ev.get("deployments", []) or []
    deployed = sum(1 for d in deps if str(d.get("status", "")).lower() == "deployed")
    rolled_back = sum(1 for d in deps if str(d.get("status", "")).lower() == "rolled_back")
    deployment = {
        "total": len(deps),
        "deploy_success_rate": round(_safe_div(deployed, len(deps)), 4),
        "rollback_rate": round(_safe_div(rolled_back, len(deps)), 4),
    }

    # ── Lineage / governance ─────────────────────────────────────────────────
    governance = {
        "models_with_lineage_pct": round(_safe_div(with_lineage, len(regs)), 4),
        "models_with_model_card_pct": round(_safe_div(with_card, len(regs)), 4),
    }

    # ── Feature Store ────────────────────────────────────────────────────────
    fgs = ev.get("feature_groups", []) or []
    reuse = sum(int(f.get("reuse_count", 0)) for f in fgs)
    skew = sum(int(f.get("skew_incidents", 0)) for f in fgs)
    feature_store = {
        "feature_groups": len(fgs),
        "feature_reuse_count": reuse,
        "training_serving_skew_incidents": skew,
    }

    scorecard: dict[str, Any] = {
        "pipeline": pipeline,
        "registry": registry,
        "monitoring": monitoring,
        "deployment": deployment,
        "governance": governance,
        "feature_store": feature_store,
    }

    # Composite health = mean of the four rate-style KPIs that are populated.
    rate_components = []
    if runs:
        rate_components.append(pipeline["success_rate"])
    if mons:
        rate_components.append(monitoring["baseline_coverage"])
    if deps:
        rate_components.append(deployment["deploy_success_rate"])
    if regs:
        rate_components.append(governance["models_with_lineage_pct"])
    composite = round(_mean(rate_components), 4) if rate_components else 0.0

    scorecard["overall"] = {
        "composite_health": composite,
        "maturity_level": maturity_level(scorecard),
    }
    return scorecard


@tool
@tool_error_boundary(source="tools.mlops.kpis")
def mlops_kpi_scorecard(events: dict[str, Any] | None = None) -> dict[str, Any]:
    """``@tool`` wrapper over :func:`compute_mlops_kpis` for the MLOps Lead (master plan §8 D7).

    Rolls collected run/registry/monitor/deploy/feature events into the MLOps KPI
    scorecard the transcript asks for ("how do we benchmark the MLOps side / what
    are the KPIs"). Fail-soft: never raises — returns a zeroed scorecard on bad input.

    Args:
        events: dict with optional ``pipeline_runs`` / ``registrations`` /
            ``monitors`` / ``deployments`` / ``feature_groups`` event lists.

    Returns:
        ``{"pipeline","registry","monitoring","deployment","governance",
        "feature_store","overall"}`` scorecard, or ``{"success": False, "error": ...}``.
    """
    try:
        return compute_mlops_kpis(events)
    except Exception as e:
        return {"success": False, "error": f"mlops_kpi_scorecard failed: {e}"}
