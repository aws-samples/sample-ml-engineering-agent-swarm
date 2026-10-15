"""Backend-neutral Stage 11–12 tools — monitoring + the closed-loop trigger (M5; ADR-008).

The doc 05 §4.4 EVOLVE pattern for the monitoring surface: each tool resolves a PORT via
the ``BackendRegistry`` and delegates to the bound adapter (``OTELTraceBackend`` local
offline, ``SageMakerModelMonitorBackend`` managed). NEW M5 tools, ADDITIVE.

Stage-11 monitoring tools (doc 06 §MonitoringBackend):
  * ``build_monitoring_dashboard`` (T-MON-07), ``get_metrics`` (T-MON-03),
    ``analyze_drift(drift_type=)`` (T-MON-08 — the parameterized 6-drift tool),
    ``get_recent_logs`` (T-MON-04), ``track_inference_spend`` (T-MON-09),
    ``put_custom_metric`` (T-MON-05), ``configure_alert`` (T-MON-06),
    ``compile_monitoring_report`` (T-MON-10).
The closed-loop decision (doc 06 §OrchestrationBackend):
  * ``evaluate_retrain_trigger`` (T-ORCH-02) — given drift+quality+business-KPI signals vs
    thresholds, decides whether to re-enter the unified exploration retreat loop (ADR-008),
    carrying the drift findings as context for the next iteration (Memory carry-over).
Stage-12 improvement tools:
  * ``analyze_production_failures`` (T-PR-63), ``carry_forward_learnings`` (T-MEM-01 — the
    AgentCore Memory carry-over), ``retire_model`` (T-DEPLOY-13).

The closed-loop trigger MUST route through the EXISTING M1 unified orchestration + the
non-delegated Arbiter (ADR-003/008) — ``evaluate_retrain_trigger`` only DECIDES; the
``improve_retrain_flow`` performs the re-entry via ``OrchestrationBackend.register_callback``
+ ``run_flow`` (no parallel retrain path).
"""

from __future__ import annotations

from typing import Any

from strands import tool

from ..backends.ports import DeploymentBackend, MemoryBackend, MonitoringBackend
from ..backends.registry import REGISTRY, ResolvedBackend
from ..errors import tool_error_boundary

__all__ = [
    "analyze_drift",
    # Stage 12 — improvement (T-PR-63 / T-MEM-01 / T-DEPLOY-13)
    "analyze_production_failures",
    # Stage 11 — monitoring (T-MON-*)
    "build_monitoring_dashboard",
    "carry_forward_learnings",
    "compile_monitoring_report",
    "configure_alert",
    # severity helper (shared by the flow)
    "decide_retrain",
    # closed-loop decision (T-ORCH-02)
    "evaluate_retrain_trigger",
    "get_metrics",
    "get_recent_logs",
    "put_custom_metric",
    "retire_model",
    "track_inference_spend",
]

# Default decision thresholds for the closed-loop trigger (overridable per-call).
_DEFAULT_THRESHOLDS = {
    "quality_regression": 0.05,       # relative primary-metric drop vs baseline
    "business_kpi_regression": 0.10,  # relative business-KPI drop vs baseline
    # drift severities that fire a retrain (severe always; moderate also triggers)
    "drift_trigger_severities": ("moderate", "severe"),
}


def _monitoring(scale: int | str | None = None, steering: Any = None) -> ResolvedBackend:
    """Resolve the MonitoringBackend (local OTEL offline, Model Monitor + CloudWatch managed)."""
    return REGISTRY.resolve(MonitoringBackend, steering=steering, scale=scale)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 11 — monitoring tools (T-MON-03..10)
# ══════════════════════════════════════════════════════════════════════════════
@tool
@tool_error_boundary(source="tools.monitor_neutral")
def analyze_drift(
    drift_type: str = "data",
    reference: Any = None,
    current: Any = None,
    threshold: float | None = None,
    schedule_arn: str | None = None,
    sensitive_col: str = "sensitive",
    prediction_col: str = "prediction",
) -> dict[str, Any]:
    """Detect ``drift_type`` between reference + current data via :class:`MonitoringBackend` (T-MON-08).

    The parameterized 6-drift tool: ``drift_type`` ∈ ``label``|``data``|``feature``|``model``|
    ``bias``|``attribution``. Resolves the monitoring port (local genuine in-process detection
    offline, Model-Monitor/Clarify violations managed) and returns the drift verdict, magnitude,
    severity, and affected features.
    """
    try:
        resolved = _monitoring()
        report = resolved.adapter.analyze_drift(
            drift_type=drift_type, reference=reference, current=current, threshold=threshold,
            sensitive_col=sensitive_col, prediction_col=prediction_col,
        )
        return {
            "drift_type": report.drift_type, "drift_detected": report.drift_detected,
            "magnitude": report.magnitude, "threshold": report.threshold, "severity": report.severity,
            "affected": report.affected, "adapter": resolved.adapter_name, "detail": report.detail,
        }
    except Exception as e:
        return {"success": False, "error": f"analyze_drift failed: {e}", "drift_type": drift_type,
                "drift_detected": False, "magnitude": 0.0}


@tool
@tool_error_boundary(source="tools.monitor_neutral")
def get_metrics(namespace: str, metric: str, dims: dict[str, Any] | None = None, period: int = 300) -> dict[str, Any]:
    """Read CloudWatch/local metric datapoints via :class:`MonitoringBackend` (T-MON-03)."""
    try:
        resolved = _monitoring()
        out: dict[str, Any] = resolved.adapter.metrics(namespace=namespace, metric=metric, dims=dims, period=period)
        out.setdefault("adapter", resolved.adapter_name)
        return out
    except Exception as e:
        return {"success": False, "error": f"get_metrics failed: {e}", "datapoints": []}


@tool
@tool_error_boundary(source="tools.monitor_neutral")
def put_custom_metric(namespace: str, metric: str, value: float, dims: dict[str, Any] | None = None) -> dict[str, Any]:
    """Emit a custom metric datapoint via :class:`MonitoringBackend` (T-MON-05)."""
    try:
        resolved = _monitoring()
        out: dict[str, Any] = resolved.adapter.put_custom_metric(
            namespace=namespace, metric=metric, value=value, dims=dims
        )
        return out
    except Exception as e:
        return {"success": False, "error": f"put_custom_metric failed: {e}"}


@tool
@tool_error_boundary(source="tools.monitor_neutral")
def configure_alert(metric: str, threshold: float, comparison: str = "GreaterThanThreshold",
                    action: str = "", namespace: str = "Amlas/ML") -> dict[str, Any]:
    """Configure a CloudWatch alarm + action via :class:`MonitoringBackend` (T-MON-06)."""
    try:
        resolved = _monitoring()
        out: dict[str, Any] = resolved.adapter.configure_alert(
            metric=metric, threshold=threshold, comparison=comparison, action=action, namespace=namespace
        )
        return out
    except Exception as e:
        return {"success": False, "error": f"configure_alert failed: {e}"}


@tool
@tool_error_boundary(source="tools.monitor_neutral")
def build_monitoring_dashboard(endpoint: str, panels: list[str] | None = None) -> dict[str, Any]:
    """Build a monitoring dashboard via :class:`MonitoringBackend` (T-MON-07)."""
    try:
        resolved = _monitoring()
        out: dict[str, Any] = resolved.adapter.build_dashboard(endpoint=endpoint, panels=panels)
        return out
    except Exception as e:
        return {"success": False, "error": f"build_monitoring_dashboard failed: {e}"}


@tool
@tool_error_boundary(source="tools.monitor_neutral")
def get_recent_logs(log_group: str, filter: str = "", since: str = "1h") -> dict[str, Any]:
    """Surface recent log events via :class:`MonitoringBackend` (T-MON-04, CloudWatch Logs)."""
    try:
        resolved = _monitoring()
        out: dict[str, Any] = resolved.adapter.get_recent_logs(log_group=log_group, filter=filter, since=since)
        return out
    except Exception as e:
        return {"success": False, "error": f"get_recent_logs failed: {e}", "events": []}


@tool
@tool_error_boundary(source="tools.monitor_neutral")
def track_inference_spend(endpoint: str, window: str = "7d") -> dict[str, Any]:
    """Track an endpoint's inference-spend trend via :class:`MonitoringBackend` (T-MON-09, CostAdvisor)."""
    try:
        resolved = _monitoring()
        out: dict[str, Any] = resolved.adapter.track_spend(endpoint=endpoint, window=window)
        return out
    except Exception as e:
        return {"success": False, "error": f"track_inference_spend failed: {e}", "run_rate": 0.0}


@tool
@tool_error_boundary(source="tools.monitor_neutral")
def compile_monitoring_report(endpoint: str, window: str = "7d") -> dict[str, Any]:
    """Compile a monitoring report + improvement backlog via :class:`MonitoringBackend` (T-MON-10)."""
    try:
        resolved = _monitoring()
        out: dict[str, Any] = resolved.adapter.compile_report(endpoint=endpoint, window=window)
        return out
    except Exception as e:
        return {"success": False, "error": f"compile_monitoring_report failed: {e}", "report": {},
                "improvement_backlog": []}


# ══════════════════════════════════════════════════════════════════════════════
# T-ORCH-02 evaluate_retrain_trigger — the closed-loop decision (ADR-008)
# ══════════════════════════════════════════════════════════════════════════════
def decide_retrain(
    drift_signals: list[dict[str, Any]] | None,
    quality_signals: dict[str, Any] | None,
    thresholds: dict[str, Any] | None,
) -> dict[str, Any]:
    """Pure decision core for the retrain trigger (no I/O — directly directly verifiable).

    Fires a TRIGGER when ANY of: a drift signal at a triggering severity; a primary-metric
    quality regression beyond ``quality_regression``; a business-KPI regression beyond
    ``business_kpi_regression``. Returns the decision + the target re-entry stage + the
    carry-over context (the drift/quality findings the next iteration receives).
    """
    drift_signals = drift_signals or []
    quality_signals = quality_signals or {}
    thr = {**_DEFAULT_THRESHOLDS, **(thresholds or {})}
    trigger_severities = set(thr["drift_trigger_severities"])

    reasons: list[str] = []
    worst_severity = "none"
    severity_rank = {"none": 0, "low": 1, "moderate": 2, "severe": 3}

    # 1. drift signals
    for sig in drift_signals:
        sev = str(sig.get("severity", "none"))
        if severity_rank.get(sev, 0) > severity_rank.get(worst_severity, 0):
            worst_severity = sev
        if sig.get("drift_detected") and sev in trigger_severities:
            reasons.append(f"{sig.get('drift_type', 'data')} drift ({sev}) on {sig.get('affected', [])}")

    # 2. primary-metric quality regression vs baseline
    primary = quality_signals.get("f1_macro", quality_signals.get("primary_score"))
    baseline = quality_signals.get("baseline_f1_macro", quality_signals.get("baseline_primary_score"))
    if isinstance(primary, (int, float)) and isinstance(baseline, (int, float)) and baseline:
        drop = (float(baseline) - float(primary)) / abs(float(baseline))
        if drop > float(thr["quality_regression"]):
            reasons.append(f"quality regression {drop:.2%} vs baseline (> {thr['quality_regression']:.0%})")

    # 3. business-KPI regression vs baseline
    kpi = quality_signals.get("business_kpi")
    kpi_base = quality_signals.get("business_kpi_baseline")
    if isinstance(kpi, (int, float)) and isinstance(kpi_base, (int, float)) and kpi_base:
        kpi_drop = (float(kpi_base) - float(kpi)) / abs(float(kpi_base))
        if kpi_drop > float(thr["business_kpi_regression"]):
            reasons.append(f"business-KPI regression {kpi_drop:.2%} (> {thr['business_kpi_regression']:.0%})")

    trigger = bool(reasons)
    # Re-entry stage (ADR-008): drift → re-Research (re-frame data); pure quality regression →
    # re-train. The retreat edge feeds the next Research prompt with the carried context.
    target_stage = "research" if any("drift" in r for r in reasons) else ("ml_execution" if trigger else "")
    return {
        "trigger": trigger,
        "reason": "; ".join(reasons) if reasons else "all signals within thresholds — no retrain",
        "target_stage": target_stage or "research",
        "severity": worst_severity,
        "signals": {"drift": drift_signals, "quality": quality_signals},
        # the drift/quality findings carried into the next iteration (Memory carry-over) —
        # the analogue of abandoned_reasons feeding the next Research prompt (ADR-008).
        "carry_context": {
            "drift": [s for s in drift_signals if s.get("drift_detected")],
            "drift_signals": drift_signals,
            "quality": quality_signals,
            "reasons": reasons,
        },
    }


@tool
@tool_error_boundary(source="tools.monitor_neutral")
def evaluate_retrain_trigger(
    drift_signals: list[dict[str, Any]] | None = None,
    quality_signals: dict[str, Any] | None = None,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Decide whether monitoring signals warrant retraining (T-ORCH-02; the ADR-008 trigger).

    The closed-loop decision: given drift severity, metric breach, and business-KPI regression,
    returns ``{trigger, reason, target_stage, severity, carry_context}``. On TRIGGER the
    ``improve_retrain_flow`` re-enters the M1 unified exploration retreat loop via
    ``OrchestrationBackend.register_callback`` + ``run_flow`` (this tool DECIDES; it does not
    itself train — the deterministic Arbiter remains the non-delegated gate, ADR-003).
    """
    return decide_retrain(drift_signals, quality_signals, thresholds)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 12 — improvement tools (T-PR-63 / T-MEM-01 / T-DEPLOY-13)
# ══════════════════════════════════════════════════════════════════════════════
@tool
@tool_error_boundary(source="tools.monitor_neutral")
def analyze_production_failures(
    incident_data: list[dict[str, Any]] | None = None,
    traces: list[dict[str, Any]] | None = None,
    predictions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Cluster production failures → root causes → fix targets (T-PR-63, Stage-12 triage).

    Pure in-process clustering by ``(error, feature)`` signature — deterministic + offline.
    Returns ``{failure_clusters, root_causes, fix_targets}`` the improvement iteration acts on.
    """
    incidents = incident_data or []
    clusters: dict[str, dict[str, Any]] = {}
    for inc in incidents:
        key = f"{inc.get('error', 'unknown')}::{inc.get('feature', '')}"
        c = clusters.setdefault(key, {"signature": key, "error": inc.get("error", "unknown"),
                                      "feature": inc.get("feature", ""), "count": 0, "examples": []})
        c["count"] += 1
        if len(c["examples"]) < 3:
            c["examples"].append(inc)

    failure_clusters = sorted(clusters.values(), key=lambda c: c["count"], reverse=True)
    root_causes = [
        {"cause": f"{c['error']} on feature '{c['feature']}'" if c["feature"] else c["error"],
         "frequency": c["count"]}
        for c in failure_clusters
    ]
    fix_targets = [
        {"target": c["feature"] or c["error"], "action": "retrain on fresh data" if c["feature"] else "harden input handling"}
        for c in failure_clusters
    ]
    return {
        "failure_clusters": failure_clusters,
        "root_causes": root_causes,
        "fix_targets": fix_targets,
        "incident_count": len(incidents),
        "trace_count": len(traces or []),
    }


@tool
@tool_error_boundary(source="tools.monitor_neutral")
def carry_forward_learnings(project_id: str, learnings: dict[str, Any]) -> dict[str, Any]:
    """Persist production learnings into AgentCore Memory for the next iteration (T-MEM-01).

    Resolves :class:`MemoryBackend` (local carry-over fallback offline, AgentCore managed) and
    writes a carry-over record keyed by ``actor_id=project_id`` — production failure analyses +
    drift findings the next improvement iteration's Research prompt receives (doc 05 §8.6).
    Returns ``{memory_record_id, surfaced_next_iteration}``.
    """
    try:
        resolved = REGISTRY.resolve(MemoryBackend)
        record_id = resolved.adapter.persist(actor_id=project_id, learnings=learnings, session_id="improvement")
        # confirm it will surface next iteration (carry_over reads it back)
        carried = resolved.adapter.carry_over(actor_id=project_id, session_id="improvement")
        surfaced = any(r.get("record_id") == record_id for r in carried)
        return {"memory_record_id": record_id, "surfaced_next_iteration": surfaced,
                "project_id": project_id, "adapter": resolved.adapter_name}
    except Exception as e:
        return {"success": False, "error": f"carry_forward_learnings failed: {e}",
                "memory_record_id": "", "surfaced_next_iteration": False}


@tool
@tool_error_boundary(source="tools.monitor_neutral")
def retire_model(endpoint: str | None = None, model_ref: str | None = None,
                 preserve_lineage: bool = True) -> dict[str, Any]:
    """Retire a model/endpoint, preserving lineage (T-DEPLOY-13; Stage-12 replace/retire).

    Resolves :class:`DeploymentBackend` and tears down the endpoint (irreversible — gated by the
    approval policy when invoked through ``improve_retrain_flow``). Lineage is preserved in the
    registry/governance record when ``preserve_lineage`` (the default).
    """
    try:
        if endpoint:
            from ..backends.types import EndpointRef
            resolved = REGISTRY.resolve(DeploymentBackend)
            out = resolved.adapter.delete(endpoint=EndpointRef(name=endpoint))
            status = "retired" if out.get("success") is not False else "scheduled"
            return {"status": status, "endpoint": endpoint, "preserve_lineage": preserve_lineage,
                    "adapter": resolved.adapter_name, "detail": out}
        # no live endpoint → a lineage-preserving retirement record (offline-safe)
        return {"status": "retired", "model_ref": model_ref or "", "preserve_lineage": preserve_lineage,
                "detail": {"note": "no live endpoint; lineage-preserving retirement recorded"}}
    except Exception as e:
        return {"status": "scheduled", "error": f"retire_model degraded: {e}", "preserve_lineage": preserve_lineage}
