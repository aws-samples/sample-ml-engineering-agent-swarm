"""``monitor_flow`` — the Stage-11 monitoring sequence over the MonitoringBackend port (roadmap M5).

The invariant Stage-11 sequence (doc 06 §5.3 ``monitor_flow`` row; ADR-008):

    **set up baselines/schedule (T-MON-01) → collect metrics (T-MON-03) → analyze_drift
    (T-MON-08, all types) → configure alerts (T-MON-06) → compile report (T-MON-10) →
    evaluate_retrain_trigger (T-ORCH-02 — the closed-loop decision)**

monitor_flow is read/observe-heavy (CHEAP_READ — it provisions nothing irreversible; the
managed schedule-setup sub-step is the only spend, gated within the flow), so the public
entrypoint runs without a blocking approval prompt. Its terminal output is the closed-loop
**retrain trigger decision** the Monitoring Lead hands to ``improve_retrain_flow`` — the
production-monitoring signal that re-enters the unified exploration retreat loop (ADR-008).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..tools import monitor_neutral as M
from ._util import raw

__all__ = ["MonitorFlowResult", "run_monitor_flow"]

# The 4 Model-Monitor-style drift families analyzed by default (data-quality, model-quality,
# bias, feature-attribution); ``label``/``feature`` are also available via ``drift_types=``.
_DEFAULT_DRIFT_TYPES = ("data", "model", "bias", "attribution")


@dataclass
class MonitorFlowResult:
    """The invariant output contract of one ``monitor_flow`` run."""

    status: str  # "completed" | "failed"
    endpoint: str = ""
    schedule: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    drift_reports: list[dict[str, Any]] = field(default_factory=list)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    retrain_decision: dict[str, Any] = field(default_factory=dict)
    # G4: the real inference-spend reading (Cost Explorer when managed; synthetic local offline) —
    # the Stage-11 cost panel the master plan requires the monitor report to surface.
    spend: dict[str, Any] = field(default_factory=dict)
    adapter: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


def run_monitor_flow(
    *,
    endpoint: str,
    reference: Any = None,
    current: Any = None,
    drift_types: list[str] | None = None,
    quality_signals: dict[str, Any] | None = None,
    baseline_uri: str = "",
    setup_schedule: bool = False,
    thresholds: dict[str, Any] | None = None,
    alert_specs: list[dict[str, Any]] | None = None,
    window: str = "7d",
    steering: Any | None = None,
) -> MonitorFlowResult:
    """Run the Stage-11 monitor sequence and return the closed-loop trigger decision.

    Fixed steps (each via the MonitoringBackend port, local offline ↔ Model Monitor/CloudWatch
    managed):
      1. **setup** the monitoring baseline/schedule (only when ``setup_schedule`` — the managed,
         spend-incurring sub-step; off by default so the flow stays CHEAP_READ);
      2. **collect** a representative metric (T-MON-03);
      3. **analyze_drift** across all requested drift types (T-MON-08);
      4. **configure alerts** for the supplied specs (T-MON-06);
      5. **compile** a monitoring report + improvement backlog (T-MON-10);
      6. **evaluate_retrain_trigger** — the closed-loop decision (T-ORCH-02) fed by the drift
         reports + quality signals; the Monitoring Lead routes a TRIGGER to improve_retrain_flow.
    """
    drift_types = [d.strip().lower() for d in (drift_types or _DEFAULT_DRIFT_TYPES)]

    # ── 1. (optional) set up the managed baseline/schedule ──────────────────────
    schedule: dict[str, Any] = {}
    if setup_schedule:
        try:
            from ..backends.ports import MonitoringBackend
            from ..backends.registry import REGISTRY

            resolved = REGISTRY.resolve(MonitoringBackend, steering=steering)
            ref = resolved.adapter.setup(endpoint=endpoint, baseline_uri=baseline_uri,
                                         monitor_types=["data_quality"], schedule="hourly")
            schedule = {"schedule_name": getattr(ref, "name", ""), "monitor_type": getattr(ref, "monitor_type", ""),
                        "adapter": resolved.adapter_name}
        except Exception as e:
            schedule = {"wired": False, "note": f"schedule setup deferred: {e}"}

    # ── 2. collect a representative metric (T-MON-03) ───────────────────────────
    metrics = raw(M.get_metrics)(namespace="Amlas/ML", metric="Invocations", dims={"endpoint": endpoint})

    # ── 3. analyze_drift across all requested types (T-MON-08) ──────────────────
    drift_reports: list[dict[str, Any]] = []
    for dt in drift_types:
        rep = raw(M.analyze_drift)(drift_type=dt, reference=reference, current=current,
                                   threshold=(thresholds or {}).get(f"{dt}_threshold"))
        drift_reports.append(rep)

    # ── 4. configure alerts (T-MON-06) ──────────────────────────────────────────
    alerts: list[dict[str, Any]] = []
    specs = alert_specs or [{"metric": "Latency", "threshold": 100.0, "comparison": "GreaterThanThreshold"},
                            {"metric": "DriftMagnitude", "threshold": 0.2, "comparison": "GreaterThanThreshold"}]
    for spec in specs:
        alerts.append(raw(M.configure_alert)(
            metric=spec["metric"], threshold=spec["threshold"],
            comparison=spec.get("comparison", "GreaterThanThreshold"),
            action=spec.get("action", "sns:amlas-retrain"),
        ))

    # ── 5. compile a monitoring report (T-MON-10) ───────────────────────────────
    report = raw(M.compile_monitoring_report)(endpoint=endpoint, window=window)

    # ── 5b. track REAL inference spend (T-MON-09 — Cost Explorer when managed, G4) ──
    # The Stage-11 cost panel: the managed adapter issues a real ``ce:GetCostAndUsage`` per
    # endpoint (run_rate / spend_trend / vs_budget); the local OTEL adapter returns the
    # offline synthetic trend. Either way the monitor report now SURFACES a spend figure
    # (master plan §G4 step 5) rather than the old P2 placeholder. Best-effort: a spend-read
    # failure never fails the (CHEAP_READ) monitor flow.
    spend = raw(M.track_inference_spend)(endpoint=endpoint, window=window)
    if isinstance(report, dict):
        report.setdefault("spend", spend)
        if isinstance(report.get("report"), dict):
            report["report"]["run_rate"] = spend.get("run_rate")
            report["report"]["spend_trend"] = spend.get("spend_trend")

    # ── 6. evaluate the closed-loop retrain trigger (T-ORCH-02 — ADR-008) ───────
    decision = raw(M.evaluate_retrain_trigger)(
        drift_signals=drift_reports, quality_signals=(quality_signals or {}), thresholds=thresholds or {},
    )

    return MonitorFlowResult(
        status="completed",
        endpoint=endpoint,
        schedule=schedule,
        metrics=metrics,
        drift_reports=drift_reports,
        alerts=alerts,
        report=report,
        retrain_decision=decision,
        spend=spend,
        adapter=metrics.get("adapter", ""),
        detail={"drift_types": drift_types, "window": window},
    )
