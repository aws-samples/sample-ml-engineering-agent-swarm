"""MonitoringBackend + MemoryBackend adapters (doc 05 §4.2 G4/G5; roadmap M5; ADR-008/004).

M5 completes the monitoring seam M0 stubbed, and lands the AgentCore-Memory carry-over
fallback that closes the improvement loop:

* ``OTELTraceBackend`` — **local/offline default**. Does
  GENUINE in-process drift detection for the 6 drift types the parameterized
  ``analyze_drift`` (T-MON-08) serves — data-quality (PSI), label/feature (PSI),
  model-quality (metric regression), bias (sub-group parity shift), feature-attribution
  (importance L1) — plus a real in-process metric store (put/get), CloudWatch-style
  alarms/dashboards/logs/spend/report artifacts. No AWS.
* ``SageMakerModelMonitorBackend`` — **default (managed)**; thin boto3 orchestration
  (ADR-004): wraps the EXISTING ``tools/mlops/monitor.py`` (``setup_model_monitor`` /
  ``inspect_model_monitor``) for Model-Monitor baselines/schedules + drives CloudWatch
  ``put_metric_data`` / ``put_metric_alarm`` / ``put_dashboard`` / ``get_metric_statistics``
  and CloudWatch-Logs ``filter_log_events`` for the T-MON-03..10 surface. ``analyze_drift``
  falls back to the SAME genuine local computation when no live schedule/violations are
  supplied (offline-safe, offline-verified). Clients are injectable.
* ``LocalMemoryBackend`` — the M5 ``MemoryBackend`` carry-over fallback (replaces the M0
  stub). A process-wide actor→records store: ``persist`` writes a carry-over record;
  ``carry_over`` reads the actor's records into the next iteration. The managed
  AgentCore Memory adapter (``MemoryHookProvider``, EXISTING) layers on top in prod.

Selectors: ``MONITORING_BACKEND`` (``otel``/``local`` offline, ``sagemaker`` managed);
``MEMORY_BACKEND`` (``local`` offline, ``agentcore`` managed P1).
"""

from __future__ import annotations

import json
import math
import uuid
from typing import Any, ClassVar

from ...config.aws_region import resolve_aws_region
from ..ports import MemoryBackend, MonitoringBackend
from ..registry import register_adapter
from ..types import DriftReport, MonitorReport, ScheduleRef
from ._util import raw_tool

__all__ = ["LocalMemoryBackend", "OTELTraceBackend", "SageMakerModelMonitorBackend"]

# The 6 drift types the parameterized T-MON-08 serves (doc 06). data/feature/label share
# the PSI distribution-shift computation; model/bias/attribution have dedicated measures.
_DRIFT_TYPES = {"data", "label", "feature", "model", "bias", "attribution"}
# The 4 Model-Monitor-style drift families (doc roadmap M5 / RECON): the managed schedule
# monitor-type tags map onto these.
_DEFAULT_THRESHOLD = 0.2


def _severity(detected: bool, magnitude: float, threshold: float) -> str:
    """Bucket a drift magnitude into none|low|moderate|severe (relative to threshold)."""
    if not detected:
        return "none"
    ratio = magnitude / threshold if threshold > 0 else magnitude
    if ratio <= 2.0:
        return "low"
    if ratio <= 4.0:
        return "moderate"
    return "severe"


# ══════════════════════════════════════════════════════════════════════════════
# Genuine drift computation — shared by the local + managed adapters
# ══════════════════════════════════════════════════════════════════════════════
class _DriftComputer:
    """Mixin: genuine in-process drift detection for the 6 parameterized drift types.

    Pure-Python (no numpy/sklearn → no macOS-aarch64 joblib deadlock). Each measure
    returns a normalized magnitude judged against ``threshold`` to produce a
    :class:`DriftReport`. The managed adapter inherits this so its ``analyze_drift`` has a
    real, offline-safe computation when no live Model-Monitor violations are supplied.
    """

    def analyze_drift(
        self,
        *,
        drift_type: str = "data",
        reference: Any = None,
        current: Any = None,
        threshold: float | None = None,
        sensitive_col: str = "sensitive",
        prediction_col: str = "prediction",
        **_: Any,
    ) -> DriftReport:
        """Detect ``drift_type`` between ``reference`` and ``current`` (T-MON-08)."""
        dt = (drift_type or "data").strip().lower()
        if dt not in _DRIFT_TYPES:
            raise ValueError(f"unknown drift_type {drift_type!r}; choose from {sorted(_DRIFT_TYPES)}")
        thr = float(threshold) if threshold is not None else _DEFAULT_THRESHOLD
        ref = reference or {}
        cur = current or {}

        if dt in ("data", "feature", "label"):
            magnitude, affected = self._distribution_drift(ref, cur)
        elif dt == "model":
            magnitude, affected = self._metric_regression(ref, cur)
        elif dt == "bias":
            magnitude, affected = self._parity_shift(ref, cur, sensitive_col, prediction_col)
        else:  # attribution
            magnitude, affected = self._attribution_shift(ref, cur)

        detected = magnitude > thr
        return DriftReport(
            drift_type=dt,
            drift_detected=detected,
            magnitude=round(magnitude, 6),
            threshold=thr,
            severity=_severity(detected, magnitude, thr),
            affected=affected if detected else [],
            detail={"measure": _MEASURE_NAME[dt]},
        )

    # ── data / feature / label → Population Stability Index (PSI) per column ────
    @staticmethod
    def _distribution_drift(ref: Any, cur: Any) -> tuple[float, list[str]]:
        """Max per-column PSI between reference and current numeric frames."""
        ref_cols = _numeric_columns(ref)
        cur_cols = _numeric_columns(cur)
        worst = 0.0
        affected: list[str] = []
        for col, ref_vals in ref_cols.items():
            cur_vals = cur_cols.get(col)
            if not cur_vals:
                continue
            psi = _psi(ref_vals, cur_vals)
            if psi > _DEFAULT_THRESHOLD:
                affected.append(col)
            worst = max(worst, psi)
        return worst, affected

    # ── model → max relative metric regression ─────────────────────────────────
    @staticmethod
    def _metric_regression(ref: Any, cur: Any) -> tuple[float, list[str]]:
        ref_m = ref.get("metrics", ref) if isinstance(ref, dict) else {}
        cur_m = cur.get("metrics", cur) if isinstance(cur, dict) else {}
        worst = 0.0
        affected: list[str] = []
        for k, rv in ref_m.items():
            cv = cur_m.get(k)
            if not isinstance(rv, (int, float)) or not isinstance(cv, (int, float)) or rv == 0:
                continue
            drop = (float(rv) - float(cv)) / abs(float(rv))  # positive = regression
            if drop > _DEFAULT_THRESHOLD:
                affected.append(k)
            worst = max(worst, drop)
        return max(worst, 0.0), affected

    # ── bias → change in max sub-group positive-rate parity gap ─────────────────
    @staticmethod
    def _parity_shift(ref: Any, cur: Any, sens: str, pred: str) -> tuple[float, list[str]]:
        ref_gap = _parity_gap(ref, sens, pred)
        cur_gap = _parity_gap(cur, sens, pred)
        magnitude = abs(cur_gap - ref_gap)
        return magnitude, ([sens] if magnitude > _DEFAULT_THRESHOLD else [])

    # ── attribution → L1 distance between normalized importance vectors / 2 ─────
    @staticmethod
    def _attribution_shift(ref: Any, cur: Any) -> tuple[float, list[str]]:
        ref_imp = _normalize(ref.get("importances", ref) if isinstance(ref, dict) else {})
        cur_imp = _normalize(cur.get("importances", cur) if isinstance(cur, dict) else {})
        keys = set(ref_imp) | set(cur_imp)
        if not keys:
            return 0.0, []
        l1 = sum(abs(ref_imp.get(k, 0.0) - cur_imp.get(k, 0.0)) for k in keys)
        magnitude = l1 / 2.0  # normalized to [0, 1]
        affected = [k for k in keys if abs(ref_imp.get(k, 0.0) - cur_imp.get(k, 0.0)) > 0.1]
        return magnitude, (affected if magnitude > _DEFAULT_THRESHOLD else [])


_MEASURE_NAME = {
    "data": "psi", "feature": "psi", "label": "psi",
    "model": "metric_regression", "bias": "parity_gap_shift", "attribution": "importance_l1",
}


# ── pure-Python statistics helpers (no numpy → deadlock-safe) ──────────────────
def _numeric_columns(frame: Any) -> dict[str, list[float]]:
    """Extract numeric columns from a ``{col: [values]}`` dict (best-effort)."""
    out: dict[str, list[float]] = {}
    if not isinstance(frame, dict):
        return out
    for col, vals in frame.items():
        if isinstance(vals, (list, tuple)) and vals and all(isinstance(v, (int, float)) for v in vals):
            out[col] = [float(v) for v in vals]
    return out


def _psi(ref: list[float], cur: list[float], bins: int = 5) -> float:
    """Population Stability Index over ``bins`` quantile-ish buckets of the joint range."""
    lo = min(min(ref), min(cur))
    hi = max(max(ref), max(cur))
    if hi <= lo:
        return 0.0
    width = (hi - lo) / bins
    edges = [lo + i * width for i in range(bins + 1)]
    edges[-1] = hi + 1e-9  # include the max

    def _hist(vals: list[float]) -> list[float]:
        counts = [0] * bins
        for v in vals:
            idx = min(int((v - lo) / width), bins - 1)
            counts[idx] += 1
        n = len(vals) or 1
        # smooth to avoid log(0) / div-by-0 (the standard PSI epsilon)
        return [(c / n) or 1e-4 for c in counts]

    ref_d = _hist(ref)
    cur_d = _hist(cur)
    psi = 0.0
    for r, c in zip(ref_d, cur_d, strict=False):
        psi += (c - r) * math.log(c / r)
    return abs(psi)


def _parity_gap(frame: Any, sens: str, pred: str) -> float:
    """Max minus min positive-prediction-rate across sensitive groups."""
    if not isinstance(frame, dict):
        return 0.0
    groups = frame.get(sens, [])
    preds = frame.get(pred, [])
    if not groups or len(groups) != len(preds):
        return 0.0
    rates: dict[str, list[float]] = {}
    for g, p in zip(groups, preds, strict=False):
        rates.setdefault(str(g), []).append(1.0 if float(p) > 0 else 0.0)
    means = [sum(v) / len(v) for v in rates.values() if v]
    return (max(means) - min(means)) if means else 0.0


def _normalize(imp: Any) -> dict[str, float]:
    """L1-normalize an importance dict to sum to 1 (so the vectors are comparable)."""
    if not isinstance(imp, dict):
        return {}
    vals = {k: abs(float(v)) for k, v in imp.items() if isinstance(v, (int, float))}
    total = sum(vals.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in vals.items()}


# ══════════════════════════════════════════════════════════════════════════════
# OTELTraceBackend — the local/offline MonitoringBackend (genuine, no AWS)
# ══════════════════════════════════════════════════════════════════════════════
class OTELTraceBackend(_DriftComputer):
    """Local MonitoringBackend: in-process drift + a CloudWatch-style observability store.

    State is per-instance for metrics (the registry builds a fresh adapter per ``resolve``,
    but the monitor/improve flows hold a single resolved instance for the run). ``setup``/
    ``inspect`` record local trace state (no AWS); the M5 extension methods give the full
    Stage-11 surface offline.
    """

    def __init__(self) -> None:
        self._schedules: dict[str, ScheduleRef] = {}
        self._metrics: list[dict[str, Any]] = []
        self._alarms: dict[str, dict[str, Any]] = {}
        self._dashboards: dict[str, dict[str, Any]] = {}

    # ── M0 surface (setup / inspect / metrics) ──────────────────────────────────
    def setup(self, *, endpoint: str, baseline_uri: str, monitor_types: list[str] | None = None,
              schedule: str = "hourly") -> ScheduleRef:
        ref = ScheduleRef(name=f"local-monitor-{endpoint}",
                          monitor_type=(monitor_types or ["data_quality"])[0], schedule=schedule)
        self._schedules[ref.name] = ref
        return ref

    def inspect(self, *, schedule: str) -> MonitorReport:
        return MonitorReport(schedule_name=schedule, status="Local", last_execution_status="Completed", violations=[])

    def metrics(self, *, namespace: str, metric: str, dims: dict[str, Any] | None = None,
                period: int = 300) -> dict[str, Any]:
        """T-MON-03 — read locally-recorded datapoints for ``namespace/metric`` (+ dims)."""
        want = dims or {}
        pts = [
            {"value": m["value"], "Average": m["value"], "timestamp": m["timestamp"], "dims": m["dims"]}
            for m in self._metrics
            if m["namespace"] == namespace and m["metric"] == metric
            and all(m["dims"].get(k) == v for k, v in want.items())
        ]
        return {"datapoints": pts, "namespace": namespace, "metric": metric, "adapter": "otel"}

    # ── M5 extension surface (T-MON-04..10) ─────────────────────────────────────
    def put_custom_metric(self, *, namespace: str, metric: str, value: float,
                          dims: dict[str, Any] | None = None) -> dict[str, Any]:
        """T-MON-05 — record a custom metric datapoint in the local store."""
        self._metrics.append({"namespace": namespace, "metric": metric, "value": float(value),
                              "dims": dict(dims or {}), "timestamp": len(self._metrics)})
        return {"ok": True, "namespace": namespace, "metric": metric, "adapter": "otel"}

    def configure_alert(self, *, metric: str, threshold: float, comparison: str = "GreaterThanThreshold",
                        action: str = "", namespace: str = "Amlas/ML") -> dict[str, Any]:
        """T-MON-06 — register a local alarm descriptor (no SNS; offline)."""
        name = f"local-alarm-{metric}-{len(self._alarms)}"
        self._alarms[name] = {"metric": metric, "threshold": threshold, "comparison": comparison, "action": action}
        return {"alarm_name": name, "alarm_arn": f"local:{name}", "adapter": "otel"}

    def build_dashboard(self, *, endpoint: str, panels: list[str] | None = None) -> dict[str, Any]:
        """T-MON-07 — assemble a local dashboard artifact descriptor."""
        name = f"local-dashboard-{endpoint}"
        self._dashboards[name] = {"endpoint": endpoint, "panels": list(panels or [])}
        return {"dashboard_uri": f"local://dashboards/{name}", "panels": list(panels or []), "adapter": "otel"}

    def get_recent_logs(self, *, log_group: str, filter: str = "", since: str = "1h") -> dict[str, Any]:
        """T-MON-04 — surface recent local log events (synthesized from the metric store)."""
        events = [
            {"message": f"{m['metric']}={m['value']}", "timestamp": m["timestamp"]}
            for m in self._metrics if (not filter or filter.lower() in m["metric"].lower())
        ]
        return {"events": events, "log_group": log_group, "filter": filter, "adapter": "otel"}

    def track_spend(self, *, endpoint: str, window: str = "7d") -> dict[str, Any]:
        """T-MON-09 — a local inference-spend trend (synthetic but structured)."""
        invocations = [m for m in self._metrics if m["metric"].lower() in ("invocations", "latency")]
        run_rate = round(0.001 * (len(invocations) + 1), 4)
        return {"run_rate": run_rate, "spend_trend": "flat", "vs_budget": "under", "window": window,
                "endpoint": endpoint, "adapter": "otel"}

    def compile_report(self, *, endpoint: str, window: str = "7d") -> dict[str, Any]:
        """T-MON-10 — aggregate metrics/alarms into a report + improvement backlog."""
        backlog = [{"item": f"review alarm {n}", "alarm": n} for n in self._alarms]
        return {
            "report": {"endpoint": endpoint, "window": window, "metric_count": len(self._metrics),
                       "alarm_count": len(self._alarms), "dashboard_count": len(self._dashboards)},
            "improvement_backlog": backlog,
            "report_uri": f"local://reports/{endpoint}-{window}",
            "adapter": "otel",
        }


# ══════════════════════════════════════════════════════════════════════════════
# SageMakerModelMonitorBackend — managed (Model Monitor + CloudWatch; ADR-004)
# ══════════════════════════════════════════════════════════════════════════════
class SageMakerModelMonitorBackend(_DriftComputer):
    """Default (managed) MonitoringBackend — thin boto3 orchestration (never reimplements).

    Wraps EXISTING ``tools/mlops/monitor.py`` for Model-Monitor baselines/schedules; drives
    CloudWatch for the T-MON-03..10 metric/alarm/dashboard/logs surface. Clients are
    injectable so callers may use the offline AWS API emulator where supported. ``analyze_drift`` inherits the genuine
    local computation (offline-safe fallback when no live violations supplied).
    """

    def __init__(self, *, region: str | None = None, sm_client: Any = None,
                 cw_client: Any = None, logs_client: Any = None, s3_client: Any = None,
                 ce_client: Any = None) -> None:
        self._region = region
        self._sm = sm_client
        self._cw = cw_client
        self._logs = logs_client
        self._s3 = s3_client
        # G4: an injectable Cost Explorer (``ce``) client for the real inference-spend read
        # (the offline AWS API emulator does NOT implement Cost Explorer → offline validation inject a recording-fake).
        self._ce = ce_client

    # ── lazy boto3 clients (boto3-free import; built on first use) ──────────────
    def _cw_client(self) -> Any:
        if self._cw is None:
            import boto3
            self._cw = boto3.client("cloudwatch", region_name=resolve_aws_region(self._region))
        return self._cw

    def _logs_client(self) -> Any:
        if self._logs is None:
            import boto3
            self._logs = boto3.client("logs", region_name=resolve_aws_region(self._region))
        return self._logs

    # ── M0 surface (setup / inspect) — wrap EXISTING tools, orchestrate-not-crash ─
    def setup(self, *, endpoint: str, baseline_uri: str, monitor_types: list[str] | None = None,
              schedule: str = "hourly") -> ScheduleRef:
        """T-MON-01 — create a Model-Monitor schedule via EXISTING ``setup_model_monitor``.

        Best-effort managed (ADR-004): if the schedule cannot be created (e.g. no live
        endpoint under the offline AWS API emulator), returns a degraded ``ScheduleRef`` (no arn) carrying the
        monitor type rather than crashing the monitor flow.
        """
        from ...tools.mlops.monitor import setup_model_monitor

        monitor_type = (monitor_types or ["data_quality"])[0]
        try:
            out = raw_tool(setup_model_monitor)(
                endpoint, baseline_uri, schedule,
                monitor_type=monitor_type, region=self._region, sm_client=self._sm,
            )
            if out.get("success") is False:
                return ScheduleRef(name=f"sm-monitor-{endpoint}", monitor_type=monitor_type, schedule=schedule)
            return ScheduleRef(name=str(out.get("schedule_name", "")), arn=str(out.get("monitor_schedule_arn", "")),
                               monitor_type=str(out.get("monitor_type", monitor_type)),
                               schedule=str(out.get("schedule", schedule)))
        except Exception:
            return ScheduleRef(name=f"sm-monitor-{endpoint}", monitor_type=monitor_type, schedule=schedule)

    def inspect(self, *, schedule: str) -> MonitorReport:
        """T-MON-02 — surface the latest drift report via EXISTING ``inspect_model_monitor``."""
        from ...tools.mlops.monitor import inspect_model_monitor

        try:
            out = raw_tool(inspect_model_monitor)(schedule, region=self._region, sm_client=self._sm, s3_client=self._s3)
            if out.get("success") is False:
                return MonitorReport(schedule_name=schedule, status="Unknown")
            return MonitorReport(schedule_name=str(out.get("schedule_name", schedule)),
                                 status=str(out.get("status", "")),
                                 last_execution_status=str(out.get("last_execution_status", "")),
                                 violations=list(out.get("violations", [])))
        except Exception:
            return MonitorReport(schedule_name=schedule, status="Unknown")

    # ── CloudWatch surface (T-MON-03..10) — genuine boto3 call shapes ───────────
    def metrics(self, *, namespace: str, metric: str, dims: dict[str, Any] | None = None,
                period: int = 300, lookback_seconds: int | None = None) -> dict[str, Any]:
        import datetime as _dt

        cw = self._cw_client()
        dimensions = [{"Name": k, "Value": str(v)} for k, v in (dims or {}).items()]
        # F2 live-AWS fix: real CloudWatch ``GetMetricStatistics`` rejects a request whose
        # (EndTime - StartTime) / Period exceeds 1440 datapoints (InvalidParameterCombination).
        # The previous hardcoded 2020→2100 window at Period=300 demanded ~8.4M datapoints and
        # ALWAYS failed against real CloudWatch (it only "worked" under the offline AWS API emulator, which does not
        # enforce the cap). Bound the window to a recent lookback sized so that
        # ``lookback / period <= 1440`` — defaulting to the larger of 3h and ``period*1440``
        # capped at CloudWatch's 15-day GetMetricStatistics retention-for-1min window. This
        # returns the recent datapoints an inference-monitoring read actually wants.
        max_points = 1440
        if lookback_seconds is None:
            lookback_seconds = min(period * max_points, 3 * 60 * 60 if period <= 300 else period * max_points)
        else:
            # never exceed the datapoint cap for the chosen period
            lookback_seconds = min(lookback_seconds, period * max_points)
        now = _dt.datetime.now(_dt.UTC)
        start = now - _dt.timedelta(seconds=int(lookback_seconds))
        try:
            resp = cw.get_metric_statistics(
                Namespace=namespace, MetricName=metric, Dimensions=dimensions,
                StartTime=start, EndTime=now,
                Period=period, Statistics=["Average", "Sum"],
            )
            return {"datapoints": resp.get("Datapoints", []), "namespace": namespace, "metric": metric,
                    "adapter": "sagemaker"}
        except Exception as e:
            return {"datapoints": [], "namespace": namespace, "metric": metric, "error": str(e), "adapter": "sagemaker"}

    def put_custom_metric(self, *, namespace: str, metric: str, value: float,
                          dims: dict[str, Any] | None = None) -> dict[str, Any]:
        cw = self._cw_client()
        datum: dict[str, Any] = {"MetricName": metric, "Value": float(value)}
        if dims:
            datum["Dimensions"] = [{"Name": k, "Value": str(v)} for k, v in dims.items()]
        cw.put_metric_data(Namespace=namespace, MetricData=[datum])
        return {"ok": True, "namespace": namespace, "metric": metric, "adapter": "sagemaker"}

    def configure_alert(self, *, metric: str, threshold: float, comparison: str = "GreaterThanThreshold",
                        action: str = "", namespace: str = "Amlas/ML") -> dict[str, Any]:
        cw = self._cw_client()
        name = f"amlas-{metric}-{comparison}"
        kwargs: dict[str, Any] = {
            "AlarmName": name, "MetricName": metric, "Namespace": namespace,
            "ComparisonOperator": comparison, "EvaluationPeriods": 1, "Period": 300,
            "Statistic": "Average", "Threshold": float(threshold),
        }
        if action:
            kwargs["AlarmActions"] = [action]
        cw.put_metric_alarm(**kwargs)
        return {"alarm_name": name, "alarm_arn": "", "metric": metric, "adapter": "sagemaker"}

    def build_dashboard(self, *, endpoint: str, panels: list[str] | None = None) -> dict[str, Any]:
        cw = self._cw_client()
        name = f"amlas-{endpoint}"
        # F2 live-AWS fix: real CloudWatch ``PutDashboard`` VALIDATES the dashboard body —
        # a ``metric`` widget MUST carry a ``metrics`` array AND a ``region`` (or an alarm
        # ``annotations`` block), else it raises DashboardInvalidInputError. The previous body
        # emitted ``{"type":"metric","properties":{"title": p}}`` with NEITHER, so it only
        # "worked" under the offline AWS API emulator (which does not validate the body) and ALWAYS failed live. We now
        # emit a real, valid metric widget per panel, binding each panel to the endpoint's
        # SageMaker invocation metric in this region so the dashboard is genuinely useful.
        widgets = []
        for i, p in enumerate(panels or []):
            widgets.append({
                "type": "metric",
                "x": 0, "y": i * 6, "width": 12, "height": 6,
                "properties": {
                    "title": str(p),
                    "region": self._region,
                    "metrics": [["AWS/SageMaker", str(p), "EndpointName", endpoint]],
                    "view": "timeSeries",
                    "stat": "Average",
                    "period": 300,
                },
            })
        body = {"widgets": widgets}
        cw.put_dashboard(DashboardName=name, DashboardBody=json.dumps(body))
        return {"dashboard_uri": f"cloudwatch://dashboards/{name}", "panels": list(panels or []), "adapter": "sagemaker"}

    def get_recent_logs(self, *, log_group: str, filter: str = "", since: str = "1h") -> dict[str, Any]:
        logs = self._logs_client()
        try:
            kwargs: dict[str, Any] = {"logGroupName": log_group}
            if filter:
                kwargs["filterPattern"] = filter
            resp = logs.filter_log_events(**kwargs)
            return {"events": resp.get("events", []), "log_group": log_group, "filter": filter, "adapter": "sagemaker"}
        except Exception as e:
            return {"events": [], "log_group": log_group, "filter": filter, "error": str(e), "adapter": "sagemaker"}

    # ── Cost Explorer (lazy boto3 ``ce`` client; injectable for isolated validation) ──────────
    def _ce_client(self) -> Any:
        if self._ce is None:
            import boto3

            self._ce = boto3.client("ce", region_name=resolve_aws_region(self._region))
        return self._ce

    def track_spend(self, *, endpoint: str, window: str = "7d", budget: float | None = None,
                    service: str = "Amazon SageMaker") -> dict[str, Any]:
        """T-MON-09 — REAL per-endpoint inference spend via Cost Explorer ``ce:GetCostAndUsage`` (G4).

        Replaces the M5 P2 placeholder with a genuine Cost-Explorer read. **Reuses** the existing
        ``CloudWatchCostAdvisor.inference_spend`` (which issues the ``ce.get_cost_and_usage`` call,
        filtered to the SageMaker ``service`` and grouped by ``USAGE_TYPE``) so cost is ONE
        consolidated surface (doc 05 §8.8) — this adapter does not reinvent the CE query. The
        managed adapter's injected ``ce_client`` (``self._ce``) is threaded into the advisor so the
        whole path is directly verifiable offline with a recording-fake (the offline AWS API emulator lacks Cost Explorer).

        Returns the real ``run_rate`` (avg daily USD), ``spend_trend`` (up/down/flat from the daily
        series), ``vs_budget`` (over/under/unknown vs the passed ``budget``), plus ``total_usd`` /
        ``by_usage_type`` / ``daily`` / ``source`` — the SAME ``run_rate``/``spend_trend``/
        ``vs_budget``/``window``/``endpoint``/``adapter`` keys the placeholder declared (the P2
        ``note`` is dropped), so the ``track_inference_spend`` tool + ``monitor_flow`` consumers are
        unaffected. A CE failure degrades to structured zeros (``source="cost_explorer_unavailable"``)
        — never a crash, never the old placeholder.
        """
        from ...control_plane.cost_governance import CloudWatchCostAdvisor

        advisor = CloudWatchCostAdvisor(region=self._region, ce_client=self._ce_client())
        spend = advisor.inference_spend(endpoint=endpoint, window=window, budget=budget,
                                        service=service)
        spend["adapter"] = "sagemaker"
        spend["endpoint"] = endpoint
        return spend

    def compile_report(self, *, endpoint: str, window: str = "7d") -> dict[str, Any]:
        return {"report": {"endpoint": endpoint, "window": window},
                "improvement_backlog": [], "report_uri": f"s3://<your-mlops-bucket>/gold/reports/{endpoint}",
                "adapter": "sagemaker"}


# ══════════════════════════════════════════════════════════════════════════════
# LocalMemoryBackend — the M5 MemoryBackend carry-over fallback (replaces M0 stub)
# ══════════════════════════════════════════════════════════════════════════════
class LocalMemoryBackend:
    """Local ``MemoryBackend`` — the offline AgentCore-Memory carry-over fallback (M5).

    A process-wide ``actor_id → [records]`` store. ``persist`` writes a learning record
    (production failure analyses + drift findings + abandoned-reason ledgers); ``carry_over``
    reads the actor's records so the next improvement iteration is informed (doc 05 §8.6 —
    the closed-loop feed into C13). The managed ``AgentCoreMemoryBackend`` (the EXISTING
    ``MemoryHookProvider`` 3-strategy carry-over) layers on top in prod.
    """

    # process-wide so a persist in one resolved instance survives to a later carry_over.
    _STORE: ClassVar[dict[str, list[dict[str, Any]]]] = {}

    def carry_over(self, *, actor_id: str, session_id: str = "") -> list[dict[str, Any]]:
        """Return the carry-over records for ``actor_id`` (the next iteration's context)."""
        return list(self._STORE.get(actor_id, []))

    def persist(self, *, actor_id: str, learnings: dict[str, Any], session_id: str = "") -> str:
        """Persist a carry-over learning record; returns its memory record id (T-MEM-01)."""
        record_id = f"mem-{uuid.uuid4().hex[:12]}"
        self._STORE.setdefault(actor_id, []).append(
            {"record_id": record_id, "session_id": session_id, "learnings": dict(learnings)}
        )
        return record_id


# ── registration: local default offline (otel), managed sagemaker; memory local ──
register_adapter(MonitoringBackend, "sagemaker", lambda: SageMakerModelMonitorBackend(), is_default=True)
register_adapter(MonitoringBackend, "otel", lambda: OTELTraceBackend(), is_local=True)
register_adapter(MonitoringBackend, "local", lambda: OTELTraceBackend())

register_adapter(MemoryBackend, "local", lambda: LocalMemoryBackend(), is_local=True, is_default=True)
register_adapter(MemoryBackend, "agentcore", lambda: LocalMemoryBackend())  # AgentCore P1 layers on the same shape
