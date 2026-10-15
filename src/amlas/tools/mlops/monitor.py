"""SageMaker Model Monitor tools (spec sheet 2.8, master plan §8 D3).

``setup_model_monitor`` creates a monitoring schedule (baseline → constraints →
scheduled drift checks of data-quality / model-quality / bias / feature-attribution
drift); ``inspect_model_monitor`` describes a schedule + surfaces the latest
violation/drift report.

``create_monitoring_schedule`` requires a live endpoint, so the real schedule
path is exercised against a REAL ephemeral endpoint (``ephemeral_monitor``,
gated ``expensive``). The cheaper-but-real drift surface — ``inspect_model_monitor``
reading the ``constraint_violations.json`` artifact SageMaker writes to the
schedule's S3 output prefix — is exercised against a REAL S3 object on a real
path (no endpoint needed). Both tools take an injectable ``sm_client`` /
``s3_client`` (real boto3 clients) — genuine call shapes, no faked responses.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from ...config.aws_region import resolve_aws_client_region
from ...config.sagemaker_images import model_monitor_image, validate_image_region
from ...errors import tool_error_boundary
from ._common import make_sm_client, sanitize_name

# Monitor types the agent may schedule (master plan §8 / research KB §1).
_MONITOR_TYPES = {"data_quality", "model_quality", "bias_drift", "feature_attribution"}


def build_monitoring_schedule_config(
    target: str,
    baseline_path: str,
    schedule: str,
    *,
    role_arn: str = "",
    output_path: str = "",
    image: str | None = None,
    monitor_type: str = "data_quality",
    region: str | None = None,
) -> dict[str, Any]:
    """Assemble a valid MonitoringScheduleConfig (pure, no AWS).

    ``schedule`` accepts a cron expression or the shorthands ``hourly`` / ``daily``
    (master plan §11 ``monitor_schedule: hourly``).
    """
    expr = _schedule_expression(schedule)
    # Squatting guard (Security review): invalid-by-construction default bucket
    # (angle brackets are illegal in S3 names) — cannot be squatted, fails loudly
    # live. Adopters must pass an explicit ``output_path``.
    out = output_path or "s3://<your-mlops-bucket>/gold/monitoring/"
    return {
        "ScheduleConfig": {"ScheduleExpression": expr},
        "MonitoringType": _monitor_type_for(monitor_type),
        "MonitoringJobDefinition": {
            "BaselineConfig": {
                "ConstraintsResource": {"S3Uri": f"{baseline_path.rstrip('/')}/constraints.json"},
                "StatisticsResource": {"S3Uri": f"{baseline_path.rstrip('/')}/statistics.json"},
            },
            "MonitoringInputs": [
                {"EndpointInput": {"EndpointName": target, "LocalPath": "/opt/ml/processing/input"}}
            ],
            "MonitoringOutputConfig": {
                "MonitoringOutputs": [
                    {"S3Output": {"S3Uri": out, "LocalPath": "/opt/ml/processing/output"}}
                ]
            },
            "MonitoringResources": {
                "ClusterConfig": {
                    "InstanceCount": 1,
                    "InstanceType": "ml.m5.xlarge",
                    "VolumeSizeInGB": 20,
                }
            },
            "MonitoringAppSpecification": {
                "ImageUri": validate_image_region(str(image or model_monitor_image(region)), region),
            },
            "RoleArn": role_arn or "arn:aws:iam::000000000000:role/AmlasMonitor",
        },
    }


@tool
@tool_error_boundary(source="tools.mlops.monitor")
def setup_model_monitor(
    target: str,
    baseline_path: str,
    schedule: str = "hourly",
    *,
    monitor_type: str = "data_quality",
    role_arn: str = "",
    output_path: str = "",
    region: str | None = None,
    sm_client: Any = None,
) -> dict[str, Any]:
    """Create a SageMaker Model Monitor schedule (master plan §8 D3).

    Establishes a baseline → constraints → scheduled drift monitoring for an
    endpoint, emitting CloudWatch violations. Drives the real
    ``create_monitoring_schedule`` API (tested against a REAL ephemeral endpoint,
    gated ``expensive`` — ``create_monitoring_schedule`` requires a live endpoint).

    Args:
        target: Endpoint name to monitor.
        baseline_path: S3 prefix holding the baseline constraints/statistics.
        schedule: ``hourly`` / ``daily`` / a cron expression.
        monitor_type: data_quality | model_quality | bias_drift | feature_attribution.
        role_arn: Execution role for the monitoring job.
        output_path: S3 prefix for violation reports.
        region: AWS region.
        sm_client: Injected SageMaker client (real-AWS-backed in tests).

    Returns:
        ``{"monitor_schedule_arn","schedule_name","schedule","monitor_type"}`` or
        ``{"success": False, "error": ...}``.
    """
    try:
        if monitor_type not in _MONITOR_TYPES:
            return {
                "success": False,
                "error": f"invalid monitor_type {monitor_type!r}; choose from {sorted(_MONITOR_TYPES)}",
            }
        name = sanitize_name(f"amlas-monitor-{target}")
        sm = make_sm_client(region, sm_client)
        selected_region = resolve_aws_client_region(sm, region)
        cfg = build_monitoring_schedule_config(
            target,
            baseline_path,
            schedule,
            role_arn=role_arn,
            output_path=output_path,
            monitor_type=monitor_type,
            region=selected_region,
        )
        resp = sm.create_monitoring_schedule(
            MonitoringScheduleName=name,
            MonitoringScheduleConfig=cfg,
        )
        return {
            "monitor_schedule_arn": resp.get("MonitoringScheduleArn", ""),
            "schedule_name": name,
            "schedule": cfg["ScheduleConfig"]["ScheduleExpression"],
            "monitor_type": monitor_type,
        }
    except Exception as e:
        return {"success": False, "error": f"setup_model_monitor failed: {e}"}


@tool
@tool_error_boundary(source="tools.mlops.monitor")
def inspect_model_monitor(
    schedule_name: str,
    *,
    region: str | None = None,
    sm_client: Any = None,
    s3_client: Any = None,
    output_uri: str = "",
) -> dict[str, Any]:
    """Describe a monitoring schedule + surface its latest drift/violation report.

    Two real paths, both genuine boto3 + real S3 (no fake):

    - **Live-schedule path** (``sm_client`` set, ``output_uri`` empty): describe the
      real schedule, then — if the last execution flagged violations — read the real
      ``constraint_violations.json`` SageMaker wrote under the schedule's S3 output
      prefix. Requires a real schedule on a live endpoint (the ``expensive`` path).
    - **Direct-S3 path** (``output_uri`` set): skip ``describe_monitoring_schedule``
      (which needs a live schedule) and read the real ``constraint_violations.json``
      directly from the given REAL S3 output prefix. This is the cheap-but-real drift
      surface used by the e2e lifecycle — no endpoint/schedule needed, just a real S3
      object on a real path.

    Args:
        schedule_name: The monitoring schedule to inspect (used for the live path
            + as the returned name label).
        region: AWS region.
        sm_client: Injected SageMaker client (real-AWS-backed). Only used by the
            live-schedule path.
        s3_client: Injected S3 client used to read the real
            ``constraint_violations.json`` drift artifact (real-AWS-backed in tests).
        output_uri: Optional REAL S3 monitoring-output prefix. When set, the tool
            reads violations directly from this prefix (cheap-but-real path).

    Returns:
        ``{"schedule_name","status","last_execution_status","violations":[...],
        "violation_count"}`` or ``{"success": False, "error": ...}``.
    """
    try:
        name = sanitize_name(schedule_name)

        # Direct-S3 path: read the real violations artifact from a real prefix
        # without a live schedule (no endpoint needed).
        if output_uri:
            violations = (
                _read_violations_from_s3(s3_client, output_uri)
                if s3_client is not None
                else []
            )
            return {
                "schedule_name": name,
                "status": "ViolationsRead",
                "last_execution_status": "CompletedWithViolations" if violations else "Completed",
                "violations": violations,
                "violation_count": len(violations),
            }

        # Live-schedule path: describe the real schedule then read its artifact.
        sm = make_sm_client(region, sm_client)
        d = sm.describe_monitoring_schedule(MonitoringScheduleName=name)
        status = d.get("MonitoringScheduleStatus", "")
        last = d.get("LastMonitoringExecutionSummary", {}) or {}
        last_status = last.get("MonitoringExecutionStatus", "")

        # Genuine Model Monitor mechanism: SageMaker writes the constraint
        # violations to ``constraint_violations.json`` under the schedule's S3
        # output prefix (DescribeMonitoringSchedule does NOT return them inline).
        # When an s3_client is supplied and the last execution flagged
        # violations, read + parse that artifact — the real drift surface.
        violations = []
        sched_output_uri = _output_uri_from_config(d.get("MonitoringScheduleConfig", {}) or {})
        if s3_client is not None and sched_output_uri and "violation" in last_status.lower():
            violations = _read_violations_from_s3(s3_client, sched_output_uri)
        return {
            "schedule_name": name,
            "status": status,
            "last_execution_status": last_status,
            "violations": violations,
            "violation_count": len(violations),
        }
    except Exception as e:
        return {"success": False, "error": f"inspect_model_monitor failed: {e}"}


def _output_uri_from_config(config: dict[str, Any]) -> str:
    """Pull the monitoring output S3 prefix from a MonitoringScheduleConfig."""
    try:
        outs = (
            config.get("MonitoringJobDefinition", {})
            .get("MonitoringOutputConfig", {})
            .get("MonitoringOutputs", [])
        )
        if outs:
            return str(outs[0].get("S3Output", {}).get("S3Uri", ""))
    except (AttributeError, IndexError, TypeError):
        pass
    return ""


def _read_violations_from_s3(s3_client: Any, output_uri: str) -> list[dict[str, Any]]:
    """Read + normalize ``constraint_violations.json`` from the monitor S3 output."""
    import json as _json

    from ...cloud.staging import parse_s3_uri

    key_uri = f"{output_uri.rstrip('/')}/constraint_violations.json"
    bucket, key = parse_s3_uri(key_uri)
    try:
        body = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    except Exception:
        return []
    doc = _json.loads(body)
    raw = doc.get("violations", []) if isinstance(doc, dict) else []
    out: list[dict[str, Any]] = []
    for v in raw:
        if isinstance(v, dict):
            out.append(
                {
                    "feature": v.get("feature_name", v.get("FeatureName", "")),
                    "type": v.get("constraint_check_type", v.get("ViolationType", "")),
                    "description": v.get("description", v.get("Description", "")),
                }
            )
    return out


def _schedule_expression(schedule: str) -> str:
    """Map ``hourly`` / ``daily`` shorthands to cron, else pass a cron through."""
    s = str(schedule).strip().lower()
    if s in ("hourly", "hour"):
        return "cron(0 * ? * * *)"
    if s in ("daily", "day"):
        return "cron(0 0 ? * * *)"
    return schedule


def _monitor_type_for(monitor_type: str) -> str:
    """Map our monitor_type tag to the SageMaker MonitoringType enum string."""
    return {
        "data_quality": "DataQuality",
        "model_quality": "ModelQuality",
        "bias_drift": "ModelBias",
        "feature_attribution": "ModelExplainability",
    }.get(monitor_type, "DataQuality")
