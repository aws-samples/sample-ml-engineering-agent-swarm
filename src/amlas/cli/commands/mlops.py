"""MLOps command group — agentic-over-SageMaker surfaces (master plan §9 / §8 D1-D7).

Drives the Phase-10 ``tools/mlops/*`` @tool functions via their ``._tool_func``:

  * ``pipeline build|run``  — SageMaker Pipelines (build the DQ→train→register DAG, run it)
  * ``register`` / ``approve`` — Model Registry (versioned package + approval gate)
  * ``monitor baseline|enable`` — Model Monitor (baseline+schedule, inspect drift)
  * ``explain`` — Clarify (bias + SHAP)
  * ``features ingest|get`` — Feature Store (ingest, point-in-time retrieval)
  * ``status`` — the §8 MLOps KPI scorecard (pure ``compute_mlops_kpis``)

The MLOps tools import boto3/sagemaker lazily and accept injectable clients, so
they are unit-tested under real AWS / a recording fake (no live AWS). Every command
supports ``--json`` for machine output, mirroring ``cli/commands/run.py``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import click

# Phase-10 MLOps tools (module scope so callers can override the dispatch layer;
# boto3/sagemaker are imported lazily inside each tool).
from ...tools.mlops import (
    approve_model_sm,
    build_sagemaker_pipeline,
    compute_mlops_kpis,
    feature_store_get,
    feature_store_ingest,
    inspect_model_monitor,
    register_model_sm,
    run_clarify,
    run_sagemaker_pipeline,
    setup_model_monitor,
)
from ..display import (
    console,
    print_error,
    print_json_output,
    print_success,
    print_table,
    print_warning,
)


@click.group()
def mlops() -> None:
    """MLOps — SageMaker pipelines, registry, monitor, explain, features, KPIs."""


# ─── mlops pipeline build|run ────────────────────────────────────────────────


@mlops.group()
def pipeline() -> None:
    """SageMaker Pipelines — build the DAG or run an execution."""


@pipeline.command("build")
@click.option("--name", default="amlas_mlops_pipeline", help="Pipeline name (sanitized to SageMaker naming).")
@click.option("--role-arn", "role_arn", default="", help="Execution role ARN.")
@click.option("--metric", "primary_metric", default="f1_macro", help="Primary metric for the QualityCheck gate.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def pipeline_build(name: str, role_arn: str, primary_metric: str, as_json: bool) -> None:
    """Build + register a SageMaker training Pipeline (D1)."""
    spec = {"name": name, "role_arn": role_arn, "primary_metric": primary_metric}
    result = build_sagemaker_pipeline._tool_func(spec)
    _emit(result, as_json, lambda r: print_success(
        f"Pipeline built: {r.get('pipeline_name')} ({r.get('steps')} steps)"
    ))


@pipeline.command("run")
@click.argument("name")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def pipeline_run(name: str, as_json: bool) -> None:
    """Start a SageMaker Pipeline execution (D1)."""
    result = run_sagemaker_pipeline._tool_func(name)
    _emit(result, as_json, lambda r: print_success(
        f"Execution started: {r.get('execution_arn')} (status: {r.get('status')})"
    ))


# ─── mlops register / approve ────────────────────────────────────────────────


@mlops.command()
@click.argument("artifact")
@click.option("--group", default="amlas_models", help="Model Package Group name.")
@click.option("--metric", "metrics", multiple=True, help="metric=value pairs (repeatable).")
@click.option("--approval", default="PendingManualApproval", help="Initial approval status.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def register(artifact: str, group: str, metrics: tuple[str, ...], approval: str, as_json: bool) -> None:
    """Register a model package into the registry (D2). ARTIFACT is an s3:// model URI."""
    metric_map = {}
    for m in metrics:
        if "=" in m:
            k, v = m.split("=", 1)
            metric_map[k] = v
    result = register_model_sm._tool_func(
        artifact=artifact, group=group, metrics=metric_map or None, approval=approval
    )
    _emit(result, as_json, lambda r: print_success(
        f"Registered {r.get('group')} v{r.get('version')} — {r.get('approval_status')}"
    ))


@mlops.command()
@click.argument("arn")
@click.option("--status", default="Approved", help="Target approval status.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def approve(arn: str, status: str, as_json: bool) -> None:
    """Transition a model package's approval status (D2). ARN is a model-package ARN."""
    result = approve_model_sm._tool_func(arn, status=status)
    _emit(result, as_json, lambda r: print_success(f"Approval status: {r.get('status')}"))


# ─── mlops monitor baseline|enable ───────────────────────────────────────────


@mlops.group()
def monitor() -> None:
    """Model Monitor — baseline+schedule, or inspect drift/violations."""


@monitor.command("baseline")
@click.argument("endpoint")
@click.argument("baseline_uri")
@click.option("--schedule", default="hourly", help="Monitor schedule (hourly/daily/cron expr).")
@click.option("--role-arn", "role_arn", default="", help="Execution role ARN.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def monitor_baseline(endpoint: str, baseline_uri: str, schedule: str, role_arn: str, as_json: bool) -> None:
    """Create a Model Monitor baseline + drift schedule (D3)."""
    result = setup_model_monitor._tool_func(
        endpoint, baseline_uri, schedule=schedule, role_arn=role_arn
    )
    _emit(result, as_json, lambda r: print_success(
        f"Monitor schedule: {r.get('monitor_schedule_arn')} ({r.get('schedule')})"
    ))


@monitor.command("enable")
@click.argument("schedule_name")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def monitor_enable(schedule_name: str, as_json: bool) -> None:
    """Inspect a Model Monitor schedule for drift/violations (D3)."""
    result = inspect_model_monitor._tool_func(schedule_name)
    _emit(result, as_json, lambda r: (
        print_success(f"Monitor {schedule_name}: {r.get('status')}")
        if r.get("violation_count", 0) == 0
        else print_warning(f"Monitor {schedule_name}: {r.get('violation_count')} violation(s)")
    ))


# ─── mlops explain (clarify) ─────────────────────────────────────────────────


@mlops.command()
@click.argument("model")
@click.argument("dataset")
@click.option("--label", required=True, help="Label column (required by Clarify).")
@click.option("--analysis", default="both", type=click.Choice(["bias", "explainability", "both"]))
@click.option("--facet", "facet_name", default=None, help="Sensitive facet column for bias analysis.")
@click.option("--role-arn", "role_arn", default="", help="Execution role ARN.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def explain(
    model: str,
    dataset: str,
    label: str,
    analysis: str,
    facet_name: str | None,
    role_arn: str,
    as_json: bool,
) -> None:
    """Run SageMaker Clarify bias + SHAP (D4). MODEL is a model name, DATASET an s3:// URI."""
    config = {"analysis": analysis, "label": label}
    if facet_name:
        config["facet_name"] = facet_name
    result = run_clarify._tool_func(model, dataset, config, role_arn=role_arn)
    _emit(result, as_json, lambda r: print_success(
        f"Clarify ({r.get('analysis')}): bias={r.get('bias_report') or 'n/a'} shap={r.get('shap_uri')}"
    ))


# ─── mlops features ingest|get ───────────────────────────────────────────────


@mlops.group()
def features() -> None:
    """Feature Store — ingest features or point-in-time retrieval."""


@features.command("ingest")
@click.argument("group")
@click.argument("data_path")
@click.option("--role-arn", "role_arn", default="", help="Execution role ARN.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def features_ingest(group: str, data_path: str, role_arn: str, as_json: bool) -> None:
    """Create a feature group + ingest rows from a CSV (D5)."""
    if not Path(data_path).exists():
        print_error(f"File not found: {data_path}")
        raise SystemExit(1)
    result = feature_store_ingest._tool_func(group, data_path, role_arn=role_arn)
    _emit(result, as_json, lambda r: print_success(
        f"Ingested {r.get('ingested')} rows into {r.get('feature_group')}"
    ))


@features.command("get")
@click.argument("group")
@click.option("--ids", multiple=True, help="Record identifier value(s) to retrieve (repeatable).")
@click.option("--point-in-time", "point_in_time", default=None, help="PIT timestamp (ISO 8601).")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def features_get(group: str, ids: tuple[str, ...], point_in_time: str | None, as_json: bool) -> None:
    """Retrieve feature records (optionally point-in-time) (D5)."""
    result = feature_store_get._tool_func(group, list(ids), point_in_time=point_in_time)
    _emit(result, as_json, lambda r: print_success(f"Retrieved {r.get('count')} record(s)."))


# ─── mlops status (KPIs) ─────────────────────────────────────────────────────


@mlops.command()
@click.option("--events", "events_path", default=None, help="Path to a JSON events file for the KPI scorecard.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def status(events_path: str | None, as_json: bool) -> None:
    """Report the §8 MLOps KPI scorecard (D7) from an events file (pure, no AWS)."""
    events: dict[str, Any] = {}
    if events_path:
        p = Path(events_path)
        if not p.exists():
            print_error(f"Events file not found: {events_path}")
            raise SystemExit(1)
        try:
            events = json.loads(p.read_text())
        except Exception as e:
            print_error(f"Failed to parse events JSON: {e}")
            raise SystemExit(1) from e

    scorecard = compute_mlops_kpis(events)

    if as_json:
        print_json_output(scorecard)
        return

    overall = scorecard.get("overall", {})
    console.print(
        f"\n[bold]MLOps KPI Scorecard[/bold] — maturity: "
        f"{overall.get('maturity_level', 'L0')}, "
        f"composite health: {overall.get('composite_health', 0.0):.2f}\n"
    )
    rows = []
    for category in ("pipeline", "registry", "monitoring", "deployment", "governance", "feature_store"):
        cat = scorecard.get(category, {})
        if cat:
            summary = ", ".join(f"{k}={v}" for k, v in list(cat.items())[:3])
            rows.append([category, summary[:70]])
    print_table(
        title="KPI Categories",
        columns=[("Category", "cyan"), ("Highlights", "dim")],
        rows=rows,
    )


# ─── shared emit helper ──────────────────────────────────────────────────────


def _emit(result: dict[str, Any], as_json: bool, on_success: Callable[[dict[str, Any]], None]) -> None:
    """Print a tool result: JSON (if requested) or a success/error human line.

    Fail-soft results (``{"success": False, "error": ...}``) print the error and
    exit non-zero; otherwise the ``on_success`` callback renders the human summary.
    """
    if as_json:
        print_json_output(result)
        return
    if result.get("success") is False:
        print_error(result.get("error", "operation failed"))
        raise SystemExit(1)
    on_success(result)
