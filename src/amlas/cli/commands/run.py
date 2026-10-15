"""Run commands: pipeline, train, tune, experiment."""

from __future__ import annotations

import json
import time
from pathlib import Path

import click

from ...config.aws_region import resolve_aws_region
from ...spec.loader import SpecLoadError, load_spec_dict
from ..display import (
    console,
    create_progress,
    format_duration,
    print_error,
    print_json_output,
    print_success,
    print_table,
)


@click.group()
def run() -> None:
    """Pipeline execution — run pipelines, train models, tune hyperparameters."""


@run.command()
@click.argument("spec_path")
@click.argument("data_path")
@click.option("--output", "-o", "output_dir", default="./output", help="Output directory for results.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def pipeline(spec_path: str, data_path: str, output_dir: str, as_json: bool) -> None:
    """Run the full ML pipeline (spec → research → train → evaluate → decide).

    SPEC_PATH is the path to a YAML spec file.
    DATA_PATH is the path to a CSV data file.
    """
    spec_file = Path(spec_path)
    data_file = Path(data_path)

    if not spec_file.exists():
        print_error(f"Spec file not found: {spec_path}")
        raise SystemExit(1)
    if not data_file.exists():
        print_error(f"Data file not found: {data_path}")
        raise SystemExit(1)

    try:
        spec_data = load_spec_dict(spec_file)
    except SpecLoadError as e:
        print_error(f"Invalid spec: {e}")
        raise SystemExit(1) from e

    from ...pipeline.orchestrator import MLPipeline

    console.print("\n[bold blue]🚀 Running ML Pipeline[/bold blue]\n")

    with create_progress() as progress:
        task = progress.add_task("Pipeline execution", total=6)

        pipe = MLPipeline(output_dir=output_dir)
        progress.update(task, advance=1, description="Initialized pipeline")

        result = pipe.run(spec_dict=spec_data, data_path=data_path)
        progress.update(task, advance=5, description="Pipeline complete")

    if as_json:
        summary = {
            "spec_id": spec_data.get("project_id", "unknown"),
            "recommendation": result.decision.recommendation,
            "confidence": result.decision.confidence,
            "num_experiments": len(result.results),
            "duration": result.total_duration_seconds,
            "artifacts_dir": result.artifacts_dir,
        }
        print_json_output(summary)
        return

    # Display results
    console.print(f"\n[bold green]✓ Pipeline Complete[/bold green] ({format_duration(result.total_duration_seconds)})\n")

    # Decision summary
    decision = result.decision
    rec_color = "green" if decision.recommendation == "deploy" else ("yellow" if "iterate" in decision.recommendation else "red")
    console.print(f"[bold]Decision:[/bold] [{rec_color}]{decision.recommendation}[/{rec_color}]")
    console.print(f"[bold]Confidence:[/bold] {decision.confidence}")
    console.print(f"[bold]Risk Level:[/bold] {decision.risk_level}")
    console.print(f"[bold]Rationale:[/bold] {decision.rationale}")

    # Experiments summary
    successful = [r for r in result.results if r.status == "success"]
    console.print(f"\n[bold]Experiments:[/bold] {len(successful)}/{len(result.results)} successful")

    if successful:
        best = max(successful, key=lambda r: r.primary_score)
        console.print(f"[bold]Best Model:[/bold] {best.algorithm} ({best.primary_metric}={best.primary_score:.4f})")

    console.print(f"\n[dim]Artifacts saved to: {result.artifacts_dir}[/dim]")


@run.command()
@click.option("--algorithm", "-a", required=True, help="Algorithm name (e.g., xgboost, random_forest).")
@click.option("--data", "data_path", required=True, help="Path to CSV data file.")
@click.option("--target", required=True, help="Target column name.")
@click.option("--output", "-o", "output_dir", default="./output", help="Output directory.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def train(algorithm: str, data_path: str, target: str, output_dir: str, as_json: bool) -> None:
    """Train a single model.

    Trains the specified algorithm on the given dataset.
    """
    data_file = Path(data_path)
    if not data_file.exists():
        print_error(f"Data file not found: {data_path}")
        raise SystemExit(1)

    from ...tools.training_tools import train_model

    console.print(f"\n[bold blue]Training {algorithm}...[/bold blue]\n")

    start = time.time()
    result = train_model(
        data_path=data_path,
        target_column=target,
        algorithm=algorithm,
        output_dir=output_dir,
    )
    duration = time.time() - start

    if as_json:
        result["duration_seconds"] = duration
        print_json_output(result)
        return

    if result.get("success"):
        print_success(f"Training complete ({format_duration(duration)})")
        metrics = result.get("metrics", {})
        if metrics:
            console.print("\n[bold]Metrics:[/bold]")
            for k, v in metrics.items():
                console.print(f"  {k}: {v:.4f}")
        if result.get("model_path"):
            console.print(f"\n[dim]Model saved to: {result['model_path']}[/dim]")
    else:
        print_error(f"Training failed: {result.get('error', 'Unknown error')}")
        raise SystemExit(1)


@run.command()
@click.option("--algorithm", "-a", required=True, help="Algorithm to tune.")
@click.option("--data", "data_path", required=True, help="Path to CSV data file.")
@click.option("--target", required=True, help="Target column name.")
@click.option("--task-type", default="classification", type=click.Choice(["classification", "regression"]))
@click.option("--trials", default=50, help="Number of Optuna trials.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def tune(algorithm: str, data_path: str, target: str, task_type: str, trials: int, as_json: bool) -> None:
    """Run Optuna HPO for an algorithm.

    Uses Optuna to find optimal hyperparameters.
    """
    data_file = Path(data_path)
    if not data_file.exists():
        print_error(f"Data file not found: {data_path}")
        raise SystemExit(1)

    from ...tools.hpo_tools import tune_model

    console.print(f"\n[bold blue]Tuning {algorithm} ({trials} trials)...[/bold blue]\n")

    result = tune_model(
        algorithm=algorithm,
        data_path=data_path,
        target_column=target,
        task_type=task_type,
        n_trials=trials,
    )

    if as_json:
        print_json_output(result)
        return

    if result.get("success"):
        print_success(f"HPO complete — best score: {result.get('best_score', 0):.4f}")
        best_params = result.get("best_params", {})
        if best_params:
            console.print("\n[bold]Best Hyperparameters:[/bold]")
            for k, v in best_params.items():
                console.print(f"  {k}: {v}")
    else:
        print_error(f"HPO failed: {result.get('error', 'Unknown error')}")


@run.command()
@click.argument("spec_path")
@click.argument("data_path")
@click.option("--output", "-o", "output_dir", default="./output", help="Output directory.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def experiment(spec_path: str, data_path: str, output_dir: str, as_json: bool) -> None:
    """Run a multi-algorithm experiment.

    SPEC_PATH is the path to a YAML spec file.
    DATA_PATH is the path to a CSV data file.
    """
    spec_file = Path(spec_path)
    data_file = Path(data_path)

    if not spec_file.exists():
        print_error(f"Spec file not found: {spec_path}")
        raise SystemExit(1)
    if not data_file.exists():
        print_error(f"Data file not found: {data_path}")
        raise SystemExit(1)

    try:
        spec_data = load_spec_dict(spec_file)
    except SpecLoadError as e:
        print_error(f"Invalid spec: {e}")
        raise SystemExit(1) from e

    from ...evaluation.store import EvaluationStore
    from ...experiments.runner import ExperimentRunner
    from ...tools.research_tools import recommend_approaches

    console.print("\n[bold blue]🧪 Running Experiment[/bold blue]\n")

    # Get recommendations
    recommendations = recommend_approaches(spec_json=json.dumps(spec_data))

    # Setup store and runner
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    store = EvaluationStore(out_path / "evaluation_store.json")
    runner = ExperimentRunner(store=store)

    # Create plan from recommendations
    plan_spec = {
        "spec_id": spec_data.get("project_id", "experiment"),
        "task_type": spec_data.get("task_type", "classification"),
        "target": spec_data.get("target", "target"),
        "primary_metric": spec_data.get("primary_metric", "f1_macro"),
        "data_path": data_path,
    }
    plan = runner.create_plan_from_recommendations(plan_spec, recommendations)

    # Run experiments
    results = runner.run_plan(plan)

    if as_json:
        summary = {
            "total_experiments": len(results),
            "successful": sum(1 for r in results if r.status == "success"),
            "results": [
                {
                    "algorithm": r.algorithm,
                    "status": r.status,
                    "primary_score": r.primary_score,
                    "duration": r.duration_seconds,
                }
                for r in results
            ],
        }
        print_json_output(summary)
        return

    # Display results
    successful = [r for r in results if r.status == "success"]
    console.print("\n[bold green]✓ Experiment Complete[/bold green]")
    console.print(f"  {len(successful)}/{len(results)} experiments successful\n")

    if successful:
        rows = []
        for r in sorted(successful, key=lambda x: x.primary_score, reverse=True):
            rows.append([
                r.algorithm,
                f"{r.primary_score:.4f}",
                format_duration(r.duration_seconds),
                r.run_id[:8],
            ])

        print_table(
            title="Experiment Results",
            columns=[("Algorithm", "cyan"), ("Score", "green"), ("Duration", ""), ("Run ID", "dim")],
            rows=rows,
        )


@run.command()
@click.argument("description")
@click.option("--dataset", "dataset_path", required=True, help="Local dataset CSV to stage to the session bucket.")
@click.option("--target", required=True, help="Target column name (e.g., failure_mode).")
@click.option("--project-id", "project_id", default="amlas_cloud_run", help="Project id (sanitized to [a-zA-Z0-9_]).")
@click.option("--metric", "primary_metric", default=None, help="Override the inferred primary metric.")
@click.option("--max-rows", "max_rows", default=None, type=int, help="Cap total rows for a cheap verification slice.")
@click.option("--max-retreats", "max_retreats", default=0, type=int, help="Iteration retreats budget (default 0).")
@click.option("--trials", "trials", default=1, type=int, help="Optuna trials cap (default 1 for smoke).")
@click.option("--wall-clock-minutes", "wall_clock_minutes", default=5, type=int, help="Wall-clock budget.")
@click.option(
    "--acceptance-delta",
    "acceptance_delta",
    default=None,
    help='Override acceptance delta-vs-baseline (e.g. ">= 0.15"). STRICTER bar forces the retreat loop.',
)
@click.option(
    "--no-subgroup-regression",
    "no_subgroup_regression",
    is_flag=True,
    default=False,
    help="Also reject any per-class/subgroup regression vs baseline (stricter acceptance).",
)
@click.option("--runtime-arn", "runtime_arn", default=None, help="Override the deployed AgentCore runtime ARN.")
@click.option("--session-bucket", "session_bucket", default=None, help="Override the session S3 bucket.")
@click.option("--region", "region", default=None, help="AWS region (uses config, environment or selected profile).")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output streamed events as JSON.")
def cloud(
    description: str,
    dataset_path: str,
    target: str,
    project_id: str,
    primary_metric: str | None,
    max_rows: int | None,
    max_retreats: int,
    trials: int,
    wall_clock_minutes: int,
    acceptance_delta: str | None,
    no_subgroup_regression: bool,
    runtime_arn: str | None,
    session_bucket: str | None,
    region: str | None,
    as_json: bool,
) -> None:
    """Dispatch a REAL experiment to the deployed AgentCore runtime (B1/B4/B5).

    DESCRIPTION is a natural-language problem statement. The client:
      1. uploads DATASET to the session S3 bucket,
      2. turns DESCRIPTION into a schema-valid task_spec.md (golden_dataset = the
         uploaded object),
      3. calls InvokeAgentRuntime on the deployed runtime with stage_data=true so
         the runtime stages data + materializes y_true server-side (N1/N2),
      4. streams the agentic orchestrator's events to completion.

    SEC-01: the runtime ARN / session bucket / region are NOT hardcoded. They
    resolve in priority order: CLI flag > env (AGENTCORE_RUNTIME_ARN /
    AMLAS_SESSION_BUCKET) > .agenticml.yaml config. When still unset,
    run_cloud_experiment fails fast with a clear ValueError (no live default).

    Example:
      agentcli run cloud "Predict the AI4I failure mode; optimize F1 macro" \\
        --dataset data/ai4i/ai4i2020.csv --target failure_mode \\
        --project-id p14b_ai4i --max-rows 200
    """
    data_file = Path(dataset_path)
    if not data_file.exists():
        print_error(f"Dataset file not found: {dataset_path}")
        raise SystemExit(1)

    # Deferred import — keeps the CLI importable without boto3/agents at load time.
    from ...cloud.dispatch import (
        DEFAULT_RUNTIME_ARN,
        DEFAULT_SESSION_BUCKET,
        DispatchConfig,
        resolve_runtime_arn,
        resolve_session_bucket,
        run_cloud_experiment,
    )
    from ...config.settings import Settings

    # SEC-01 resolution: CLI flag > env (DEFAULT_*) > SSM (/amlas/*) > .agenticml.yaml.
    # FIX-13 (H8 family): the SSM step resolves the LIVE deploy-published runtime
    # ARN + session bucket so a plain ``amlas run cloud`` works right after deploy
    # with NO manual env export (matrix §5 step 4). Empty after all four → the
    # run_cloud_experiment fail-fast ValueError still fires (never a live default).
    settings = Settings.load()
    resolved_region = resolve_aws_region(region or settings.aws.region, profile_name=settings.aws.effective_profile)
    resolved_arn = (
        resolve_runtime_arn(runtime_arn or DEFAULT_RUNTIME_ARN, region=resolved_region)
        or settings.aws.agentcore_runtime_arn
    )
    resolved_bucket = (
        resolve_session_bucket(session_bucket or DEFAULT_SESSION_BUCKET, region=resolved_region)
        or settings.aws.effective_session_bucket
    )

    cfg = DispatchConfig(
        runtime_arn=resolved_arn,
        session_bucket=resolved_bucket,
        region=resolved_region,
        max_rows=max_rows,
        max_retreats=max_retreats,
        trials=trials,
        wall_clock_minutes=wall_clock_minutes,
        acceptance_delta=acceptance_delta,
        no_subgroup_regression=no_subgroup_regression,
    )

    console.print("\n[bold blue]☁️  Dispatching cloud experiment to AgentCore[/bold blue]\n")
    console.print(f"[dim]Runtime:[/dim] {cfg.runtime_arn or '(unset — will fail fast)'}")
    console.print(f"[dim]Session bucket:[/dim] {cfg.session_bucket or '(unset — will fail fast)'}")

    try:
        result = run_cloud_experiment(
            description,
            dataset_path=dataset_path,
            project_id=project_id,
            target=target,
            config=cfg,
            primary_metric=primary_metric,
        )
    except ValueError as e:
        # SEC-01 fail-fast: missing runtime ARN / session bucket.
        print_error(str(e))
        raise SystemExit(1) from e

    if as_json:
        print_json_output(
            {
                "session_id": result.session_id,
                "dataset_uri": result.dataset_uri,
                "status": result.status,
                "reached_training": result.reached_training,
                "events": result.events,
            }
        )
        return

    console.print(f"\n[dim]Uploaded dataset:[/dim] {result.dataset_uri}")
    console.print(f"[dim]Session id:[/dim] {result.session_id}\n")
    # Cloud-run events are STATUS/metadata records emitted by our own runner
    # (run ids, stage names, timings) — not user dataset content. Keep it that
    # way: never add raw data samples to these event payloads.
    for ev in result.events:
        kind = ev.get("event", "?")
        console.print(f"  [cyan]{kind}[/cyan]: {json.dumps({k: v for k, v in ev.items() if k != 'event'})}")

    status = result.status or "unknown"
    color = "green" if status == "accepted" else ("yellow" if status == "budget_exhausted" else "red")
    console.print(f"\n[bold]Run status:[/bold] [{color}]{status}[/{color}]")
    if result.reached_training:
        console.print("[green]✓ Reached training/eval (data staged, run started)[/green]")
