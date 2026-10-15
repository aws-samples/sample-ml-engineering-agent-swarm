"""Results commands: show, compare, model-card, export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from ..display import (
    console,
    format_duration,
    print_error,
    print_json_output,
    print_panel,
    print_success,
    print_table,
)


@click.group()
def results() -> None:
    """Results & artifacts — show, compare, export models."""


@results.command()
@click.argument("run_id")
@click.option("--store", "store_path", default=None, help="Path to evaluation store JSON.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def show(run_id: str, store_path: str | None, as_json: bool) -> None:
    """Show detailed results for a run.

    RUN_ID is the experiment run identifier.
    """
    from ...evaluation.store import EvaluationStore

    s_path = Path(store_path) if store_path else Path("./output/evaluation_store.json")
    store = EvaluationStore(s_path)

    matches = store.get_by_run_ids([run_id])
    if not matches:
        # Try partial match
        all_results = store.get_all()
        matches = [r for r in all_results if r.run_id.startswith(run_id)]

    if not matches:
        console.print(f"[dim]Run '{run_id}' not found in store.[/dim]")
        return

    result = matches[0]

    if as_json:
        print_json_output(result.to_dict())
        return

    # Rich display
    console.print(f"\n[bold]Run: {result.run_id}[/bold]")
    console.print(f"[bold]Algorithm:[/bold] {result.algorithm}")
    console.print(f"[bold]Status:[/bold] {'[green]success[/green]' if result.status == 'success' else '[red]' + result.status + '[/red]'}")
    console.print(f"[bold]Spec:[/bold] {result.spec_id}")
    console.print(f"[bold]Duration:[/bold] {format_duration(result.duration_seconds)}")
    console.print(f"[bold]Timestamp:[/bold] {result.timestamp}")

    # Metrics
    if result.metrics:
        console.print("\n[bold]Metrics:[/bold]")
        for k, v in sorted(result.metrics.items()):
            is_primary = " ⭐" if k == result.primary_metric else ""
            console.print(f"  {k}: {v:.4f}{is_primary}")

    # Hyperparameters
    if result.hyperparameters:
        console.print("\n[bold]Hyperparameters:[/bold]")
        for k, v in sorted(result.hyperparameters.items()):
            console.print(f"  {k}: {v}")

    if result.artifact_path:
        console.print(f"\n[dim]Artifact: {result.artifact_path}[/dim]")


@results.command()
@click.argument("id1")
@click.argument("id2")
@click.option("--store", "store_path", default=None, help="Path to evaluation store JSON.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def compare(id1: str, id2: str, store_path: str | None, as_json: bool) -> None:
    """Statistical comparison of two models.

    ID1 and ID2 are run identifiers to compare.
    """
    from ...evaluation.store import EvaluationStore

    s_path = Path(store_path) if store_path else Path("./output/evaluation_store.json")
    store = EvaluationStore(s_path)

    # Find runs (support partial IDs)
    all_results = store.get_all()

    def find_run(run_id: str) -> Any:
        exact = [r for r in all_results if r.run_id == run_id]
        if exact:
            return exact[0]
        partial = [r for r in all_results if r.run_id.startswith(run_id)]
        return partial[0] if partial else None

    r1 = find_run(id1)
    r2 = find_run(id2)

    if not r1 and not r2:
        console.print(f"[dim]Neither run '{id1}' nor '{id2}' found in store.[/dim]")
        return
    if not r1:
        console.print(f"[dim]Run '{id1}' not found.[/dim]")
        return
    if not r2:
        console.print(f"[dim]Run '{id2}' not found.[/dim]")
        return

    if as_json:
        comparison = {
            "run_1": r1.to_dict(),
            "run_2": r2.to_dict(),
            "score_diff": r1.primary_score - r2.primary_score,
            "winner": r1.run_id if r1.primary_score >= r2.primary_score else r2.run_id,
        }
        print_json_output(comparison)
        return

    # Display comparison
    console.print("\n[bold]Model Comparison[/bold]\n")

    rows = [
        ["Algorithm", r1.algorithm, r2.algorithm],
        ["Primary Score", f"{r1.primary_score:.4f}", f"{r2.primary_score:.4f}"],
        ["Metric", r1.primary_metric, r2.primary_metric],
        ["Duration", format_duration(r1.duration_seconds), format_duration(r2.duration_seconds)],
        ["Status", r1.status, r2.status],
    ]

    # Add all shared metrics
    all_metrics = set(r1.metrics.keys()) | set(r2.metrics.keys())
    for metric in sorted(all_metrics):
        v1 = r1.metrics.get(metric, 0)
        v2 = r2.metrics.get(metric, 0)
        rows.append([metric, f"{v1:.4f}", f"{v2:.4f}"])

    print_table(
        title="Comparison",
        columns=[("Attribute", "bold"), (f"Run {id1[:8]}", "cyan"), (f"Run {id2[:8]}", "green")],
        rows=rows,
    )

    # Winner
    diff = r1.primary_score - r2.primary_score
    if abs(diff) < 0.001:
        console.print("\n[yellow]⚖ Results are essentially equivalent.[/yellow]")
    elif diff > 0:
        console.print(f"\n[green]🏆 Winner: {r1.algorithm} (+{diff:.4f})[/green]")
    else:
        console.print(f"\n[green]🏆 Winner: {r2.algorithm} (+{abs(diff):.4f})[/green]")


@results.command("model-card")
@click.argument("run_id")
@click.option("--store", "store_path", default=None, help="Path to evaluation store JSON.")
@click.option("--output", "-o", "output_path", default=None, help="Save model card to file.")
def model_card(run_id: str, store_path: str | None, output_path: str | None) -> None:
    """Generate a model card for a run.

    RUN_ID is the experiment run identifier.
    """
    from ...decisions.model_card import generate_model_card
    from ...evaluation.store import EvaluationStore

    s_path = Path(store_path) if store_path else Path("./output/evaluation_store.json")
    store = EvaluationStore(s_path)

    all_results = store.get_all()
    matches = [r for r in all_results if r.run_id == run_id or r.run_id.startswith(run_id)]

    if not matches:
        console.print(f"[dim]Run '{run_id}' not found.[/dim]")
        return

    result = matches[0]
    spec = {
        "project_id": result.spec_id,
        "task_type": "classification",
        "target": "target",
        "primary_metric": result.primary_metric,
        "threshold": 0.75,
        "baseline_score": 0.0,
    }

    card_md = generate_model_card(result=result, spec=spec)

    if output_path:
        Path(output_path).write_text(card_md)
        print_success(f"Model card saved to {output_path}")
    else:
        print_panel("Model Card", card_md, style="green")


@results.command()
@click.argument("run_id")
@click.argument("export_path")
@click.option("--store", "store_path", default=None, help="Path to evaluation store JSON.")
def export(run_id: str, export_path: str, store_path: str | None) -> None:
    """Export model artifact.

    RUN_ID is the run to export.
    EXPORT_PATH is the destination path.
    """
    import shutil

    from ...evaluation.store import EvaluationStore

    s_path = Path(store_path) if store_path else Path("./output/evaluation_store.json")
    store = EvaluationStore(s_path)

    all_results = store.get_all()
    matches = [r for r in all_results if r.run_id == run_id or r.run_id.startswith(run_id)]

    if not matches:
        console.print(f"[dim]Run '{run_id}' not found.[/dim]")
        return

    result = matches[0]
    artifact = Path(result.artifact_path)

    if not artifact.exists():
        print_error(f"Artifact not found at: {result.artifact_path}")
        raise SystemExit(1)

    dest = Path(export_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(artifact, dest)
    print_success(f"Exported {result.algorithm} model → {export_path}")
