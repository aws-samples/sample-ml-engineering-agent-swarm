"""Research commands: recommend and algorithms."""

from __future__ import annotations

import json
from pathlib import Path

import click

from ...spec.loader import SpecLoadError, load_spec_dict
from ..display import console, print_error, print_json_output, print_panel, print_table


@click.group()
def research() -> None:
    """Research and algorithm discovery."""


@research.command()
@click.argument("spec_path")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def recommend(spec_path: str, as_json: bool) -> None:
    """Get algorithm recommendations for a spec.

    SPEC_PATH is the path to a YAML spec file.
    """
    path = Path(spec_path)
    if not path.exists():
        print_error(f"Spec file not found: {spec_path}")
        raise SystemExit(1)

    # CLI-01: route through the frontmatter-aware loader so .md specs parse.
    try:
        spec_data = load_spec_dict(path)
    except SpecLoadError as e:
        print_error(f"Invalid spec: {e}")
        raise SystemExit(1) from e

    from ...tools.research_tools import recommend_approaches

    result = recommend_approaches(spec_json=json.dumps(spec_data))

    if as_json:
        print_json_output(result)
        return

    # Display recommendations
    console.print("\n[bold]Algorithm Recommendations[/bold]\n")

    algos = result.get("recommended_algorithms", [])
    if algos:
        rows = []
        for algo in algos:
            rows.append([
                str(algo.get("rank", "—")),
                algo.get("name", "unknown"),
                algo.get("sklearn_class", "—"),
                algo.get("rationale", "")[:50],
            ])

        print_table(
            title="Recommended Algorithms",
            columns=[("Rank", "bold"), ("Algorithm", "cyan"), ("Class", "green"), ("Rationale", "dim")],
            rows=rows,
        )

    # Suggested approach
    approach = result.get("suggested_approach", "")
    if approach:
        print_panel("Suggested Approach", approach, style="green")

    # Experiment suggestions
    suggestions = result.get("experiment_suggestions", [])
    if suggestions:
        console.print("\n[bold]Experiment Suggestions:[/bold]")
        for i, s in enumerate(suggestions, 1):
            console.print(f"  {i}. {s}")

@research.command()
@click.option("--task-type", default="classification", type=click.Choice(["classification", "regression"]))
@click.option("--data-type", default="tabular", help="Data modality.")
@click.option("--data-size", default="medium", type=click.Choice(["small", "medium", "large"]))
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def algorithms(task_type: str, data_type: str, data_size: str, as_json: bool) -> None:
    """List algorithms for a given task type."""
    from ...tools.research_tools import search_algorithms

    result = search_algorithms(task_type=task_type, data_type=data_type, data_size=data_size)

    if as_json:
        print_json_output(result)
        return

    algos = result.get("algorithms", [])
    if not algos:
        console.print("[dim]No algorithms found.[/dim]")
        return

    console.print(f"\n[bold]Algorithms for {task_type} ({data_type}, {data_size})[/bold]\n")

    rows = []
    for algo in algos:
        strengths = ", ".join(algo.get("strengths", [])[:2])
        rows.append([
            str(algo.get("rank", "—")),
            algo.get("display_name", algo.get("name", "")),
            algo.get("category", "—"),
            algo.get("complexity", "—"),
            strengths[:40],
        ])

    print_table(
        title=f"Ranked Algorithms ({result.get('total_found', 0)} found)",
        columns=[
            ("Rank", "bold"),
            ("Algorithm", "cyan"),
            ("Category", ""),
            ("Complexity", "yellow"),
            ("Strengths", "green"),
        ],
        rows=rows,
    )
