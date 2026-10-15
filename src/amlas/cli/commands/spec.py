"""Spec management commands: create, validate, list, show."""
# Security note (CWE-22 review 2026-07): SPEC_PATH/output_path are CLI arguments
# typed by the local operator (same trust model as `cp`/`cat`); no privilege
# boundary is crossed. Agent/LLM-facing tools use amlas.tools.pathref containment.


from __future__ import annotations

from pathlib import Path

import click
import yaml

from ...spec.loader import SpecLoadError, load_spec_dict
from ..display import console, print_error, print_json_output, print_panel, print_success


@click.group()
def spec() -> None:
    """Spec management — create, validate, list, and show ML specs."""


@spec.command()
@click.argument("description")
@click.option("--data", "data_path", default=None, help="Path to CSV data file for column validation.")
@click.option("--output", "-o", "output_path", default=None, help="Output path for generated spec YAML.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def create(description: str, data_path: str | None, output_path: str | None, as_json: bool) -> None:
    """Create a spec from a natural language description.

    DESCRIPTION is a free-text description of the ML problem.

    Example:
        amlas spec create "Predict which passengers were transported. Target is 'Transported'. Need F1 > 0.75"
    """
    from ...tools.spec_generation_tools import generate_spec_from_description

    result = generate_spec_from_description(description=description, data_path=data_path)

    if result.get("error"):
        print_error(f"Spec generation issue: {result['error']}")

    if as_json:
        print_json_output(result)
        return

    # Save to file if output path specified
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.dump(result, default_flow_style=False))
        print_success(f"Spec saved to {output_path}")
    else:
        # Display the generated spec
        spec_yaml = yaml.dump(result, default_flow_style=False)
        print_panel("Generated Spec", spec_yaml, style="green")

    # Summary
    console.print(f"\n[bold]Task Type:[/bold] {result.get('task_type', 'unknown')}")
    console.print(f"[bold]Target:[/bold] {result.get('target', 'unknown')}")
    console.print(f"[bold]Metric:[/bold] {result.get('primary_metric', 'unknown')}")
    console.print(f"[bold]Threshold:[/bold] {result.get('success_threshold', 'N/A')}")


@spec.command()
@click.argument("spec_path")
def validate(spec_path: str) -> None:
    """Validate an existing spec file.

    SPEC_PATH is the path to a YAML spec file.
    """
    path = Path(spec_path)
    if not path.exists():
        print_error(f"Spec file not found: {spec_path}")
        raise SystemExit(1)

    # CLI-01: route through the frontmatter-aware loader so .md specs parse.
    try:
        data = load_spec_dict(path)
    except SpecLoadError as e:
        print_error(f"Invalid spec: {e}")
        raise SystemExit(1) from e

    if not isinstance(data, dict):
        print_error("Spec file must be a YAML dictionary")
        raise SystemExit(1)

    # Validate required fields
    required_fields = ["task_type", "target", "primary_metric"]
    missing = [f for f in required_fields if f not in data]

    if missing:
        print_error(f"Missing required fields: {', '.join(missing)}")
        for field in missing:
            console.print(f"  [red]✗[/red] {field}")
        raise SystemExit(1)

    # Validate task_type values (mirror SpecModel's accepted set)
    valid_task_types = {"classification", "regression", "segmentation", "detection"}
    if data.get("task_type") not in valid_task_types:
        print_error(f"Invalid task_type: {data.get('task_type')}. Must be one of: {valid_task_types}")
        raise SystemExit(1)

    print_success(f"Spec '{spec_path}' is valid")
    console.print(f"  Task: {data.get('task_type')}")
    console.print(f"  Target: {data.get('target')}")
    console.print(f"  Metric: {data.get('primary_metric')}")


@spec.command("list")
@click.option("--dir", "specs_dir", default="./specs", help="Directory to search for spec files.")
def list_specs(specs_dir: str) -> None:
    """List available spec files."""
    specs_path = Path(specs_dir)

    if not specs_path.exists():
        console.print(f"[dim]No specs directory at {specs_dir}[/dim]")
        return

    # CLI-02: the shipped specs are YAML front-matter + markdown body (``.md``),
    # which the rest of the spec surface (``spec show`` / ``spec validate``) parses
    # via the frontmatter-aware ``load_spec_dict``. ``spec list`` must discover the
    # SAME set — glob ``.md`` alongside legacy ``.yaml``/``.yml`` and parse each
    # through ``load_spec_dict`` (never plain ``yaml.safe_load``, which returns a
    # str for a front-matter doc and yields no fields).
    spec_files = (
        list(specs_path.glob("*.md"))
        + list(specs_path.glob("*.yaml"))
        + list(specs_path.glob("*.yml"))
    )

    if not spec_files:
        console.print("[dim]No spec files found.[/dim]")
        return

    from ..display import print_table

    rows = []
    for f in sorted(spec_files):
        try:
            data = load_spec_dict(f)
            task = data.get("task_type", "—") if isinstance(data, dict) else "—"
            target = data.get("target", "—") if isinstance(data, dict) else "—"
            metric = data.get("primary_metric", "—") if isinstance(data, dict) else "—"
        except Exception:  # a malformed spec should not abort the listing
            task = target = metric = "—"
        rows.append([f.name, task, target, metric])

    print_table(
        title="Available Specs",
        columns=[("File", "cyan"), ("Task Type", ""), ("Target", ""), ("Metric", "green")],
        rows=rows,
    )


@spec.command()
@click.argument("spec_path")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def show(spec_path: str, as_json: bool) -> None:
    """Display spec details.

    SPEC_PATH is the path to a YAML spec file.
    """
    path = Path(spec_path)
    if not path.exists():
        print_error(f"Spec file not found: {spec_path}")
        raise SystemExit(1)

    # CLI-01: route through the frontmatter-aware loader so .md specs parse.
    try:
        data = load_spec_dict(path)
    except SpecLoadError as e:
        print_error(f"Invalid spec: {e}")
        raise SystemExit(1) from e

    if as_json:
        print_json_output(data)
        return

    spec_yaml = yaml.dump(data, default_flow_style=False)
    print_panel(f"Spec: {path.name}", spec_yaml, style="cyan")
