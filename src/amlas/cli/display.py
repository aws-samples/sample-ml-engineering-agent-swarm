"""Rich display helpers for CLI output.

Provides consistent formatting for tables, panels, progress bars, and status displays.
"""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.table import Table

# Shared console instance
console = Console()
error_console = Console(stderr=True)


def print_success(message: str) -> None:
    """Print a success message with green checkmark."""
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str) -> None:
    """Print an error message with red X."""
    error_console.print(f"[red]✗[/red] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[yellow]⚠[/yellow] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[blue]ℹ[/blue] {message}")


def print_json_output(data: Any) -> None:
    """Print data as formatted JSON (for --json flag).

    Uses click.echo (not rich) to avoid ANSI escape codes in JSON output.

    Security note: callers must pass RESULT/metadata dicts only — never raw
    configuration containing secrets. Secrets are resolved via Secrets Manager
    at runtime and are never part of the dicts routed here.
    """
    import click
    click.echo(json.dumps(data, indent=2, default=str))


def print_panel(title: str, content: str, style: str = "blue") -> None:
    """Print content in a bordered panel."""
    console.print(Panel(content, title=title, border_style=style))


def print_yaml(content: str) -> None:
    """Print YAML with syntax highlighting."""
    syntax = Syntax(content, "yaml", theme="monokai", line_numbers=False)
    console.print(syntax)


def print_table(
    title: str,
    columns: list[tuple[str, str]],
    rows: list[list[str]],
    show_lines: bool = False,
) -> None:
    """Print a rich table.

    Args:
        title: Table title.
        columns: List of (name, style) tuples for column headers.
        rows: List of row data (list of strings per row).
        show_lines: Whether to show row separator lines.
    """
    table = Table(title=title, show_lines=show_lines)
    for col_name, col_style in columns:
        table.add_column(col_name, style=col_style)
    for row in rows:
        table.add_row(*row)
    console.print(table)


def create_progress() -> Progress:
    """Create a progress bar context manager."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m {secs}s"


def format_score(score: float, metric: str = "") -> str:
    """Format a metric score with appropriate coloring."""
    score_str = f"{score:.4f}"
    if metric:
        return f"{metric}={score_str}"
    return score_str
