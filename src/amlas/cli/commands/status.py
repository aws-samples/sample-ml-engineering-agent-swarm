"""Status commands: dashboard, projects, watch, leaderboard."""

from __future__ import annotations

import time
from pathlib import Path

import click

from ..display import console, print_json_output, print_table


@click.group()
def status() -> None:
    """Real-time monitoring — dashboard, projects, watch, leaderboard."""


@status.command()
@click.option("--output-dir", default="./output", help="Output directory with project data.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def dashboard(output_dir: str, as_json: bool) -> None:
    """Show the project dashboard.

    Displays all projects with their current status and results.
    """
    from ...scheduler.dashboard import ProjectDashboard
    from ...scheduler.project_manager import ProjectManager

    manager = ProjectManager(output_dir=output_dir)
    dash = ProjectDashboard(manager)

    if as_json:
        resource_usage = dash.get_resource_usage()
        print_json_output(resource_usage)
        return

    summary = dash.generate_summary()
    console.print(summary)


@status.command()
@click.option("--output-dir", default="./output", help="Output directory with project data.")
@click.option("--status-filter", "status_filter", default=None, help="Filter by status.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def projects(output_dir: str, status_filter: str | None, as_json: bool) -> None:
    """List all projects with status."""
    from ...scheduler.project_manager import ProjectManager

    manager = ProjectManager(output_dir=output_dir)
    all_projects = manager.list_projects(status=status_filter)

    if as_json:
        data = [
            {
                "project_id": p.project_id,
                "status": p.status,
                "priority": p.priority,
                "created_at": p.created_at,
            }
            for p in all_projects
        ]
        print_json_output(data)
        return

    if not all_projects:
        console.print("[dim]No projects found.[/dim]")
        return

    rows = []
    status_icons = {
        "queued": "⏳",
        "running": "🔄",
        "completed": "✅",
        "failed": "❌",
        "paused": "⏸️",
    }
    for p in all_projects:
        icon = status_icons.get(p.status, "?")
        rows.append([
            p.project_id[:12],
            f"{icon} {p.status}",
            str(p.priority),
            p.created_at or "—",
        ])

    print_table(
        title="Projects",
        columns=[("ID", "cyan"), ("Status", ""), ("Priority", ""), ("Created", "dim")],
        rows=rows,
    )


@status.command()
@click.argument("project_id")
@click.option("--output-dir", default="./output", help="Output directory with project data.")
@click.option("--interval", default=2, help="Refresh interval in seconds.")
def watch(project_id: str, output_dir: str, interval: int) -> None:
    """Live-watch a running project.

    PROJECT_ID is the project to monitor.
    """
    from rich.live import Live
    from rich.text import Text

    from ...scheduler.project_manager import ProjectManager

    manager = ProjectManager(output_dir=output_dir)

    console.print(f"[bold]Watching project: {project_id}[/bold]")
    console.print(f"[dim]Press Ctrl+C to stop. Refreshing every {interval}s.[/dim]\n")

    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                project = manager.get_project(project_id)
                if project is None:
                    live.update(Text(f"Project '{project_id}' not found.", style="red"))
                    break

                status_text = Text()
                status_text.append(f"Project: {project_id}\n", style="bold")
                status_text.append(f"Status: {project.status}\n")
                status_text.append(f"Priority: {project.priority}\n")

                if project.status in ("completed", "failed"):
                    status_text.append(f"\nFinal status: {project.status}", style="bold green" if project.status == "completed" else "bold red")
                    live.update(status_text)
                    break

                live.update(status_text)
                # Intentionally long-lived interactive watch; user-cancellable
                # (KeyboardInterrupt handled below) — P3 audit H-06.
                time.sleep(interval)  # nosemgrep: arbitrary-sleep

    except KeyboardInterrupt:
        console.print("\n[dim]Watch stopped.[/dim]")


@status.command()
@click.option("--spec-id", default=None, help="Filter by spec ID.")
@click.option("--store", "store_path", default=None, help="Path to evaluation store JSON.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON.")
def leaderboard(spec_id: str | None, store_path: str | None, as_json: bool) -> None:
    """Show the current leaderboard.

    Displays ranked experiment results by score.
    """
    from ...evaluation.leaderboard import Leaderboard
    from ...evaluation.store import EvaluationStore

    # Find store path
    if store_path:
        s_path = Path(store_path)
    else:
        # Try common locations
        candidates = [
            Path("./output/evaluation_store.json"),
            Path("./evaluation_store.json"),
        ]
        s_path = next((c for c in candidates if c.exists()), Path("./output/evaluation_store.json"))

    store = EvaluationStore(s_path)
    lb = Leaderboard(store=store)

    if spec_id:
        ranked = lb.get_leaderboard(spec_id=spec_id)
    else:
        # Show all results
        ranked = store.get_all()
        ranked = [r for r in ranked if r.status == "success"]
        ranked.sort(key=lambda r: r.primary_score, reverse=True)

    if as_json:
        data = [
            {
                "run_id": r.run_id,
                "algorithm": r.algorithm,
                "score": r.primary_score,
                "metric": r.primary_metric,
                "spec_id": r.spec_id,
            }
            for r in ranked
        ]
        print_json_output(data)
        return

    if not ranked:
        console.print("[dim]No results in leaderboard. Run experiments first.[/dim]")
        return

    rows = []
    for i, r in enumerate(ranked[:20], 1):
        rows.append([
            str(i),
            r.algorithm,
            f"{r.primary_score:.4f}",
            r.primary_metric,
            r.spec_id,
            r.run_id[:8],
        ])

    print_table(
        title="Leaderboard",
        columns=[
            ("Rank", "bold"),
            ("Algorithm", "cyan"),
            ("Score", "green"),
            ("Metric", ""),
            ("Spec", ""),
            ("Run ID", "dim"),
        ],
        rows=rows,
    )
