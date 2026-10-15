"""Scheduler Tools — @tool-decorated functions for agent access to the scheduler.

These tools provide the interface for LLM agents to interact with the
multi-project scheduler via tool-calling.
"""

from __future__ import annotations

import json
import os
from typing import Any

from strands import tool

from ..errors import tool_error_boundary
from ..scheduler.dashboard import ProjectDashboard
from ..scheduler.project_manager import ProjectManager
from ..scheduler.scheduler import ProjectScheduler

# ─── Singleton Manager ────────────────────────────────────────────────────────

_MANAGER: ProjectManager | None = None


def _get_manager() -> ProjectManager:
    """Get or create the global ProjectManager singleton."""
    global _MANAGER
    if _MANAGER is None:
        output_dir = os.environ.get("SCHEDULER_OUTPUT_DIR", "./output/scheduler")
        _MANAGER = ProjectManager(output_dir=output_dir)
        _MANAGER.load_state()
    return _MANAGER


# ─── Tools ────────────────────────────────────────────────────────────────────


@tool
@tool_error_boundary(source="tools.scheduler_tools")
def submit_project(spec_yaml: str, data_path: str, priority: int = 5) -> dict[str, Any]:
    """Submit a new ML project to the queue.

    Args:
        spec_yaml: JSON string of the spec dict (project_id, task_type, target, etc.)
        data_path: Path to the training data file.
        priority: Priority level 1 (highest) to 10 (lowest). Default 5.

    Returns:
        Dict with project_id and queue_position.
    """
    manager = _get_manager()

    # Parse spec from JSON string
    try:
        spec = json.loads(spec_yaml)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid spec JSON ({type(e).__name__} at parse)"}

    project_id = manager.add_project(spec=spec, data_path=data_path, priority=priority)
    manager.save_state()

    # Determine queue position
    queue = manager.get_queue()
    position = next(
        (i + 1 for i, p in enumerate(queue) if p.project_id == project_id),
        len(queue),
    )

    return {
        "project_id": project_id,
        "queue_position": position,
        "status": "queued",
        "message": f"Project submitted successfully at position {position}.",
    }


@tool
@tool_error_boundary(source="tools.scheduler_tools")
def get_project_status(project_id: str) -> dict[str, Any]:
    """Get current status of a project.

    Args:
        project_id: UUID of the project.

    Returns:
        Dict with project details and status.
    """
    manager = _get_manager()

    try:
        project = manager.get_project(project_id)
    except KeyError:
        return {"error": f"Project not found: {project_id}"}

    return {
        "project_id": project.project_id,
        "status": project.status,
        "priority": project.priority,
        "created_at": project.created_at,
        "started_at": project.started_at,
        "completed_at": project.completed_at,
        "error": project.error,
        "spec": project.spec,
    }


@tool
@tool_error_boundary(source="tools.scheduler_tools")
def list_all_projects() -> dict[str, Any]:
    """List all projects with their status.

    Returns:
        Dict with total count and list of project summaries.
    """
    manager = _get_manager()
    all_projects = manager.list_projects()

    projects_summary = []
    for p in all_projects:
        projects_summary.append({
            "project_id": p.project_id,
            "name": p.spec.get("project_id", "unknown"),
            "status": p.status,
            "priority": p.priority,
            "created_at": p.created_at,
        })

    return {
        "total": len(projects_summary),
        "projects": projects_summary,
    }


@tool
@tool_error_boundary(source="tools.scheduler_tools")
def get_dashboard() -> str:
    """Generate project dashboard markdown.

    Returns:
        Markdown string with the full project dashboard.
    """
    manager = _get_manager()
    dashboard = ProjectDashboard(manager=manager)
    return dashboard.generate_summary()


@tool
@tool_error_boundary(source="tools.scheduler_tools")
def run_project_queue(max_concurrent: int = 3) -> dict[str, Any]:
    """Execute all queued projects (up to max_concurrent in parallel).

    Args:
        max_concurrent: Maximum number of projects to run simultaneously.

    Returns:
        Dict with execution summary.
    """
    manager = _get_manager()
    scheduler = ProjectScheduler(manager=manager, max_concurrent=max_concurrent)

    completed = scheduler.run_all()

    return {
        "processed_count": len(completed),
        "completed": sum(1 for p in completed if p.status == "completed"),
        "failed": sum(1 for p in completed if p.status == "failed"),
        "remaining_queued": len(manager.get_queue()),
        "message": f"Processed {len(completed)} projects.",
    }
