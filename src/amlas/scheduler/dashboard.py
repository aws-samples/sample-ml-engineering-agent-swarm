"""Project Dashboard — generates markdown summaries and cross-project comparisons.

Provides real-time visibility into the multi-project scheduler state.
"""

from __future__ import annotations

from typing import Any

from .project_manager import ProjectManager

# Status emoji mapping
_STATUS_ICONS = {
    "queued": "⏳ Queued",
    "running": "🔄 Running",
    "completed": "✅ Completed",
    "failed": "❌ Failed",
    "paused": "⏸️ Paused",
}


class ProjectDashboard:
    """Generates markdown dashboards for multi-project scheduler state."""

    def __init__(self, manager: ProjectManager) -> None:
        self.manager = manager

    def generate_summary(self) -> str:
        """Generate markdown dashboard summary table.

        Shows all projects with status, score, algorithm, and duration.
        """
        all_projects = self.manager.list_projects()

        if not all_projects:
            return "# Project Dashboard\n\nNo projects in the queue."

        # Sort: running first, then queued, then completed, then failed
        status_order = {"running": 0, "queued": 1, "completed": 2, "failed": 3, "paused": 4}
        sorted_projects = sorted(
            all_projects,
            key=lambda p: (status_order.get(p.status, 5), p.priority, p.created_at),
        )

        lines = [
            "# Project Dashboard",
            "",
            f"**Total**: {len(all_projects)} projects",
            "",
            "| # | Project | Status | Score | Algorithm | Duration |",
            "|---|---------|--------|-------|-----------|----------|",
        ]

        for idx, project in enumerate(sorted_projects, 1):
            project_name = project.spec.get("project_id", project.project_id[:8])
            status_display = _STATUS_ICONS.get(project.status, project.status)
            score = "—"
            algorithm = "—"
            duration = "—"

            if project.result is not None:
                try:
                    dur_seconds = project.result.total_duration_seconds
                    if dur_seconds >= 60:
                        minutes = int(dur_seconds // 60)
                        seconds = int(dur_seconds % 60)
                        duration = f"{minutes}m {seconds}s"
                    else:
                        duration = f"{dur_seconds:.1f}s"
                except (AttributeError, TypeError):
                    pass

                try:
                    results = project.result.results
                    if results:
                        best = max(results, key=lambda r: getattr(r, "primary_score", 0))
                        score_val = getattr(best, "primary_score", None)
                        if score_val is not None:
                            metric_name = project.spec.get("primary_metric", "score")
                            score = f"{metric_name}={score_val:.2f}"
                        algo = getattr(best, "algorithm", None)
                        if algo:
                            algorithm = algo
                except (AttributeError, TypeError):
                    pass

            elif project.status == "running" and project.started_at:
                duration = "in progress..."

            lines.append(
                f"| {idx} | {project_name} | {status_display} | {score} | {algorithm} | {duration} |"
            )

        # Summary stats
        completed = sum(1 for p in all_projects if p.status == "completed")
        running = sum(1 for p in all_projects if p.status == "running")
        queued = sum(1 for p in all_projects if p.status == "queued")
        failed = sum(1 for p in all_projects if p.status == "failed")

        lines.extend([
            "",
            "---",
            f"**Progress**: {completed}/{len(all_projects)} completed | "  # nosemgrep
            f"{running} running | {queued} queued | {failed} failed",
        ])

        return "\n".join(lines)

    def generate_comparison(self) -> str:
        """Cross-project comparison of completed results.

        Compares metrics, algorithms, and durations across all completed projects.
        """
        completed = self.manager.list_projects(status="completed")

        if not completed:
            return "# Cross-Project Comparison\n\nNo completed projects to compare."

        lines = [
            "# Cross-Project Comparison",
            "",
            "| Project | Best Score | Algorithm | Duration | Confidence |",
            "|---------|-----------|-----------|----------|------------|",
        ]

        for project in completed:
            project_name = project.spec.get("project_id", project.project_id[:8])
            score = "—"
            algorithm = "—"
            duration = "—"
            confidence = "—"

            if project.result is not None:
                try:
                    dur_seconds = project.result.total_duration_seconds
                    if dur_seconds >= 60:
                        minutes = int(dur_seconds // 60)
                        seconds = int(dur_seconds % 60)
                        duration = f"{minutes}m {seconds}s"
                    else:
                        duration = f"{dur_seconds:.1f}s"
                except (AttributeError, TypeError):
                    pass

                try:
                    results = project.result.results
                    if results:
                        best = max(results, key=lambda r: getattr(r, "primary_score", 0))
                        score_val = getattr(best, "primary_score", None)
                        if score_val is not None:
                            score = f"{score_val:.4f}"
                        algo = getattr(best, "algorithm", None)
                        if algo:
                            algorithm = algo
                except (AttributeError, TypeError):
                    pass

                try:
                    confidence_val = project.result.decision.confidence
                    confidence = f"{confidence_val:.2f}"
                except (AttributeError, TypeError):
                    pass

            lines.append(
                f"| {project_name} | {score} | {algorithm} | {duration} | {confidence} |"
            )

        return "\n".join(lines)

    def get_resource_usage(self) -> dict[str, Any]:
        """Estimate resource usage across all projects.

        Returns:
            Dict with total duration, completed count, and per-project breakdown.
        """
        all_projects = self.manager.list_projects()
        completed = [p for p in all_projects if p.status == "completed"]

        total_duration = 0.0
        per_project: list[dict[str, Any]] = []

        for project in completed:
            dur = 0.0
            if project.result is not None:
                try:
                    dur = float(project.result.total_duration_seconds)
                except (AttributeError, TypeError, ValueError):
                    pass
            total_duration += dur
            per_project.append({
                "project_id": project.spec.get("project_id", project.project_id[:8]),
                "duration_seconds": dur,
            })

        return {
            "total_duration_seconds": total_duration,
            "completed_count": len(completed),
            "queued_count": sum(1 for p in all_projects if p.status == "queued"),
            "running_count": sum(1 for p in all_projects if p.status == "running"),
            "failed_count": sum(1 for p in all_projects if p.status == "failed"),
            "per_project": per_project,
        }
