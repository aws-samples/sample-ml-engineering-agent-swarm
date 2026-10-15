"""Parallel Project Scheduler — executes ML projects with concurrency control.

Uses ThreadPoolExecutor for parallelism with priority-based ordering and
graceful failure isolation between projects.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..pipeline.orchestrator import MLPipeline, PipelineResult
from .project_manager import Project, ProjectManager

logger = logging.getLogger(__name__)


class ProjectScheduler:
    """Parallel project scheduler with concurrency limits.

    Processes queued projects in priority order using a thread pool.
    Each project gets its own output directory and EvaluationStore.
    Failed projects are isolated — they don't crash other projects.
    """

    def __init__(self, manager: ProjectManager, max_concurrent: int = 3) -> None:
        self.manager = manager
        self.max_concurrent = max_concurrent
        self._stop_requested = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Reset stop flag to allow processing."""
        with self._lock:
            self._stop_requested = False

    def stop(self) -> None:
        """Request graceful stop: finish running projects, don't start new ones."""
        with self._lock:
            self._stop_requested = True

    def _is_stopped(self) -> bool:
        """Check if stop has been requested."""
        with self._lock:
            return self._stop_requested

    def run_project(self, project: Project) -> PipelineResult:
        """Execute a single project through the MLPipeline.

        Creates an isolated output directory and EvaluationStore for the project.

        Args:
            project: The Project to execute.

        Returns:
            PipelineResult from the pipeline run.
        """
        # Create project-specific output directory
        project_output_dir = Path(self.manager.output_dir) / f"project_{project.project_id}"
        project_output_dir.mkdir(parents=True, exist_ok=True)

        # Create and run pipeline
        pipeline = MLPipeline(output_dir=str(project_output_dir))
        result = pipeline.run(
            spec_dict=project.spec,
            data_path=project.data_path,
        )

        return result

    def run_all(self) -> list[Project]:
        """Run all queued projects respecting concurrency limits.

        Returns list of projects that were processed (completed or failed).
        """
        self.start()
        processed: list[Project] = []

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            futures: dict[Future[Any], str] = {}

            while True:
                if self._is_stopped():
                    break

                # Get next projects to fill available slots
                active_count = len([f for f in futures if not f.done()])
                available_slots = self.max_concurrent - active_count

                if available_slots > 0:
                    queue = self.manager.get_queue()
                    to_submit = queue[:available_slots]

                    for project in to_submit:
                        if self._is_stopped():
                            break

                        # Mark as running before submitting
                        self.manager.update_status(project.project_id, "running")

                        future = executor.submit(self._execute_project, project)
                        futures[future] = project.project_id

                # Check for completed futures
                done_futures = [f for f in futures if f.done()]
                for future in done_futures:
                    project_id = futures.pop(future)
                    try:
                        result = future.result()
                        self.manager.update_status(
                            project_id, "completed", result=result
                        )
                    except Exception as exc:
                        error_msg = str(exc)
                        self.manager.update_status(
                            project_id, "failed", error=error_msg
                        )
                        logger.warning(
                            "Project %s failed: %s", project_id, error_msg
                        )

                    processed.append(self.manager.get_project(project_id))

                # Check if there's more work to do
                remaining_queue = self.manager.get_queue()
                if not remaining_queue and not futures:
                    break

                # If nothing changed, wait for a future to complete
                if futures and not done_futures:
                    # Wait for at least one future to complete (blocking side effect);
                    # the completed future is re-collected on the next loop iteration.
                    completed_iter = as_completed(list(futures.keys()))
                    next(completed_iter)
                    continue

        # Persist state after all processing
        self.manager.save_state()
        return processed

    def _execute_project(self, project: Project) -> PipelineResult:
        """Internal wrapper for project execution (called in thread)."""
        return self.run_project(project)

    def get_active_count(self) -> int:
        """How many projects are currently running."""
        return len(self.manager.list_projects(status="running"))

    def get_progress(self) -> dict[str, int]:
        """Overall progress across all projects.

        Returns:
            Dict with counts: {queued, running, completed, failed}
        """
        all_projects = self.manager.list_projects()
        return {
            "queued": sum(1 for p in all_projects if p.status == "queued"),
            "running": sum(1 for p in all_projects if p.status == "running"),
            "completed": sum(1 for p in all_projects if p.status == "completed"),
            "failed": sum(1 for p in all_projects if p.status == "failed"),
        }
