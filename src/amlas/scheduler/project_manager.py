"""Project Manager — manages a queue of ML projects with priority ordering.

Provides:
  - Project: dataclass representing a single queued/running/completed ML project
  - ProjectManager: thread-safe project queue with persistence
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class Project:
    """A single ML project in the scheduler queue."""

    project_id: str  # UUID
    spec: dict[str, Any]
    data_path: str
    status: str  # "queued" | "running" | "completed" | "failed" | "paused"
    priority: int  # 1 (highest) to 10 (lowest)
    created_at: str  # ISO 8601
    started_at: str | None = None
    completed_at: str | None = None
    result: Any | None = None  # PipelineResult (not serialized)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON persistence (excludes result)."""
        data = {
            "project_id": self.project_id,
            "spec": self.spec,
            "data_path": self.data_path,
            "status": self.status,
            "priority": self.priority,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Project:
        """Deserialize from dictionary."""
        return cls(
            project_id=data["project_id"],
            spec=data["spec"],
            data_path=data["data_path"],
            status=data["status"],
            priority=data["priority"],
            created_at=data["created_at"],
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            result=None,  # Not persisted
            error=data.get("error"),
        )


class ProjectManager:
    """Thread-safe project queue manager with JSON persistence.

    Manages the lifecycle of ML projects: queued → running → completed/failed.
    Projects are prioritized (1=highest) and ordered FIFO within same priority.
    """

    STATE_FILENAME = "scheduler_state.json"

    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        self.projects: dict[str, Project] = {}
        self._lock = threading.Lock()

        Path(output_dir).mkdir(parents=True, exist_ok=True)

    def add_project(
        self,
        spec: dict[str, Any],
        data_path: str,
        priority: int = 5,
    ) -> str:
        """Add a project to the queue. Returns project_id (UUID)."""
        project_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()

        project = Project(
            project_id=project_id,
            spec=spec,
            data_path=data_path,
            status="queued",
            priority=priority,
            created_at=now,
        )

        with self._lock:
            self.projects[project_id] = project

        return project_id

    def remove_project(self, project_id: str) -> bool:
        """Remove a project from the queue. Returns True if removed, False if not found."""
        with self._lock:
            if project_id in self.projects:
                del self.projects[project_id]
                return True
            return False

    def get_project(self, project_id: str) -> Project:
        """Get project details by ID. Raises KeyError if not found."""
        with self._lock:
            if project_id not in self.projects:
                raise KeyError(f"Project not found: {project_id}")
            return self.projects[project_id]

    def list_projects(self, status: str | None = None) -> list[Project]:
        """List all projects, optionally filtered by status."""
        with self._lock:
            if status is None:
                return list(self.projects.values())
            return [p for p in self.projects.values() if p.status == status]

    def get_queue(self) -> list[Project]:
        """Get queued projects sorted by priority (lowest number first), then by created_at."""
        with self._lock:
            queued = [p for p in self.projects.values() if p.status == "queued"]
            return sorted(queued, key=lambda p: (p.priority, p.created_at))

    def update_status(self, project_id: str, status: str, **kwargs: Any) -> None:
        """Update project status with optional metadata.

        Keyword args:
            error: str — error message (for 'failed' status)
            result: Any — pipeline result (for 'completed' status)
        """
        now = datetime.now(UTC).isoformat()

        with self._lock:
            if project_id not in self.projects:
                raise KeyError(f"Project not found: {project_id}")

            project = self.projects[project_id]
            project.status = status

            if status == "running" and project.started_at is None:
                project.started_at = now

            if status in ("completed", "failed"):
                project.completed_at = now

            if "error" in kwargs:
                project.error = kwargs["error"]

            if "result" in kwargs:
                project.result = kwargs["result"]

    def save_state(self) -> None:
        """Persist project state to JSON file."""
        state_path = Path(self.output_dir) / self.STATE_FILENAME

        with self._lock:
            data = [p.to_dict() for p in self.projects.values()]

        # Atomic write
        tmp_path = state_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp_path.rename(state_path)

    def load_state(self) -> None:
        """Load project state from JSON file."""
        state_path = Path(self.output_dir) / self.STATE_FILENAME

        if not state_path.exists():
            return

        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            with self._lock:
                self.projects = {}
                for item in data:
                    project = Project.from_dict(item)
                    self.projects[project.project_id] = project
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # Graceful degradation — start fresh
