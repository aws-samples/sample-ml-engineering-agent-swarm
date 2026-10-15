"""Multi-Project Scheduler package.

Provides:
  - ProjectManager: queue management with priority ordering and persistence
  - ProjectScheduler: parallel execution with concurrency control
  - ProjectDashboard: markdown reporting and cross-project comparison
  - ResearchScanScheduler: the Stage-0 scheduled/recurring research scan (gap-closure §G5)
"""

from .dashboard import ProjectDashboard
from .project_manager import Project, ProjectManager
from .research_scheduler import ResearchScanScheduler
from .scheduler import ProjectScheduler

__all__ = [
    "Project",
    "ProjectDashboard",
    "ProjectManager",
    "ProjectScheduler",
    "ResearchScanScheduler",
]
