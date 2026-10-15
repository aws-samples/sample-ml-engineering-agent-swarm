"""DP-2 background execution package.

Exposes the :class:`JobManager`, the :class:`Job` dataclass, the job-status
constants, and the pluggable executors (:class:`InlineExecutor` (default,
deterministic/headless), :class:`ThreadExecutor` (the real Textual-app path),
:class:`DeferredExecutor` (validation-only, drives jobs on demand)).

See ``docs/audit/phase7-road-to-100-design.md`` §3 for the design and
``docs/audit/phase11-background-and-routing.md`` for the implementation report.
"""

from __future__ import annotations

from .manager import (
    JOB_CANCELLED,
    JOB_COMPLETE,
    JOB_FAILED,
    JOB_RUNNING,
    JOB_SUBMITTED,
    TERMINAL_STATUSES,
    DeferredExecutor,
    InlineExecutor,
    Job,
    JobCancelled,
    JobManager,
    ThreadExecutor,
)

__all__ = [
    "JOB_CANCELLED",
    "JOB_COMPLETE",
    "JOB_FAILED",
    "JOB_RUNNING",
    "JOB_SUBMITTED",
    "TERMINAL_STATUSES",
    "DeferredExecutor",
    "InlineExecutor",
    "Job",
    "JobCancelled",
    "JobManager",
    "ThreadExecutor",
]
