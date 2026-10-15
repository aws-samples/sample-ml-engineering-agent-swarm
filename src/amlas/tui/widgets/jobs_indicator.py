"""DP-2 — JobsIndicator: a compact footer widget showing background-job counts.

Per design §3.2: a small status widget showing ``⟳ 2 running · ✓ 5 done`` driven
by the shared :class:`~amlas.tui.jobs.JobManager`. The app updates it on
every job event (marshalled onto the UI thread via ``call_from_thread`` — the same
``status_bar.py`` discipline), so finished/failed/cancelled runs surface without a
manual reload.
"""

from __future__ import annotations

from typing import Any

from textual.widgets import Static

from ..jobs import (
    JOB_CANCELLED,
    JOB_COMPLETE,
    JOB_FAILED,
    JOB_RUNNING,
    JOB_SUBMITTED,
    JobManager,
)


class JobsIndicator(Static):
    """Footer indicator that summarizes the JobManager's job counts."""

    def __init__(self, job_manager: JobManager | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._jobs = job_manager

    def attach_manager(self, job_manager: JobManager) -> None:
        """Bind a JobManager and render the current counts."""
        self._jobs = job_manager
        self.refresh_counts()

    def render_text(self) -> str:
        """Build the one-line indicator text from the manager's counts."""
        if self._jobs is None:
            return "⟳ jobs: idle"
        counts = self._jobs.counts()
        running = counts.get(JOB_RUNNING, 0) + counts.get(JOB_SUBMITTED, 0)
        done = counts.get(JOB_COMPLETE, 0)
        failed = counts.get(JOB_FAILED, 0)
        cancelled = counts.get(JOB_CANCELLED, 0)
        parts = [f"⟳ {running} running", f"✓ {done} done"]
        if failed:
            parts.append(f"✗ {failed} failed")
        if cancelled:
            parts.append(f"⊘ {cancelled} cancelled")
        return " · ".join(parts)

    def refresh_counts(self) -> None:
        """Re-render the indicator (safe to call from the UI thread)."""
        try:
            self.update(self.render_text())
        except Exception:  # nosec B110
            # Not yet mounted (e.g. in offline validation constructing the widget directly).
            pass
