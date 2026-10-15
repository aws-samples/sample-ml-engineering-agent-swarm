"""Phase 5 §2a (D-1) — ``ActiveJobsPanel``: ALL concurrent jobs at a glance.

This is the ``src/amlas/tui/widgets/active_jobs.py`` widget named in
``docs/tasks/phase2-cross-context-extend-fidelity-spec.md`` §2a. The Dashboard's
existing :class:`~amlas.tui.screens.dashboard.PipelineStatusWidget` shows only the
SINGLE focused ``state.job``; this panel renders ``state.jobs`` — the multi-job
collection ``set_job`` now upserts — so N concurrent jobs (user- AND agent-launched,
running AND recently-finished) are all visible with their status.

**Data source:** ``state.jobs`` on the shared ``ContextBus`` (``job_id → ``
:class:`~amlas.tui.context.models.ActiveJob`). The panel SUBSCRIBES to the ``job``
event and repaints, sorted by ``job_id`` for deterministic rendering.

** — interactive selection (D-req #4):** in addition to the text summary
(``#active-jobs-content``, kept for the locked render tests), the panel now renders a
row-selectable :class:`~textual.widgets.DataTable` (``#active-jobs-table``,
``cursor_type == "row"``). Selecting a row publishes a dashboard ``selection`` on the
bus (``{"tab": "dashboard", "job_id": …, "kind": …}``) which drives the
:class:`~amlas.tui.widgets.dynamic_primary.DynamicPrimaryPane` to the agent-chosen,
job-appropriate view.

**Discipline (mirrors** :class:`~amlas.tui.widgets.agent_activity.AgentActivityPanel`
**exactly):** a ``set_job`` may fire from the ``@work(thread=True)`` job thread, so the
repaint marshals via ``app.call_from_thread`` with a direct-call fallback; every
``query_one`` is ``try/except``-guarded so a pre-mount / headless subscriber never
crashes. No ``@work`` thread of its own — it renders from the snapshot the ``job``
event already carries.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import DataTable, Static

from ..context import AppContextState, ContextEvent

__all__ = ["ActiveJobsPanel"]


class ActiveJobsPanel(Static):
    """Live table of ALL concurrent jobs sourced from ``state.jobs`` (§2a)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Row-index → (job_id, kind) map so a RowSelected can publish the selection
        # even though the DataTable cells are display strings ( req #4).
        self._row_jobs: dict[int, tuple[str, str]] = {}

    def compose(self) -> ComposeResult:
        with Vertical():
            # The text summary — kept as the single source of truth for the header +
            # the locked render tests (``#active-jobs-content``).
            yield Static(self._placeholder(), id="active-jobs-content")
            # : the row-selectable table driving the dynamic primary pane.
            yield DataTable(id="active-jobs-table")

    # ─── ContextBus wiring (mirrors AgentActivityPanel) ─────────────────────────

    def on_mount(self) -> None:
        """Subscribe to the bus and prime the table once. Null-safe (no bus → no-op)."""
        try:
            table = self.query_one("#active-jobs-table", DataTable)
            table.cursor_type = "row"
            if not table.columns:
                table.add_columns("Job", "Kind", "Status", "Stage", "Progress", "Started by")
        except Exception:  # nosec B110
            pass
        bus = self._bus()
        if bus is None:
            return
        try:
            bus.subscribe(self._on_context)
            self._apply_jobs(bus.state())
        except Exception:  # nosec B110
            pass

    def _bus(self) -> Any | None:
        """Return the App-owned ``ContextBus`` or ``None`` (headless / pre-mount)."""
        try:
            return getattr(self.app, "context_bus", None)
        except Exception:
            return None

    def _on_context(self, event: ContextEvent, state: AppContextState) -> None:
        """Repaint on the ``job`` event, marshalled to the UI thread. Never raises."""
        if event != "job":
            return
        try:
            app = self.app
        except Exception:
            return
        try:
            app.call_from_thread(self._apply_jobs, state)
        except Exception:
            try:
                self._apply_jobs(state)
            except Exception:  # nosec B110
                pass

    # ─── Interaction ( req #4) ─────────────────────────────────────────

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Selecting a job row publishes a dashboard ``selection`` on the bus.

        The selection (``{"tab": "dashboard", "job_id": …, "kind": …}``) is what the
        :class:`~amlas.tui.widgets.dynamic_primary.DynamicPrimaryPane` reacts to,
        swapping the primary pane to the agent-chosen view for that job type. Guarded
        + null-safe (no bus / bad row → no-op).
        """
        raw_row = getattr(event, "cursor_row", None)
        try:
            row_index = int(raw_row) if raw_row is not None else None
        except Exception:
            row_index = None
        if row_index is None:
            return
        pair = self._row_jobs.get(row_index)
        if not pair:
            return
        job_id, kind = pair
        bus = self._bus()
        if bus is None:
            return
        try:
            bus.set_selection({"tab": "dashboard", "job_id": job_id, "kind": kind})
        except Exception:  # nosec B110
            pass

    # ─── Render ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _placeholder() -> str:
        return "📦 Active Jobs\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n  No active jobs"

    def _apply_jobs(self, state: AppContextState) -> None:
        """Repaint the jobs summary + selectable table from ``state.jobs`` (guarded)."""
        jobs = state.jobs or {}
        self._apply_jobs_text(jobs)
        self._apply_jobs_table(jobs)

    def _apply_jobs_text(self, jobs: dict[str, Any]) -> None:
        """Repaint the text summary Static (kept for the locked render tests)."""
        try:
            content = self.query_one("#active-jobs-content", Static)
        except Exception:
            return

        if not jobs:
            try:
                content.update(self._placeholder())
            except Exception:  # nosec B110
                pass
            return

        lines = ["📦 Active Jobs", "━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        lines.append("Job | Kind | Status | Stage | Progress | Started by")
        for job_id in sorted(jobs):
            job = jobs[job_id]
            kind = job.kind or "—"
            status = job.status or "idle"
            stage = job.stage or "—"
            progress = f"{float(job.progress or 0.0):.0f}%"
            started_by = job.started_by or "—"
            lines.append(f"{job_id} | {kind} | {status} | {stage} | {progress} | {started_by}")

        try:
            content.update("\n".join(lines))
        except Exception:  # nosec B110
            pass

    def _apply_jobs_table(self, jobs: dict[str, Any]) -> None:
        """Repaint the selectable DataTable + rebuild the row-index → job map (guarded)."""
        try:
            table = self.query_one("#active-jobs-table", DataTable)
        except Exception:
            return
        try:
            table.clear()
        except Exception:  # nosec B110
            pass
        self._row_jobs = {}
        for idx, job_id in enumerate(sorted(jobs)):
            job = jobs[job_id]
            kind = job.kind or "—"
            status = job.status or "idle"
            stage = job.stage or "—"
            progress = f"{float(job.progress or 0.0):.0f}%"
            started_by = job.started_by or "—"
            try:
                table.add_row(
                    str(job_id), str(kind), str(status), str(stage), str(progress), str(started_by),
                    key=str(job_id),
                )
            except Exception:  # nosec B110
                pass
            self._row_jobs[idx] = (str(job_id), str(job.kind or ""))
