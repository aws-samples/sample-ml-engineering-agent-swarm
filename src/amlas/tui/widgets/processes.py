"""Phase 5 §2d (D-3) — ``ProcessesPanel``: local processes/threads/subprocesses.

This is the ``src/amlas/tui/widgets/processes.py`` widget named in
``docs/tasks/phase2-cross-context-extend-fidelity-spec.md`` §2d. It surfaces an
OS/subprocess/thread inventory DISTINCT from an ``ActiveJob`` — a local training
subprocess, an Optuna HPO thread, a SageMaker submit-poll thread — so the operator
sees WHAT is running locally, not just the monitored job.

**Data source:** ``state.processes`` on the shared ``ContextBus`` (``process_id → ``
:class:`~amlas.tui.context.models.ProcessActivity`), populated by the new
``ContextBus.set_process_activity`` mutator (the ``processes`` event). Producers: the
app's ``_on_job_event`` for local ThreadExecutor jobs, or any tool via
``get_active_bus().set_process_activity(...)``.

**Discipline (mirrors** :class:`~amlas.tui.widgets.agent_activity.AgentActivityPanel`
**exactly):** subscribes to the ``processes`` event; marshals the repaint via
``app.call_from_thread`` with a direct-call fallback; every ``query_one`` is guarded so
a pre-mount / headless subscriber never crashes.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.widgets import Static

from ..context import AppContextState, ContextEvent

__all__ = ["ProcessesPanel"]


class ProcessesPanel(Static):
    """Live table of local processes sourced from ``state.processes`` (§2d)."""

    def compose(self) -> ComposeResult:
        yield Static(self._placeholder(), id="processes-content")

    # ─── ContextBus wiring (mirrors AgentActivityPanel) ─────────────────────────

    def on_mount(self) -> None:
        """Subscribe to the bus and prime the table once. Null-safe (no bus → no-op)."""
        bus = self._bus()
        if bus is None:
            return
        try:
            bus.subscribe(self._on_context)
            self._apply_processes(bus.state())
        except Exception:  # nosec B110
            pass

    def _bus(self) -> Any | None:
        """Return the App-owned ``ContextBus`` or ``None`` (headless / pre-mount)."""
        try:
            return getattr(self.app, "context_bus", None)
        except Exception:
            return None

    def _on_context(self, event: ContextEvent, state: AppContextState) -> None:
        """Repaint on the ``processes`` event, marshalled to the UI thread. Never raises."""
        if event != "processes":
            return
        try:
            app = self.app
        except Exception:
            return
        try:
            app.call_from_thread(self._apply_processes, state)
        except Exception:
            try:
                self._apply_processes(state)
            except Exception:  # nosec B110
                pass

    # ─── Render ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _placeholder() -> str:
        return "🧵 Local Processes\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n  No local processes"

    def _apply_processes(self, state: AppContextState) -> None:
        """Repaint the processes table from ``state.processes`` (guarded pre-mount)."""
        processes = state.processes or {}
        try:
            content = self.query_one("#processes-content", Static)
        except Exception:
            return

        if not processes:
            try:
                content.update(self._placeholder())
            except Exception:  # nosec B110
                pass
            return

        lines = ["🧵 Local Processes", "━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        lines.append("Process | Kind | Status | Started by | Detail")
        for process_id in sorted(processes):
            proc = processes[process_id]
            kind = proc.kind or "—"
            status = proc.status or "running"
            started_by = proc.started_by or "—"
            detail = proc.detail or "—"
            lines.append(f"{process_id} | {kind} | {status} | {started_by} | {detail}")

        try:
            content.update("\n".join(lines))
        except Exception:  # nosec B110
            pass
