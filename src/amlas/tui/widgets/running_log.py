"""``RunningLogPane``: the dashboard's realtime running-log ticker.

The dashboard already has per-domain panels (Active Jobs, Agent Activity,
Observability/CloudWatch, Local Processes), but no single *consolidated*,
chronological feed of everything happening. This pane is that "running log": it
SUBSCRIBES to EVERY :class:`~amlas.tui.context.bus.ContextBus` event and appends
one bounded, newest-first activity line per mutation — so the operator sees, in
realtime, what the TUI agent + the 7 agentcore/strands agents + jobs/runs/models
+ CloudWatch-fed activity are doing, all in one place.

**Data source:** every ``ContextEvent`` on the shared bus. Unlike the domain
panels (each filters to one event type), this pane reacts to *all* of them and
renders a rolling timeline.

**Discipline (mirrors** :class:`~amlas.tui.widgets.agent_activity.AgentActivityPanel`
**/** :class:`~amlas.tui.widgets.processes.ProcessesPanel` **exactly):** a bus
mutation may fire from the ``@work(thread=True)`` trace poller / a background job
thread, so the repaint is marshalled onto the UI thread via
``app.call_from_thread`` with a direct-call fallback; every ``query_one`` is
guarded so a headless / pre-mount subscriber never crashes. Content is rendered
with ``markup=False`` and every dynamic token is passed
through :func:`~amlas.tui.markup.escape_markup`, so a bracket-shaped id / repr can
never raise a ``MarkupError`` on the render path.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from textual.app import ComposeResult
from textual.widgets import Static

from ..context import AppContextState, ContextEvent
from ..markup import escape_markup

__all__ = ["RunningLogPane"]


class RunningLogPane(Static):
    """Realtime, newest-first running log of all context-bus activity."""

    # Bounded rolling buffer — the feed never grows without limit (a long session
    # keeps only the most-recent activity, oldest lines age out).
    _MAX_ENTRIES = 200

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Newest-appended-last internally; rendered newest-first. Bounded by maxlen.
        self._entries: deque[str] = deque(maxlen=self._MAX_ENTRIES)

    def compose(self) -> ComposeResult:
        # markup=False → dynamic activity text (job ids, tool names, URIs, reprs) is
        # literal and never parsed as Rich markup.
        yield Static(self._placeholder(), id="running-log-content", markup=False)

    # ─── ContextBus wiring (mirrors AgentActivityPanel / ProcessesPanel) ─────────

    def on_mount(self) -> None:
        """Subscribe to the bus and prime from the current state. Null-safe (no bus → no-op)."""
        bus = self._bus()
        if bus is None:
            return
        try:
            bus.subscribe(self._on_context)
            # Prime: seed the feed from whatever is already active so entering the
            # Dashboard after work started shows it immediately.
            self._prime(bus.state())
        except Exception:  # nosec B110
            pass

    def _bus(self) -> Any | None:
        """Return the App-owned ``ContextBus`` or ``None`` (headless / pre-mount)."""
        try:
            return getattr(self.app, "context_bus", None)
        except Exception:
            return None

    def _on_context(self, event: ContextEvent, state: AppContextState) -> None:
        """Record every event, marshalled onto the UI thread. Never raises.

        Unlike the single-domain panels, this pane reacts to ALL event types —
        that is what makes it the consolidated running log.
        """
        try:
            app = self.app
        except Exception:
            app = None
        if app is None:
            # Headless / pre-mount: apply directly (buffer still fills for isolated validation).
            self._apply_event(event, state)
            return
        try:
            app.call_from_thread(self._apply_event, event, state)
        except Exception:
            try:
                self._apply_event(event, state)
            except Exception:  # nosec B110
                pass

    # ─── Feed construction ───────────────────────────────────────────────────────

    def _apply_event(self, event: ContextEvent, state: AppContextState) -> None:
        """Fold one bus event into a feed line and repaint (guarded pre-mount)."""
        line = self._format_event(event, state)
        if line is None:
            return  # nothing meaningful to log for this mutation
        self._entries.append(line)
        self._repaint()

    def _prime(self, state: AppContextState) -> None:
        """Seed the feed from the current snapshot (on-mount / tab re-enter)."""
        for event in ("job", "run", "model", "agents", "processes"):
            line = self._format_event(event, state)
            if line is not None:
                self._entries.append(line)
        self._repaint()

    @staticmethod
    def _stamp() -> str:
        """A short HH:MM:SS wall-clock stamp for the feed line."""
        return time.strftime("%H:%M:%S")

    def _format_event(self, event: ContextEvent, state: AppContextState) -> str | None:
        """Render one bus event as a single realtime feed line, or ``None`` to skip.

        Every dynamic token is escaped so the ``markup=False`` Static can never be
        handed a bracket-shaped token that trips the render path.
        """
        detail = self._event_detail(event, state)
        if detail is None:
            return None
        return f"{self._stamp()} · {event}: {escape_markup(detail)}"

    @staticmethod
    def _event_detail(event: ContextEvent, state: AppContextState) -> str | None:
        """Return the human detail string for an event, or ``None`` when empty/no-op."""
        if event in ("job", "selection") and state.job.job_id:
            job = state.job
            kind = f" [{job.kind}]" if job.kind else ""
            stage = f" @ {job.stage}" if job.stage else ""
            return f"{job.job_id}{kind} — {job.status} ({job.progress:.0f}%){stage}"
        if event == "run" and state.run.run_id:
            run = state.run
            status = run.status or "—"
            score = (
                f" score={run.best_score:.3f}"
                if isinstance(run.best_score, (int, float))
                else ""
            )
            return f"{run.run_id} — {status}{score}"
        if event == "model" and state.model.model_ref:
            model = state.model
            approval = f" ({model.approval})" if model.approval else ""
            endpoint = f" → {model.endpoint}" if model.endpoint else ""
            return f"{model.model_ref}{approval}{endpoint}"
        if event == "agents" and state.agents:
            # Report the most-recently-updated agent (the one this event is about).
            agent = max(state.agents.values(), key=lambda a: (a.updated_at or ""))
            tool = f" · {agent.last_tool}" if agent.last_tool else ""
            phase = f" · {agent.phase}" if agent.phase else ""
            return f"{agent.agent_name} — {agent.status}{phase}{tool}"
        if event == "processes" and state.processes:
            proc = max(state.processes.values(), key=lambda p: (p.updated_at or ""))
            detail = f" · {proc.detail}" if proc.detail else ""
            return f"{proc.process_id} [{proc.kind}] — {proc.status}{detail}"
        if event == "spec" and state.spec.path:
            return f"{state.spec.path}"
        if event == "dataset" and state.dataset.path:
            return f"{state.dataset.path}"
        if event == "steering" and state.steering is not None:
            return "steering profile updated"
        if event == "problem" and state.problem.problem_id:
            return f"{state.problem.problem_id}"
        return None

    # ─── Render ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _placeholder() -> str:
        return "📜 Running Log\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n  Waiting for activity…"

    def _render_text(self) -> str:
        """Compose the newest-first feed body (a pure string, directly verifiable headless)."""
        if not self._entries:
            return self._placeholder()
        header = ["📜 Running Log", "━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        # Newest first: reverse the append-ordered buffer.
        body = list(reversed(self._entries))
        return "\n".join(header + body)

    def _repaint(self) -> None:
        """Update the content widget from the buffer (guarded pre-mount / headless)."""
        try:
            content = self.query_one("#running-log-content", Static)
        except Exception:
            return
        try:
            content.update(self._render_text())
        except Exception:  # nosec B110
            pass

    def entries(self) -> list[str]:
        """Return a newest-first snapshot of the feed lines (test/inspection seam)."""
        return list(reversed(self._entries))
