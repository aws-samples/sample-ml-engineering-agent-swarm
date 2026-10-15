"""Dashboard pane — the live observability surface (Phase 4, design §E.6/§E.7/§E.8).

Phase 4 (TUI context integration) turns the Dashboard from a mount-once static
screen into a **live observability surface** driven entirely by the shared
``ContextBus`` (fixes GT-2 "never updates after mount", GT-3 "no auto-monitor",
GO-1 "no CloudWatch", GO-3/GO-4/GO-5 "no agent-activity visibility"):

* **Auto-monitor the active job (§E.6):** the pane SUBSCRIBES to the bus; on a
  ``job`` event it repaints :class:`PipelineStatusWidget` from ``state.job``
  (status / stage / progress) — replacing the static "Idle" text. Because every
  ``JobManager`` transition publishes a ``job`` event (via the app's job seam), the
  Dashboard follows the active job live; when a NEW job becomes the ``ActiveJob``
  (any tab / the agent) the display re-points to it.
* **CloudWatch + observability (§E.7):** mounts :class:`ObservabilityPanel`, which
  surfaces recent CloudWatch log lines + key metrics + inference spend for the active
  endpoint (offline-safe placeholder when no creds). Re-points on the ``model`` event.
* **Agent activity (§E.8):** mounts :class:`AgentActivityPanel`, fed by
  ``state.agents`` on the bus. A ``@work(thread=True)`` :class:`TraceTailReader`
  poller runs WHILE an ``ActiveJob`` is running (the ``SystemHealthWidget``
  discipline): it folds ``runs/<run_id>/trace.jsonl`` + ``kanban.json`` (offline) or
  the CloudWatch trace log group (cloud) into ``AgentActivity`` records and publishes
  each via ``context_bus.set_agent_activity`` — so both the swarm and the chat agent
  appear.

Everything is additive and null-safe: the Dashboard mounts cleanly headless with no
active job / no agents / no bus.

 (dynamic dashboard revamp) rearchitected the layout: the "Active Projects"
``ProjectTable`` was REMOVED, ``SystemHealthWidget`` moved to the top-left, the RIGHT
pane became the 7-agent :class:`~amlas.tui.widgets.agent_logs.AgentLogsPane`, and the
Active-Jobs rows now drive the agent-chosen
:class:`~amlas.tui.widgets.dynamic_primary.DynamicPrimaryPane` (per-job DS chart /
scores / training / data-quality view) via bus ``selection`` events.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

from textual import work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import ProgressBar, Static
from textual.worker import NoActiveWorker, get_current_worker

from ..context import ActiveJob, AppContextState, ContextEvent
from ..context.trace_reader import TraceTailReader
from ..widgets.active_jobs import ActiveJobsPanel
from ..widgets.agent_activity import AgentActivityPanel
from ..widgets.agent_logs import AgentLogsPane
from ..widgets.dynamic_primary import DynamicPrimaryPane
from ..widgets.observability import ObservabilityPanel
from ..widgets.processes import ProcessesPanel
from ..widgets.running_log import RunningLogPane
from ..widgets.status_bar import SystemHealthWidget

# JobManager statuses that mean "still working" — the trace poller only runs then.
_ACTIVE_JOB_STATUSES = frozenset({"submitted", "running"})


# The live-app default trace-poll cadence, in seconds. Surfaced in the Dashboard UI
# (:data:`DASHBOARD_REFRESH_LABEL`) so the operator can SEE how often the pane repaints.
LIVE_TRACE_POLL_INTERVAL_S: float = 1.0


def _trace_poll_interval_s() -> float:
    """Phase 5 §2b (D-4): the periodic trace-poll interval in seconds.

    Resolution:
      1. ``AMLAS_TRACE_POLL_INTERVAL_S`` when set — the on-the-fly override (read at call
         time so it stays env-overridable).
      2. When unset, :data:`LIVE_TRACE_POLL_INTERVAL_S` (``1.0``) so the
         Dashboard auto-refreshes running jobs once a second with no env export needed.
    """
    raw = os.environ.get("AMLAS_TRACE_POLL_INTERVAL_S")
    if raw is not None and raw.strip() != "":
        try:
            value = float(raw)
            return value if math.isfinite(value) else 0.0
        except Exception:
            return 0.0
    return LIVE_TRACE_POLL_INTERVAL_S


def _dashboard_refresh_label() -> str:
    """Human caption for the Dashboard's live-refresh cadence (surfaced in the pane).

    Reflects the SAME resolution :func:`_trace_poll_interval_s` uses, so what the operator
    reads matches what the poller actually does. A ``<= 0`` interval (CI single-pass) reads
    as "on demand"; a positive interval reads as e.g. "every 1.0s".
    """
    interval = _trace_poll_interval_s()
    if interval <= 0:
        return "⟳ Live refresh: on demand (single pass)"
    return f"⟳ Live refresh: every {interval:g}s"


class DashboardPane(Static):
    """Main dashboard — live overview + auto-monitor + observability + agent activity."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Phase 5 §2c (D-2/T-5): the CloudWatch log reader for cloud-session polling.
        # DI seam (like ``ObservabilityPanel.__init__`` sets ``self._log_reader``) so
        # offline validation inject a deterministic fake — NO live AWS in CI. Defaults to the
        # lazily-resolved ``get_recent_logs`` tool via :meth:`_resolve_log_reader`.
        self._log_reader: Any | None = None

    def compose(self) -> ComposeResult:
        """Build dashboard layout."""
        # : redundant per-tab title removed (tab-nav already labels the tab).
        #
        #  (dynamic dashboard revamp): the dashboard is rearchitected —
        #   1. RIGHT pane = the 7-agent logs pane (:class:`AgentLogsPane`, summaries +
        #      errors/warnings for the seven ML-automation leads) — replacing the old
        #      generic running-log feed as the right-hand surface.
        #   2. "Active Projects" (the old ``ProjectTable``) is REMOVED.
        #   3. "System Health" is moved to the TOP-LEFT of the content column.
        #   4. "Active Jobs" is kept AND made interactive — its rows are selectable and a
        #      selection publishes a dashboard ``selection`` on the bus which drives …
        #   5. … the :class:`DynamicPrimaryPane` — the agent-chosen, per-job primary view
        #      (training curve / scores / inference / data-quality) via ``choose_dynamic_view``.
        #
        #  (dedupe headers): the self-titling widgets (ActiveJobsPanel,
        # ObservabilityPanel, AgentActivityPanel, ProcessesPanel, RunningLogPane,
        # AgentLogsPane, DynamicPrimaryPane) own their header — no screen-level
        # ``.section-title`` Static for those (that would render the header twice). Only
        # PipelineStatusWidget + SystemHealthWidget (which DON'T self-title) keep a
        # screen-level ``.section-title`` so every section shows exactly one header.
        with Horizontal(classes="dashboard-split"):
            with Vertical(classes="dashboard-content"):
                # (#13 req #3) System Health at the TOP-LEFT.
                with Vertical(classes="health-section"):
                    yield Static("💚 System Health", classes="section-title")
                    yield SystemHealthWidget()
                    # Operator-visible live-refresh cadence (env AMLAS_TRACE_POLL_INTERVAL_S;
                    # live default 1s). markup=False → the caption is rendered literally.
                    yield Static(
                        _dashboard_refresh_label(),
                        id="dashboard-refresh-rate",
                        classes="dashboard-refresh-rate",
                        markup=False,
                    )
                # (#13 req #4) Active Jobs — kept, now row-selectable (drives the
                # dynamic primary pane). Self-titles "📦 Active Jobs".
                yield ActiveJobsPanel()
                # (#13 req #5) The agent-chosen, per-job dynamic primary pane. Self-titles.
                yield DynamicPrimaryPane(classes="dynamic-primary-section")
                # Phase 5 §2a (D-1): the focused single-job pipeline progress.
                with Vertical(classes="progress-section"):
                    yield Static("⚡ Pipeline Progress", classes="section-title")
                    yield PipelineStatusWidget()
                with Horizontal(classes="observability-section"):
                    with Vertical(classes="cloudwatch-section"):
                        # (#12) Title omitted — ObservabilityPanel self-titles.
                        yield ObservabilityPanel()
                    with Vertical(classes="agent-activity-section"):
                        # (#12) Title omitted — AgentActivityPanel self-titles.
                        yield AgentActivityPanel()
                # Phase 5 §2d (D-3): local processes/threads/subprocesses region.
                with Vertical(classes="processes-section"):
                    # (#12) Title omitted — ProcessesPanel self-titles.
                    yield ProcessesPanel()
                # : the consolidated realtime running-log ticker (kept; it moved
                # into the left column now that the RIGHT pane is the agent logs). Self-titles.
                with Vertical(classes="running-log-section"):
                    yield RunningLogPane()
            # (#13 req #1) RIGHT pane: the 7-agent logs pane. Self-titles "🕵️ Agent Logs".
            with Vertical(classes="agent-logs-section"):
                yield AgentLogsPane()

    # ─── ContextBus wiring (design §E.6/§E.7/§E.8) ───────────────────────────────

    def on_mount(self) -> None:
        """Subscribe to the shared ``ContextBus`` and prime from current state.

        Null-safe: with no ``app.context_bus`` (bare-pane / headless) this is a
        no-op so the pane still mounts. Priming on mount means switching TO the
        Dashboard after a job is already active shows it immediately.
        """
        bus = self._bus()
        if bus is None:
            return
        try:
            bus.subscribe(self._on_context)
            self._apply_context(bus.state())
        except Exception:  # nosec B110
            pass

    def _bus(self) -> Any | None:
        """Return the App-owned ``ContextBus`` or ``None`` (headless / pre-mount)."""
        try:
            return getattr(self.app, "context_bus", None)
        except Exception:
            return None

    def _on_context(self, event: ContextEvent, state: AppContextState) -> None:
        """Bus subscriber — marshal the re-render onto the UI thread.

        Mirrors the App's ``_on_context_event`` discipline: a ``ContextBus`` mutation
        may originate on a background job thread (via ``set_job`` in the app's job
        seam), so widget updates go through ``call_from_thread``. Only ``job`` /
        ``run`` / ``model`` / ``agents`` events matter to the Dashboard. Never raises.
        """
        if event not in ("job", "run", "model", "agents", "selection"):
            return
        try:
            app = self.app
        except Exception:
            return
        try:
            app.call_from_thread(self._apply_context_event, event, state)
        except Exception:
            try:
                self._apply_context_event(event, state)
            except Exception:  # nosec B110
                pass

    def _apply_context_event(self, event: ContextEvent, state: AppContextState) -> None:
        """UI-thread dispatch for a single context event (guarded per widget)."""
        if event in ("job", "selection"):
            self._apply_job(state.job)
            # A running active job (re)starts the agent-activity trace poller.
            if event == "job" and state.job.status in _ACTIVE_JOB_STATUSES and state.job.run_id:
                self._start_trace_poller(state.job.run_id)
        if event == "model":
            self._refresh_observability_panel()

    def _apply_context(self, state: AppContextState) -> None:
        """Prime every panel from the current snapshot (on-mount / re-enter)."""
        self._apply_job(state.job)
        if state.job.status in _ACTIVE_JOB_STATUSES and state.job.run_id:
            self._start_trace_poller(state.job.run_id)

    # ─── §E.6 auto-monitor the active job ────────────────────────────────────────

    def _apply_job(self, job: ActiveJob) -> None:
        """Drive :class:`PipelineStatusWidget` from the monitored active job."""
        try:
            widget = self.query_one(PipelineStatusWidget)
        except Exception:
            return
        try:
            widget.update_from_job(job)
        except Exception:  # nosec B110
            pass

    # ─── §E.7 observability re-point ─────────────────────────────────────────────

    def _refresh_observability_panel(self) -> None:
        """Re-point the CloudWatch panel to the (new) active endpoint (guarded)."""
        try:
            panel = self.query_one(ObservabilityPanel)
        except Exception:
            return
        try:
            panel.refresh_observability()
        except Exception:  # nosec B110
            pass

    # ─── §E.8 agent-activity trace poller ────────────────────────────────────────

    def _start_trace_poller(self, run_id: str) -> None:
        """Start the threaded trace-tail poller for ``run_id`` (idempotent, guarded).

        Only one poller runs at a time (``exclusive=True``); it exits when the job is
        no longer active. Null-safe: if there is no running app/worker infra it simply
        does nothing (headless).
        """
        self._poll_run_id = run_id
        try:
            self._run_trace_poller()
        except Exception:  # nosec B110
            pass

    def _resolve_log_reader(self) -> Any | None:
        """Return the CloudWatch log reader for cloud-session polling (§2c, injectable).

        If a test injected ``self._log_reader`` it wins (NO live AWS). Otherwise lazily
        resolve the real ``get_recent_logs`` tool via the SAME ``_resolve_tool`` pattern
        :class:`ObservabilityPanel` uses, so the Dashboard and Observability share one
        CloudWatch reader. Best-effort: any resolution failure returns ``None`` (offline
        → the cloud branch is simply skipped, null-safe).
        """
        if self._log_reader is not None:
            return self._log_reader
        try:
            from ..widgets.observability import _default_log_reader

            return _default_log_reader
        except Exception:
            return None

    @work(thread=True, exclusive=True)
    def _run_trace_poller(self) -> None:
        """Fold active-run trace spans into ``AgentActivity`` on an interval loop (§2b/§2c).

        Phase 5 upgrades the one-shot poller into a **bounded, interval-gated poll loop**
        (D-4) that ALSO polls cloud sessions by ``session_id`` (D-2/T-5):

        * **Per pass:** read the focused run's local ``runs/<run_id>/trace.jsonl`` via
          :class:`TraceTailReader` AND, for every ``state.agents`` record that carries a
          ``session_id`` and is still ``running``, read the durable CloudWatch trace log
          group filtered by that ``session_id`` (injected ``log_reader`` — NO live AWS in
          CI). Each folded record is published via ``set_agent_activity`` (the ``agents``
          event the panel renders); publishes happen off the UI thread and the panel
          marshals its own repaint.
        * **Loop/stop:** continue only while a job is active — any
          ``state.jobs``/``state.job`` with ``status in _ACTIVE_JOB_STATUSES``. When no
          job is active the loop exits (D-4 "stop when terminal").
        * **Interval:** :func:`_trace_poll_interval_s`. ``<= 0`` (test/CI default) ⇒
          EXACTLY ONE pass then exit — byte-identical to today's one-shot semantics, so
          the existing single-pass battery stays green. A positive value sleeps that many
          seconds between passes (``time.sleep`` inside this ``@work(thread=True)`` body,
          NOT the event loop — the ``ObservabilityPanel._run_refresh_worker`` discipline).
          ``exclusive=True`` means a new active job cancels the prior loop cleanly.
        """
        run_id = getattr(self, "_poll_run_id", None)
        if not run_id:
            return
        bus = self._bus()
        if bus is None:
            return

        interval = _trace_poll_interval_s()
        try:
            worker = get_current_worker()
        except NoActiveWorker:
            worker = None  # Direct/headless callers retain the one-pass contract.
        while worker is None or not worker.is_cancelled:
            self._poll_once(bus, run_id)
            # Stop when no job is active (terminal) — D-4 exit condition.
            if not self._any_job_active(bus):
                return
            # interval <= 0 ⇒ exactly one pass (deterministic CI); else sleep + loop.
            if interval <= 0:
                return
            try:
                if worker is not None:
                    if worker.cancelled_event.wait(interval):
                        return
                else:
                    time.sleep(interval)
            except Exception:
                return

    def _any_job_active(self, bus: Any) -> bool:
        """True while any job (focused ``job`` or any ``jobs``) is still working (§2b)."""
        try:
            state = bus.state()
        except Exception:
            return False
        jobs = list(state.jobs.values()) if state.jobs else []
        if state.job and state.job.job_id:
            jobs.append(state.job)
        return any((j.status in _ACTIVE_JOB_STATUSES) for j in jobs)

    def _poll_once(self, bus: Any, run_id: str) -> None:
        """One poll pass: fold the local run trace + every live cloud session (§2b/§2c)."""
        # Local trace for the focused run.
        try:
            reader = TraceTailReader(run_id, session_id=None, log_reader=None)
            for activity in reader.read():
                try:
                    bus.set_agent_activity(activity)
                except Exception:  # nosec B110
                    pass
        except Exception:  # nosec B110
            pass

        # Cloud sessions: any running agent record carrying a session_id (§2c / T-5).
        log_reader = self._resolve_log_reader()
        if log_reader is None:
            return
        try:
            agents = list(bus.state().agents.values())
        except Exception:
            return
        seen_sessions: set[str] = set()
        for agent in agents:
            sid = getattr(agent, "session_id", None)
            status = getattr(agent, "status", None)
            if not sid or status not in _ACTIVE_JOB_STATUSES or sid in seen_sessions:
                continue
            seen_sessions.add(sid)
            try:
                cloud_reader = TraceTailReader(agent.run_id or run_id, session_id=sid, log_reader=log_reader)
                for activity in cloud_reader.read():
                    try:
                        bus.set_agent_activity(activity)
                    except Exception:  # nosec B110
                        pass
            except Exception:  # nosec B110
                pass


class PipelineStatusWidget(Static):
    """Shows the monitored active job's live progress (Phase 4, §E.6).

    Replaces the previous static "Stage: Idle" text: :meth:`update_from_job` drives the
    stage label, the progress bar, and the status text from the bus's ``ActiveJob``.
    With no active job it renders the null-safe idle fallback.
    """

    def compose(self) -> ComposeResult:
        yield Static("Stage: Idle", id="pipeline-stage-label")
        yield ProgressBar(total=100, show_eta=True, id="pipeline-progress")
        yield Static("No active pipeline run", id="pipeline-status-text")

    def update_from_job(self, job: ActiveJob) -> None:
        """Repaint the stage label / progress / status text from ``job`` (guarded).

        Null-safe: an empty ``ActiveJob`` (no ``job_id``) renders the idle fallback so
        the Dashboard mounts cleanly with nothing running.
        """
        has_job = bool(getattr(job, "job_id", None))
        stage = job.stage or ("Idle" if not has_job else "—")
        try:
            label = self.query_one("#pipeline-stage-label", Static)
            label.update(f"Stage: {stage}")
        except Exception:  # nosec B110
            pass

        try:
            progress = self.query_one("#pipeline-progress", ProgressBar)
            progress.update(progress=float(job.progress or 0.0))
        except Exception:  # nosec B110
            pass

        try:
            status_text = self.query_one("#pipeline-status-text", Static)
            if not has_job:
                status_text.update("No active pipeline run")
            else:
                kind = f" [{job.kind}]" if job.kind else ""
                started = f" · via {job.started_by}" if job.started_by else ""
                status_text.update(f"Job `{job.job_id}`{kind} — {job.status} ({job.progress:.0f}%){started}")
        except Exception:  # nosec B110
            pass
