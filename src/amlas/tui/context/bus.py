"""Phase 2 — ``ContextBus``: the App-owned shared-context pub/sub (Textual-free).

This is the ``src/amlas/tui/context/bus.py`` module named in the design
(``docs/tui-context-integration-design.md`` §E.1 / §E.2 / §E.3 / §F). It is the
**literal generalization** of the two cross-component patterns already proven in
this codebase (design §A.6):

* App-attribute session state — ``app.job_manager`` / ``app.session_steering``; and
* the observer callback marshalled by ``App.call_from_thread`` —
  ``JobManager.on_event`` (``src/amlas/tui/jobs/manager.py:321``).

``ContextBus`` holds the single ``AppContextState`` record (see :mod:`.models`) and
exposes typed mutators that each publish **exactly one** typed ``ContextEvent`` to a
LIST of subscribers (multi-subscriber, unlike JobManager's single ``on_event`` — but a
raising sink never breaks the others, identical to ``JobManager._emit``). It is
guarded by an ``RLock`` so a background job thread can call ``set_job(...)`` and the app
marshals the UI refresh via ``call_from_thread`` (design §E.4).

**Additive / backward-compatible (design §E.3):** an optional ``engine.context`` dict
can be attached via :meth:`ContextBus.attach_engine_context`; ``set_spec`` / ``set_dataset``
then mirror into ``engine.context["current_spec"]`` / ``["data_summary"]`` so the existing
``_build_context`` and DataPane reader keep working unchanged. The bus is therefore
purely additive — the 500+ existing callers stay green.

**Agent read seam (design §E.4):** a process-local handle to the active bus
(:func:`set_active_bus` / :func:`get_active_bus`) lets the read-only ``get_tui_context``
``@tool`` return the live snapshot; ``None`` (headless / no app) → the tool reports
``{"available": false}``.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any

from .models import (
    ActiveDataset,
    ActiveJob,
    ActiveModel,
    ActiveProblem,
    ActiveRun,
    ActiveSpec,
    ActiveView,
    AgentActivity,
    AppContextState,
    ProcessActivity,
)

logger = logging.getLogger(__name__)

# The typed event names a mutator publishes (design §E.2; Phase 5 §2d adds "processes";
# plan §B2 adds "view" — the only event that carries a RENDER REQUEST rather than state).
ContextEvent = str  # "spec"|"dataset"|"job"|"run"|"model"|"problem"|"steering"|"agents"|"selection"|"processes"|"view"

Subscriber = Callable[[ContextEvent, AppContextState], None]

__all__ = [
    "ContextBus",
    "ContextEvent",
    "get_active_bus",
    "get_active_ledger",
    "set_active_bus",
    "set_active_ledger",
]


# ─── Process-local active-bus handle (the agent-tool read seam, design §E.4) ────
#
# The app sets this at mount so the headless ``get_tui_context`` @tool — which the
# Strands agent calls in its own thread — can reach the live context WITHOUT a
# reference to the App. ``None`` means "no app / headless" → the tool degrades to
# ``{"available": false}``. Guarded by a lock so set/get is race-free.

_active_bus_lock = threading.RLock()
_active_bus: ContextBus | None = None

# ─── Process-local active-ledger handle (Phase C2, design §D.5) ─────────────────
#
# The App-owned :class:`~amlas.tui.context.session_ledger.SessionLedger` mirror of the
# active-bus handle above: it lets the read-only ``get_tui_context`` @tool merge a
# bounded ``"session"`` timeline into its snapshot WITHOUT a reference to the App.
# ``None`` (headless / no app) → the tool omits the ``"session"`` key (byte-identical to
# pre-C2). Typed ``Any`` to avoid importing ``session_ledger`` (keeps ``bus.py`` cheap
# and free of an import cycle — the ledger imports nothing from the bus at module scope).

_active_ledger_lock = threading.RLock()
_active_ledger: Any | None = None


def set_active_bus(bus: ContextBus | None) -> None:
    """Register (or clear) the process-local active ``ContextBus`` for the agent tool."""
    global _active_bus
    with _active_bus_lock:
        _active_bus = bus


def get_active_bus() -> ContextBus | None:
    """Return the process-local active ``ContextBus`` (or ``None`` when headless)."""
    with _active_bus_lock:
        return _active_bus


def set_active_ledger(ledger: Any | None) -> None:
    """Register (or clear) the process-local active ``SessionLedger`` (design §D.5)."""
    global _active_ledger
    with _active_ledger_lock:
        _active_ledger = ledger


def get_active_ledger() -> Any | None:
    """Return the process-local active ``SessionLedger`` (or ``None`` when headless)."""
    with _active_ledger_lock:
        return _active_ledger


class ContextBus:
    """Holds the shared ``AppContextState`` + an in-process pub/sub (JobManager parity).

    Typed mutators (``set_spec``/``set_dataset``/``set_job``/…) each mutate the single
    record under an ``RLock`` and publish exactly one ``ContextEvent`` to every
    subscriber. :meth:`state` hands out a deep copy so a subscriber can never mutate the
    authoritative record. An optional attached ``engine.context`` dict is mirrored on
    spec/dataset writes so legacy consumers keep working (design §E.3).
    """

    def __init__(self, on_event: Subscriber | None = None) -> None:
        self._state = AppContextState()
        self._lock = threading.RLock()
        self._subscribers: list[Subscriber] = []
        # Optional legacy mirror target (ChatEngine.context) — attached by the engine.
        self._engine_context: dict[str, Any] | None = None
        # ``on_event`` is registered as the FIRST subscriber (JobManager ergonomics:
        # ``ContextBus(on_event=self._on_context_event)`` mirrors ``JobManager(on_event=…)``).
        if on_event is not None:
            self._subscribers.append(on_event)

    # ─── Snapshot reads ────────────────────────────────────────────────────────

    def state(self) -> AppContextState:
        """Return an immutable (deep-copied) snapshot of the current context.

        The copy means a subscriber (or the agent tool) can read/inspect freely
        without ever mutating the bus's authoritative record.
        """
        with self._lock:
            return _copy_state(self._state)

    def snapshot_dict(self) -> dict[str, Any]:
        """Return the snapshot as nested plain data with an ``available: true`` marker.

        This is exactly what the ``get_tui_context`` @tool returns (design §E.4).
        """
        snap = self.state().to_dict()
        snap["available"] = True
        return snap

    # ─── Legacy engine.context mirror (design §E.3, additive/null-safe) ─────────

    def attach_engine_context(self, engine_context: dict[str, Any]) -> None:
        """Attach a ``ChatEngine.context`` dict to mirror spec/dataset writes into.

        Additive: after this, ``set_spec``/``set_dataset`` also write
        ``engine.context["current_spec"]`` / ``["data_summary"]`` so ``_build_context``
        and the DataPane reader stay green during migration.
        """
        with self._lock:
            self._engine_context = engine_context

    # ─── Typed mutators (each publishes exactly one ContextEvent) ───────────────

    def set_spec(
        self,
        *,
        path: str | None = None,
        spec_dict: dict[str, Any] | None = None,
        project_id: str | None = None,
        task_type: str | None = None,
        target: str | None = None,
        primary_metric: str | None = None,
        golden_dataset: str | None = None,
        valid: bool | None = None,
        source: str = "",
    ) -> None:
        """Set the active spec and publish a ``"spec"`` event (mirrors current_spec)."""
        with self._lock:
            self._state.spec = ActiveSpec(
                path=path,
                spec_dict=dict(spec_dict) if spec_dict else {},
                project_id=project_id,
                task_type=task_type,
                target=target,
                primary_metric=primary_metric,
                golden_dataset=golden_dataset,
                valid=valid,
                source=source,
            )
            if self._engine_context is not None and path is not None:
                self._engine_context["current_spec"] = path
        self._emit("spec")

    def set_dataset(
        self,
        *,
        path: str | None = None,
        rows: int | None = None,
        cols: int | None = None,
        columns: list[str] | None = None,
        dtypes: dict[str, str] | None = None,
        dq_score: float | None = None,
        source: str = "",
    ) -> None:
        """Set the active dataset and publish a ``"dataset"`` event (mirrors data_summary)."""
        with self._lock:
            self._state.dataset = ActiveDataset(
                path=path,
                rows=rows,
                cols=cols,
                columns=list(columns) if columns else [],
                dtypes=dict(dtypes) if dtypes else {},
                dq_score=dq_score,
                source=source,
            )
            if self._engine_context is not None and path is not None:
                self._engine_context["data_summary"] = {
                    "path": path,
                    "rows": rows,
                    "cols": cols,
                    "columns": list(columns) if columns else [],
                    "dtypes": dict(dtypes) if dtypes else {},
                }
        self._emit("dataset")

    def set_job(
        self,
        *,
        job_id: str | None = None,
        kind: str | None = None,
        status: str = "idle",
        progress: float = 0.0,
        stage: str = "",
        run_id: str | None = None,
        started_by: str = "",
    ) -> None:
        """Set the monitored job and publish a ``"job"`` event (mirrors JobManager.Job).

        Phase 5 §2a (D-1): in addition to writing the SINGLE focused ``self._state.job``
        (the back-compat alias every existing ``set_job`` test asserts), UPSERT the same
        ``ActiveJob`` into ``self._state.jobs[job_id]`` when ``job_id`` is truthy so the
        Dashboard's Active-Jobs panel shows N concurrent jobs. A terminal status updates
        the row IN PLACE (it is NOT removed — the console shows recently-finished jobs;
        pruning is a render concern). Still exactly ONE ``"job"`` event is published
        (invariant locked by ``test_each_mutator_publishes_exactly_one_typed_event``).
        """
        with self._lock:
            job = ActiveJob(
                job_id=job_id,
                kind=kind,
                status=status,
                progress=progress,
                stage=stage,
                run_id=run_id,
                started_by=started_by,
            )
            self._state.job = job
            if job_id:
                self._state.jobs[job_id] = job
        self._emit("job")

    def set_run(
        self,
        *,
        run_id: str | None = None,
        spec_id: str | None = None,
        best_algorithm: str | None = None,
        primary_metric: str | None = None,
        best_score: float | None = None,
        decision: str | None = None,
        artifacts_dir: str | None = None,
        status: str | None = None,
    ) -> None:
        """Set the active (completed) run and publish a ``"run"`` event.

        Phase B1 (additive): ``status`` ("complete" | "partial" | "failed") lets the
        Results/Dashboard tabs distinguish a successful run from a failed one instead
        of a stale placeholder. Still exactly ONE ``"run"`` event is published
        (invariant locked by ``test_each_mutator_publishes_exactly_one_typed_event``).
        """
        with self._lock:
            self._state.run = ActiveRun(
                run_id=run_id,
                spec_id=spec_id,
                best_algorithm=best_algorithm,
                primary_metric=primary_metric,
                best_score=best_score,
                decision=decision,
                artifacts_dir=artifacts_dir,
                status=status,
            )
        self._emit("run")

    def set_model(
        self,
        *,
        model_ref: str | None = None,
        version: str | None = None,
        approval: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        """Set the active model and publish a ``"model"`` event."""
        with self._lock:
            self._state.model = ActiveModel(
                model_ref=model_ref,
                version=version,
                approval=approval,
                endpoint=endpoint,
            )
        self._emit("model")

    def set_problem(
        self,
        *,
        problem_id: str | None = None,
        kpi: str | None = None,
        problem_card_path: str | None = None,
    ) -> None:
        """Set the active problem and publish a ``"problem"`` event."""
        with self._lock:
            self._state.problem = ActiveProblem(
                problem_id=problem_id,
                kpi=kpi,
                problem_card_path=problem_card_path,
            )
        self._emit("problem")

    def set_steering(self, profile: Any) -> None:
        """Set the session steering profile and publish a ``"steering"`` event."""
        with self._lock:
            self._state.steering = profile
        self._emit("steering")

    def set_agent_activity(self, activity: AgentActivity) -> None:
        """Upsert one agent's activity by ``agent_name`` and publish an ``"agents"`` event."""
        with self._lock:
            self._state.agents[activity.agent_name] = activity
        self._emit("agents")

    def set_process_activity(self, activity: ProcessActivity) -> None:
        """Upsert one local process by ``process_id`` and publish a ``"processes"`` event.

        Phase 5 §2d (D-3): byte-identical shape to :meth:`set_agent_activity` — the only
        NEW event this spec introduces. A local subprocess/thread/HPO/poll registers
        itself here (the ``_on_job_event`` job seam for ThreadExecutor jobs, or any tool
        via ``get_active_bus().set_process_activity(...)``). ``RLock``-guarded; the
        ``ProcessesPanel`` marshals its own repaint on the ``"processes"`` event.
        """
        with self._lock:
            self._state.processes[activity.process_id] = activity
        self._emit("processes")

    def set_selection(self, selection: dict[str, str]) -> None:
        """Set the transient UI selection and publish a ``"selection"`` event."""
        with self._lock:
            self._state.selection = dict(selection)
        self._emit("selection")

    def set_view(self, view: ActiveView) -> None:
        """Set the REQUESTED dashboard view and publish a ``"view"`` event (plan §B2).

        The one mutator that carries an instruction rather than state: it is how the agent
        says "render the correlation matrix as a heat map" and how
        :class:`~amlas.tui.widgets.dynamic_primary.DynamicPrimaryPane` learns to swap its
        content. Same discipline as every other mutator — one lock, exactly one event.
        """
        with self._lock:
            self._state.view = view
        self._emit("view")

    def clear(self, field: ContextEvent | None = None) -> None:
        """Reset one field (publishing its event) or the whole record (publishing all).

        ``clear("spec")`` resets only the active spec; ``clear()`` resets everything.
        """
        if field is None:
            with self._lock:
                self._state = AppContextState()
            # Publish every field event so subscribers fully re-render (Phase 5 §2d adds
            # "processes"; the ``jobs`` collection resets with the "job" event since it
            # has no separate event — it is a superset write inside ``set_job``).
            for ev in (
                "spec", "dataset", "job", "run", "model", "problem",
                "steering", "agents", "selection", "processes", "view",
            ):
                self._emit(ev)
            return

        with self._lock:
            if field == "spec":
                self._state.spec = ActiveSpec()
            elif field == "dataset":
                self._state.dataset = ActiveDataset()
            elif field == "job":
                self._state.job = ActiveJob()
                self._state.jobs = {}
            elif field == "run":
                self._state.run = ActiveRun()
            elif field == "model":
                self._state.model = ActiveModel()
            elif field == "problem":
                self._state.problem = ActiveProblem()
            elif field == "steering":
                self._state.steering = None
            elif field == "agents":
                self._state.agents = {}
            elif field == "processes":
                self._state.processes = {}
            elif field == "selection":
                self._state.selection = {}
            elif field == "view":
                self._state.view = ActiveView()
        self._emit(field)

    # ─── Pub/sub (LIST of sinks; a raising sink never breaks others) ────────────

    def subscribe(self, fn: Subscriber) -> None:
        """Register a subscriber ``fn(event, state)`` fired on every mutation."""
        with self._lock:
            self._subscribers.append(fn)

    def _emit(self, event: ContextEvent) -> None:
        """Notify every subscriber with ``(event, snapshot)``; swallow raising sinks.

        Identical discipline to ``JobManager._emit`` (``manager.py:321``): a sink that
        raises is logged and skipped so one bad subscriber never breaks the others or
        the mutation that triggered the event.
        """
        with self._lock:
            subscribers = list(self._subscribers)
        snapshot = self.state()
        for fn in subscribers:
            try:
                fn(event, snapshot)
            except Exception as exc:
                logger.warning("ContextBus subscriber raised on %r: %s", event, exc)


# ─── Deep-copy helper for immutable snapshots ───────────────────────────────────


def _copy_state(state: AppContextState) -> AppContextState:
    """Return a deep copy of the record's mutable fields (steering kept by reference).

    ``steering`` is a ``SteeringProfile`` (or ``None``) shared by reference — it is
    treated as an immutable handle, exactly like ``engine.steering_profile`` is today.
    """
    return AppContextState(
        spec=ActiveSpec(
            path=state.spec.path,
            spec_dict=dict(state.spec.spec_dict),
            project_id=state.spec.project_id,
            task_type=state.spec.task_type,
            target=state.spec.target,
            primary_metric=state.spec.primary_metric,
            golden_dataset=state.spec.golden_dataset,
            valid=state.spec.valid,
            source=state.spec.source,
        ),
        dataset=ActiveDataset(
            path=state.dataset.path,
            rows=state.dataset.rows,
            cols=state.dataset.cols,
            columns=list(state.dataset.columns),
            dtypes=dict(state.dataset.dtypes),
            dq_score=state.dataset.dq_score,
            source=state.dataset.source,
        ),
        job=ActiveJob(
            job_id=state.job.job_id,
            kind=state.job.kind,
            status=state.job.status,
            progress=state.job.progress,
            stage=state.job.stage,
            run_id=state.job.run_id,
            started_by=state.job.started_by,
        ),
        # Phase 5 §2a: deep-copy the multi-job collection so the snapshot stays
        # immutable (same per-field copy pattern as ``agents``).
        jobs={
            jid: ActiveJob(
                job_id=job.job_id,
                kind=job.kind,
                status=job.status,
                progress=job.progress,
                stage=job.stage,
                run_id=job.run_id,
                started_by=job.started_by,
            )
            for jid, job in state.jobs.items()
        },
        run=ActiveRun(
            run_id=state.run.run_id,
            spec_id=state.run.spec_id,
            best_algorithm=state.run.best_algorithm,
            primary_metric=state.run.primary_metric,
            best_score=state.run.best_score,
            decision=state.run.decision,
            artifacts_dir=state.run.artifacts_dir,
            status=state.run.status,  # Phase B1: carry the run lifecycle status through
        ),
        model=ActiveModel(
            model_ref=state.model.model_ref,
            version=state.model.version,
            approval=state.model.approval,
            endpoint=state.model.endpoint,
        ),
        problem=ActiveProblem(
            problem_id=state.problem.problem_id,
            kpi=state.problem.kpi,
            problem_card_path=state.problem.problem_card_path,
        ),
        steering=state.steering,
        agents={
            name: AgentActivity(
                agent_name=act.agent_name,
                run_id=act.run_id,
                session_id=act.session_id,
                phase=act.phase,
                current_span=act.current_span,
                last_tool=act.last_tool,
                last_model=act.last_model,
                aws_services=list(act.aws_services),
                status=act.status,
                updated_at=act.updated_at,
                signed_on=act.signed_on,  # Phase 5 §2e — carry the roster flag through
            )
            for name, act in state.agents.items()
        },
        # Phase 5 §2d: deep-copy the local-processes collection (same pattern as agents).
        processes={
            pid: ProcessActivity(
                process_id=proc.process_id,
                kind=proc.kind,
                pid=proc.pid,
                status=proc.status,
                started_by=proc.started_by,
                progress=proc.progress,
                detail=proc.detail,
                updated_at=proc.updated_at,
            )
            for pid, proc in state.processes.items()
        },
        selection=dict(state.selection),
        # Plan §B2: deep-copy the requested view so a subscriber cannot mutate the pending
        # render request (same per-field discipline as every record above — this function
        # enumerates fields explicitly, so a new field MUST be added here or the snapshot
        # silently drops it).
        view=ActiveView(
            view_id=state.view.view_id,
            chart=state.view.chart,
            title=state.view.title,
            series={name: list(vals) for name, vals in state.view.series.items()},
            values=dict(state.view.values),
            matrix=[list(row) for row in state.view.matrix],
            x=list(state.view.x),
            y=list(state.view.y),
            labels=list(state.view.labels),
            source=state.view.source,
            detail=state.view.detail,
            target=state.view.target,
        ),
    )
