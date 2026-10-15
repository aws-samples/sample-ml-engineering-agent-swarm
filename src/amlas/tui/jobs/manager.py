"""DP-2 — ``JobManager`` + ``Job``: in-process background execution for the TUI.

Design (``docs/audit/phase7-road-to-100-design.md`` §3): long, blocking ops
(``MLPipeline.run``, ``train_model``, ``tune_model``, ``ExperimentRunner``,
``ProjectScheduler.run_all``) must NOT freeze the Textual event loop. They run on
a background OS thread; UI mutation marshals back via ``App.call_from_thread`` —
the proven ``status_bar.py:31`` discipline.

Testability seam (§3.3, §6 — worker-flakiness mitigation): the manager takes a
pluggable ``executor``. Three executors ship:

* :class:`InlineExecutor` — **the default**. Runs ``fn`` synchronously inside
  ``submit()`` so the job is already terminal (``complete``/``failed``) when
  ``submit()`` returns. This makes the manager fully deterministic with NO threads
  and NO running event loop, so a headless ``ChatEngine()`` (every existing caller)
  observes immediate completion — the long-op handlers can return their real
  ``*_complete`` ``ChatResponse`` exactly as before Phase 11.
* :class:`ThreadExecutor` — the real app path: ``fn`` runs on a daemon thread;
  ``submit()`` returns immediately with the job still ``running`` so the UI never
  blocks. Completion events optionally marshal through an app's ``call_from_thread``.
* :class:`DeferredExecutor` — validation-only: ``submit()`` returns with the job
  ``running`` (non-blocking proof) and the test drives it to completion on demand
  via :meth:`DeferredExecutor.run_pending` — deterministic, no ``sleep``.

Cancellation (§3.1): cooperative. ``submit`` passes the worker ``fn`` a
``cancel_token`` (a ``threading.Event``); long loops check it between units. A job
cancelled before it starts (or that observes the token / raises
:class:`JobCancelled`) ends in ``cancelled``.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# ─── Job status constants (the §3.1 state machine) ──────────────────────────────
JOB_SUBMITTED = "submitted"
JOB_RUNNING = "running"
JOB_COMPLETE = "complete"
JOB_FAILED = "failed"
JOB_CANCELLED = "cancelled"

TERMINAL_STATUSES = frozenset({JOB_COMPLETE, JOB_FAILED, JOB_CANCELLED})


class JobCancelled(Exception):
    """Raised by a worker ``fn`` (or detected via the cancel token) to end a job
    in the ``cancelled`` state rather than ``failed``."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class Job:
    """One unit of background work and its full lifecycle state (§3.1).

    ``fn`` is called as ``fn(progress_cb, cancel_token)`` where ``progress_cb`` is
    ``(pct: float, stage: str) -> None`` and ``cancel_token`` is a
    ``threading.Event`` the worker may poll to support cooperative cancellation.
    """

    id: str
    kind: str
    fn: Callable[..., Any]
    args: dict[str, Any] = field(default_factory=dict)
    status: str = JOB_SUBMITTED
    progress: float = 0.0
    stage: str = ""
    result: Any = None
    error: str | None = None
    created_at: str = field(default_factory=_now)
    finished_at: str | None = None
    cancel_token: threading.Event = field(default_factory=threading.Event)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def cancel_requested(self) -> bool:
        return self.cancel_token.is_set()


# ─── Executors (the injectable seam) ────────────────────────────────────────────


class InlineExecutor:
    """Runs the worker synchronously inside ``submit`` (default; deterministic).

    No threads, no event loop required. After :meth:`run`, the job is terminal.
    This is what every existing headless test exercises, so the long-op handlers
    can synchronously read the finished job and return their real completion
    response — preserving the pre-Phase-11 behaviour exactly.
    """

    def run(self, runner: Callable[[], None]) -> None:
        runner()


class ThreadExecutor:
    """Runs the worker on a daemon thread; ``submit`` returns immediately (app path).

    Each job runs on its own thread so the Textual event loop stays responsive
    (the §3 decision: threads, not asyncio tasks, since the ops are blocking
    Python). Threads are tracked so validation coverage can ``join`` them deterministically.
    """

    def __init__(self) -> None:
        self._threads: list[threading.Thread] = []

    def run(self, runner: Callable[[], None]) -> None:
        thread = threading.Thread(target=runner, daemon=True)
        self._threads.append(thread)
        thread.start()

    def join(self, timeout: float | None = None) -> None:
        """Join all spawned threads (bounded — used by the real-worker test)."""
        for thread in list(self._threads):
            thread.join(timeout)


class DeferredExecutor:
    """Validation-only: queues the worker; ``submit`` returns with the job ``running``.

    Proves the non-blocking contract deterministically: after ``submit`` the job
    is NOT terminal; the calls :meth:`run_pending` to drive each queued worker
    to completion — no threads, no ``sleep``, fully ordered.
    """

    def __init__(self) -> None:
        self._pending: list[Callable[[], None]] = []

    def run(self, runner: Callable[[], None]) -> None:
        self._pending.append(runner)

    def run_pending(self) -> int:
        """Execute (and clear) all queued workers. Returns how many ran."""
        pending, self._pending = self._pending, []
        for runner in pending:
            runner()
        return len(pending)


# ─── JobManager ─────────────────────────────────────────────────────────────────


class JobManager:
    """Registers jobs, runs them via the injected executor, tracks lifecycle.

    Thread-safe registry (a lock guards mutations). An optional ``on_event``
    callback is fired on every state transition with the ``Job`` — the engine uses
    it to post a chat completion line; the app routes it through
    ``call_from_thread`` so widget updates land on the UI thread (§3.2).
    """

    def __init__(
        self,
        executor: Any | None = None,
        on_event: Callable[[Job], None] | None = None,
    ) -> None:
        self.executor = executor or InlineExecutor()
        self.on_event = on_event
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()

    # ─── Registry queries ──────────────────────────────────────────────────────

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def all_jobs(self) -> list[Job]:
        with self._lock:
            return [self._jobs[j] for j in self._order]

    def jobs_by_status(self, status: str) -> list[Job]:
        with self._lock:
            return [self._jobs[j] for j in self._order if self._jobs[j].status == status]

    def counts(self) -> dict[str, int]:
        """Status → count summary used by the jobs indicator (§3.2)."""
        out = {
            JOB_SUBMITTED: 0,
            JOB_RUNNING: 0,
            JOB_COMPLETE: 0,
            JOB_FAILED: 0,
            JOB_CANCELLED: 0,
        }
        with self._lock:
            for job in self._jobs.values():
                out[job.status] = out.get(job.status, 0) + 1
        return out

    @property
    def active_count(self) -> int:
        """Number of jobs not yet terminal (submitted or running)."""
        with self._lock:
            return sum(1 for j in self._jobs.values() if not j.is_terminal)  # nosemgrep

    # ─── Submission ──────────────────────────────────────────────────────────────

    def submit(
        self,
        kind: str,
        fn: Callable[..., Any],
        *,
        args: dict[str, Any] | None = None,
        job_id: str | None = None,
    ) -> Job:
        """Register a job and hand its runner to the executor.

        With the default :class:`InlineExecutor` the runner executes here and the
        returned ``Job`` is already terminal. With Thread/Deferred executors the
        returned ``Job`` is ``running`` (or still ``submitted`` then ``running``)
        and completes later — ``submit`` does not block.
        """
        jid = job_id or f"{kind}-{uuid.uuid4().hex[:8]}"
        job = Job(id=jid, kind=kind, fn=fn, args=dict(args or {}))
        with self._lock:
            self._jobs[jid] = job
            self._order.append(jid)
        self._emit(job)  # submitted

        self.executor.run(lambda: self._run_job(jid))
        return job

    # ─── Worker body (runs in-executor) ──────────────────────────────────────────

    def _run_job(self, job_id: str) -> None:
        job = self.get(job_id)
        if job is None:  # pragma: no cover — registry always has it
            return

        # Honour a cancel requested before the worker actually started.
        if job.cancel_requested:
            self._finish(job, JOB_CANCELLED, error="cancelled before start")
            return

        self._transition(job, JOB_RUNNING)

        def progress_cb(pct: float, stage: str = "") -> None:
            # Cooperative cancellation checkpoint: a worker that calls progress_cb
            # between units gets a clean exit signal when cancellation is requested.
            if job.cancel_requested:
                raise JobCancelled(stage or "cancelled")
            job.progress = float(pct)
            if stage:
                job.stage = stage
            self._emit(job)

        try:
            result = job.fn(progress_cb, job.cancel_token)
        except JobCancelled as exc:
            self._finish(job, JOB_CANCELLED, error=str(exc) or "cancelled")
            return
        except Exception as exc:
            logger.warning("Job %s (%s) failed: %s", job.id, job.kind, exc)
            self._finish(job, JOB_FAILED, error=str(exc))
            return

        # A worker may also signal cancellation by returning after observing the
        # token (without raising) — respect that too.
        if job.cancel_requested:
            self._finish(job, JOB_CANCELLED, error="cancelled", result=result)
        else:
            self._finish(job, JOB_COMPLETE, result=result)

    # ─── Cancellation ─────────────────────────────────────────────────────────────

    def cancel(self, job_id: str) -> bool:
        """Request cooperative cancellation of a job.

        Returns ``True`` if the job exists and is not already terminal. A job that
        has not yet started (or whose worker polls the token / ``progress_cb``)
        transitions to ``cancelled``. A submitted-but-unstarted job is moved to
        ``cancelled`` immediately so the indicator reflects the user's intent.
        """
        job = self.get(job_id)
        if job is None or job.is_terminal:  # nosemgrep
            return False
        job.cancel_token.set()
        if job.status == JOB_SUBMITTED:
            # Never started executing → finalize now.
            self._finish(job, JOB_CANCELLED, error="cancelled before start")
        return True

    # ─── Internal transition helpers ──────────────────────────────────────────────

    def _transition(self, job: Job, status: str) -> None:
        with self._lock:
            job.status = status
        self._emit(job)

    def _finish(
        self,
        job: Job,
        status: str,
        *,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            job.status = status
            job.result = result
            job.error = error
            job.finished_at = _now()
            if status == JOB_COMPLETE:
                job.progress = 100.0
        self._emit(job)

    def _emit(self, job: Job) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(job)
        except Exception as exc:
            logger.warning("JobManager on_event sink raised: %s", exc)
