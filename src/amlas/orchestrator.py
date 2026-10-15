"""Orchestrator — drives the Research → FE → ML → Arbiter loop with retreat.

R&D-Agent skeleton (arXiv:2505.14738): Research phase produces a signed hypothesis,
Development phase implements + tests it, Arbiter (deterministic Python) grades the
output against the golden set, and on failure the Kanban "abandoned" lane becomes
the input to the next iteration's Research phase.

We intentionally DO NOT use Strands GraphBuilder here — Arbiter is not an agent, and
the iteration loop needs to pull `y_pred` off disk and invoke Arbiter between Graph
runs. A plain Python driver is cleaner and identical in semantics. See PLAN.md §3.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from .arbiter import grade
from .arbiter.schema import ArbiterReport
from .config.aws_region import resolve_aws_region
from .spec.schema import SpecModel
from .telemetry.kanban import KanbanTask, RunKanban
from .telemetry.trace import TraceLogger


class BudgetExhaustedError(RuntimeError):
    """Raised when wall_clock or iteration budget is exhausted without acceptance."""


# FIX-26 (operator directive: NO arbitrary caps on the work): there is NO absolute wall-clock
# ceiling on a run. A wall-clock deadline is honored ONLY when the operator explicitly sets a
# POSITIVE ``wall_clock_minutes`` in the spec (or shrinks it via ``call_timeout_seconds`` /
# AMLAS_LEAD_CALL_TIMEOUT_SECONDS). Absent that, the Orchestrator loop is UNBOUNDED — it runs
# to natural completion (acceptance, real exhaustion of the agent's own work, or a genuine
# error). The authoritative terminal-event guarantee still lives in app.py's entrypoint.

# FIX-20b (Subtask-13): the canonical y_pred paths the ML Lead's trainer_tuner writes and
# FIX-20 mirrors OUT to the runtime PROCESS fs. When a mid-run kernel reprovision resets the
# ML Lead's context so it retreats WITHOUT handing back a ``y_pred_path``, the Orchestrator
# falls back to grading this durable mirror — a real prediction the model produced is never
# lost to a lead-context reset. Kept in-sync with code_interpreter._YPRED_SANDBOX_CANDIDATES.
_CANONICAL_YPRED_CANDIDATES: tuple[str, ...] = (
    "/tmp/amlas/y_pred_test.npy",  # nosec B108 — canonical sandbox→process mirror path
    "/tmp/amlas/y_pred_test.csv",  # nosec B108
)


def _durable_ypred_path() -> str | None:
    """First canonical y_pred path that EXISTS on the process fs (FIX-20 mirror), else None."""
    for candidate in _CANONICAL_YPRED_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


Phase = Literal["R", "D1", "D2", "ARBITER"]


@dataclass
class OrchestratorResult:
    status: Literal["accepted", "budget_exhausted", "failed"]
    arbiter_report: ArbiterReport | None
    iterations: int
    kanban: RunKanban
    final_hypothesis: dict[str, Any] | None = None
    phase_log: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0


class Orchestrator:
    """Drives the full iteration loop.

    Dependency-injected so callers can substitute stubbed Leads + a fake y_pred loader.
    """

    def __init__(
        self,
        *,
        spec: SpecModel,
        research_lead: Any,
        fe_lead: Any,
        ml_lead: Any,
        y_true_loader: Callable[[], Any],
        y_pred_loader: Callable[[str], Any],
        run_id: str | None = None,
        runs_dir: str | Path = "runs",
        trace_logger: TraceLogger | None = None,
        data_paths: dict[str, Any] | None = None,
        grader: Callable[..., Any] | None = None,
        call_timeout_seconds: float | None = None,
        max_no_progress_iterations: int | None = None,
    ):
        self.spec = spec
        self.research_lead = research_lead
        self.fe_lead = fe_lead
        self.ml_lead = ml_lead
        self.y_true_loader = y_true_loader
        self.y_pred_loader = y_pred_loader
        # M1 (doc 05 §5.4; ADR-003): the deterministic Arbiter grade is a
        # NON-DELEGATED decision step. ``grader`` lets the OrchestrationBackend
        # LocalDriver adapter inject the EvaluationBackend's ``grade`` (the
        # DeterministicArbiter wrapping the SAME ``arbiter.grade``) so the loop
        # runs through the port — but it is BEHAVIOR-PRESERVING: when ``grader``
        # is ``None`` (every existing caller) the module-level ``grade`` is used
        # exactly as before, so results are byte-identical.
        self._grade: Callable[..., Any] = grader or grade
        # FIX-15 (Subtask-5 durable server-side termination guard): the Subtask-4
        # live run produced 0 client events because a long/non-converging agent
        # loop never let the ``@app.entrypoint`` generator yield a terminal event.
        # The HARD per-run deadline + the terminal-event guarantee live in
        # ``app.py`` (the only layer that can keep the AgentCore SSE stream alive
        # via heartbeats while ``orch.run()`` executes on a worker thread). Here
        # the Orchestrator keeps CHEAP defense-in-depth: a wall-clock check at each
        # iteration boundary AND before each Lead call (so it never STARTS another
        # multi-minute Lead once the budget is spent), capped by
        # ``_ABSOLUTE_CAP_MINUTES``. ``call_timeout_seconds`` (or
        # ``AMLAS_LEAD_CALL_TIMEOUT_SECONDS``) optionally shrinks the effective
        # wall-clock deadline; ``max_no_progress_iterations`` (or
        # ``AMLAS_MAX_NO_PROGRESS_ITERATIONS``) is a belt-and-braces stall guard for
        # a loop that keeps retreating without ever reaching an Arbiter grade.
        self._call_timeout_seconds: float | None = (
            call_timeout_seconds
            if call_timeout_seconds is not None
            else _env_float("AMLAS_LEAD_CALL_TIMEOUT_SECONDS")
        )
        # FIX-26: the stall guard is FAILURE-DETECTION, not a work cap — it stops ONLY a loop
        # that keeps retreating without EVER reaching an Arbiter grade (zero real progress). With
        # the arbitrary wall-clock/iteration caps removed, this is the safety belt that guarantees
        # an unbounded run still terminates on genuine non-convergence (never an infinite spin).
        # Explicit arg > env > a sane default (25 fruitless iterations) so it never caps real work.
        _explicit_no_progress = (
            max_no_progress_iterations
            if max_no_progress_iterations is not None
            else _env_int("AMLAS_MAX_NO_PROGRESS_ITERATIONS")
        )
        self._max_no_progress: int | None = (
            _explicit_no_progress if _explicit_no_progress is not None else 25
        )
        self._deadline: float = 0.0  # absolute monotonic deadline; set at run() start
        # N1 (Phase 14b): concrete, already-staged train/validation paths + resolved
        # target. When present these are threaded verbatim into the Research
        # prompt (so framing's problem card carries them) AND the FE/ML Lead
        # user messages (so data_integrity(train_path, test_path, target) runs
        # on REAL files instead of guessing data/ai4i/train.csv). When None the
        # message shapes are exactly as before (backward-compatible).
        self.data_paths: dict[str, Any] | None = dict(data_paths) if data_paths else None
        # Layered-config: overlay the agent-SWARM ``.env.swarm`` (per-role model
        # tiers, retreat/budget knobs, memory/KB + steering defaults) on top of
        # the shared-base root ``.env`` at swarm-assembly time, BEFORE the leads/
        # sub-agents read ``load_config()``/env. Idempotent + shell-wins (see
        # ``config/env_loader``). Best-effort: never let env loading block a run.
        try:
            from .config.env_loader import load_layered_env

            load_layered_env("swarm")
        except Exception:  # nosec B110
            pass
        self.run_id = run_id or _mk_run_id(spec.project_id)
        self.run_dir = Path(runs_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace = trace_logger or TraceLogger(self.run_dir / "trace.jsonl")
        self.kanban = RunKanban(run_id=self.run_id, project_id=spec.project_id)
        # Hooks factory closure — produces a `[MemoryHookProvider(...)]` scoped to
        # (memory_id=env.MEMORY_ID, actor_id=spec.project_id, session_id=run_id)
        # when MEMORY_ID is set, else `[]`. Callers (Leads/Subagents) pass this
        # into `build_agent(hooks_factory=...)` so every agent built
        # within this run shares the same MemoryClient instance. See bugfix
        # spec v1-correctness-and-safety-hardening task 2.2 (C1 → P1).
        self._hooks_factory: Callable[[], list[Any]] = self._build_hooks_factory()

    def run(self) -> OrchestratorResult:
        t0 = time.monotonic()
        # FIX-15: the wall-clock ceiling used for the Orchestrator's CHEAP
        # defense-in-depth (checked at each iteration boundary + before each Lead
        # call, so it never STARTS another multi-minute Lead once the budget is
        # spent). The spec's wall_clock_minutes is capped at _ABSOLUTE_CAP_MINUTES;
        # ``call_timeout_seconds`` (or AMLAS_LEAD_CALL_TIMEOUT_SECONDS), when set,
        # shrinks it further. The AUTHORITATIVE hard deadline + terminal-event
        # guarantee live in app.py's entrypoint generator (which runs run() on a
        # daemon worker and stops waiting at the deadline).
        # FIX-26: no absolute ceiling. A deadline exists ONLY if the operator explicitly set a
        # positive wall_clock_minutes (optionally shrunk by call_timeout_seconds). Otherwise the
        # loop is UNBOUNDED (_deadline = inf) and runs to natural completion.
        spec_minutes = float(getattr(self.spec.compute_budget, "wall_clock_minutes", 0) or 0)
        if spec_minutes > 0:
            wall_seconds = spec_minutes * 60.0
            if self._call_timeout_seconds is not None:
                wall_seconds = min(wall_seconds, self._call_timeout_seconds)
            self._deadline = t0 + wall_seconds
        elif self._call_timeout_seconds is not None:
            self._deadline = t0 + self._call_timeout_seconds
        else:
            self._deadline = float("inf")  # unbounded — no arbitrary wall-clock cap
        # FIX-26: max_retreats is the operator's retreat budget when they set one; 0/unset means
        # DO NOT cap iterations (the agent iterates until it converges or genuinely gives up).
        _retreats = int(getattr(self.spec.compute_budget, "max_retreats", 0) or 0)
        max_iter = (_retreats + 1) if _retreats > 0 else None
        phase_log: list[dict[str, Any]] = []
        last_report: ArbiterReport | None = None
        last_hypothesis: dict[str, Any] | None = None
        no_progress = 0  # FIX-15 belt-and-braces: consecutive iterations without a grade
        iteration = 0
        deadline_hit = False

        with self.trace.span(f"orchestrator_run:{self.run_id}", "agent", project=self.spec.project_id):
            iteration = 0
            while True:
                # FIX-26: honor an explicit retreat budget when set; otherwise iterate
                # unbounded (the stall guard + acceptance + genuine retreat-exhaustion below
                # are the natural stopping conditions, not an arbitrary iteration cap). Check
                # BEFORE incrementing so ``iteration`` reflects the last EXECUTED iteration.
                if max_iter is not None and iteration >= max_iter:
                    break
                iteration += 1
                if time.monotonic() >= self._deadline:
                    deadline_hit = True
                    break
                # FIX-15 belt-and-braces stall guard: if the loop keeps retreating
                # without ever reaching an Arbiter grade, stop early instead of
                # burning the entire (possibly large) retreat budget on a
                # non-converging agent — always yields a terminal result.
                if self._max_no_progress is not None and no_progress >= self._max_no_progress:
                    print(
                        f"STALL_GUARD: no progress for {no_progress} iterations — terminating",
                        flush=True,
                    )
                    break
                try:
                    hypothesis, _fe_output, ml_output = self._run_iteration(iteration)
                except BudgetExhaustedError as e:
                    # FIX-15: a Lead call blew the hard wall-clock deadline. Stop
                    # cleanly with a terminal result carrying partial progress.
                    print(f"WALL_CLOCK_GUARD: {e}", flush=True)
                    phase_log.append(
                        {"iteration": iteration, "retreat_reason": str(e), "phase": "wall_clock"}
                    )
                    deadline_hit = True
                    break
                except _IterationRetreat as e:
                    phase_log.append({"iteration": iteration, "retreat_reason": e.reason, "phase": e.phase})
                    # FIX-20b: the ML Lead retreated (e.g. its context was reset by a mid-run kernel
                    # reprovision), but a REAL y_pred may already be durable on the process fs (the
                    # FIX-20 mirror). Grade it best-effort so a produced prediction is never lost to a
                    # lead-context reset — this sets ``last_report`` so even a later budget_exhausted
                    # terminal carries a NON-null arbiter_report (G-METRIC/G-A3), and an accepting
                    # score can still finalize the run.
                    accepted = self._maybe_grade_durable_ypred(
                        iteration, last_hypothesis, phase_log, t0
                    )
                    if accepted is not None:
                        if isinstance(accepted, OrchestratorResult):
                            return accepted
                        last_report = accepted
                    # _run_iteration already abandoned the failing task in kanban; just advance.
                    self.kanban.retreat(iteration + 1)
                    self._save_kanban()
                    no_progress += 1
                    continue

                last_hypothesis = hypothesis
                # FIX-20b: prefer the Lead's reported y_pred_path; else fall back to the DURABLE
                # y_pred the model already wrote (FIX-20 mirror on the process fs). A mid-run
                # kernel reprovision can reset the ML Lead's context so it omits y_pred_path even
                # though a real prediction was produced — grade that mirror rather than lose it.
                y_pred_path = ml_output.get("y_pred_path") or _durable_ypred_path()
                if not y_pred_path:
                    self._record_phase_retreat(iteration, "ml_lead returned no y_pred_path", "D2")
                    phase_log.append(
                        {"iteration": iteration, "retreat_reason": "missing y_pred_path", "phase": "D2"}
                    )
                    continue
                if not ml_output.get("y_pred_path"):
                    print(
                        f"FIX20B_DURABLE_YPRED: iter={iteration} ml_lead omitted y_pred_path — "
                        f"grading the durable mirror at {y_pred_path!r}",
                        flush=True,
                    )

                with self.trace.span(f"arbiter_iter{iteration}", "phase"):
                    # FIX-38b (ADR-0002 Phase 6): loading THIS iteration's y_pred must never crash the
                    # whole run. A bad/absent y_pred_path (e.g. a lead that mis-reported a /tmp path
                    # instead of the dispatched job's s3:// URI, or a transient S3 miss) is an
                    # ITERATION-level retreat — record it and let the next iteration try — NOT a
                    # run-level failure that discards all prior graded iterations. Previously an
                    # uncaught FileNotFoundError here propagated out of run() → terminal status=failed
                    # with arbiter_report=null even when EARLIER iterations graded fine.
                    try:
                        y_true = self.y_true_loader()
                        y_pred = self.y_pred_loader(y_pred_path)
                    except Exception as load_err:
                        print(
                            f"YPRED_LOAD_RETREAT: iter={iteration} path={y_pred_path!r} "
                            f"{type(load_err).__name__}: {load_err} — retreating this iteration",
                            flush=True,
                        )
                        self._record_phase_retreat(
                            iteration, f"y_pred not loadable: {type(load_err).__name__}", "D2"
                        )
                        phase_log.append(
                            {"iteration": iteration, "retreat_reason": "y_pred_unloadable", "phase": "D2"}
                        )
                        continue
                    # NON-DELEGATED deterministic grade (ADR-003). ``self._grade``
                    # is the module-level ``grade`` by default (behavior-preserving)
                    # or the injected EvaluationBackend ``grade`` (M1 LocalDriver),
                    # which wraps the SAME ``arbiter.grade`` — identical decision.
                    report = self._grade(
                        spec=self.spec,
                        y_true=y_true,
                        y_pred=y_pred,
                        phase_log=phase_log,
                    )
                # N5 instrumentation: surface the Arbiter grade to CloudWatch so
                # "the run reached the Arbiter and graded" is observable server-side
                # (the streamed run_completed event is buffered until run end).
                # Metadata-only log line (score/direction/acceptance) — never add data
                # samples or predictions to this print (Sensitive-data note).
                print(
                    f"ARBITER_GRADED: iter={iteration} "
                    f"overall_score={report.overall_score} "
                    f"direction={report.direction} "
                    f"passes_acceptance={report.passes_acceptance}",
                    flush=True,
                )
                last_report = report
                phase_log.append(
                    {
                        "iteration": iteration,
                        "score": report.overall_score,
                        "direction": report.direction,
                        "passes": report.passes_acceptance,
                    }
                )

                if report.passes_acceptance:
                    self._finalize(iteration, status="accepted", report=report, hypothesis=hypothesis)
                    return OrchestratorResult(
                        status="accepted",
                        arbiter_report=report,
                        iterations=iteration,
                        kanban=self.kanban,
                        final_hypothesis=hypothesis,
                        phase_log=phase_log,
                        elapsed_seconds=time.monotonic() - t0,
                    )
                # Arbiter rejected: record an ARBITER retreat with a fresh abandoned entry
                self._record_arbiter_retreat(iteration, report.acceptance_reason)

        # FIX-15: report the iterations actually REACHED (never a fixed max_iter),
        # so a wall-clock/stall termination surfaces the true progress. On the
        # deadline path the last iteration was started but did not complete.
        status: Literal["budget_exhausted", "failed"] = "budget_exhausted"
        iterations_reached = iteration if iteration else (max_iter or 0)
        if deadline_hit:
            print(
                f"RUN_TERMINATED: status={status} iterations={iterations_reached} "
                f"deadline_hit=True elapsed={time.monotonic() - t0:.1f}s",
                flush=True,
            )
        return OrchestratorResult(
            status=status,
            arbiter_report=last_report,
            iterations=iterations_reached,
            kanban=self.kanban,
            final_hypothesis=last_hypothesis,
            phase_log=phase_log,
            elapsed_seconds=time.monotonic() - t0,
        )

    # --- internals ---

    def _build_hooks_factory(self) -> Callable[[], list[Any]]:
        """Build the hooks-factory closure for this run.

        Returns a zero-arg callable that produces the per-agent ``hooks`` list to
        pass into ``build_agent(..., hooks_factory=...)``. Two cases:

        1. ``MEMORY_ID`` is unset / empty → returns ``lambda: []``. Preserves the
           dev-mode path (Requirement 3.1): when no AgentCore Memory resource is
           provisioned, agents build with zero ``MemoryHookProvider`` attached.
        2. ``MEMORY_ID`` is set → returns a closure that lazily imports
           ``bedrock_agentcore.memory.MemoryClient`` on first invocation,
           constructs it once with the selected AWS region, and on every call returns a single-element list
           ``[MemoryHookProvider(client=<shared>, memory_id=MEMORY_ID,
           actor_id=self.spec.project_id, session_id=self.run_id)]`` so every
           Agent constructed within this run shares the same MemoryClient.

        The import of ``MemoryClient`` is deferred so offline validation that build the
        Orchestrator without ``MEMORY_ID`` never pull in ``bedrock_agentcore``
        at module-load time (Requirement 3.2). The closure itself is only
        invoked from inside ``build_agent``; tests that pass a pre-built
        ``model=`` (a real BedrockModel) without ``MEMORY_ID`` exercise the
        ``return lambda: []`` branch below.
        """
        memory_id = os.environ.get("MEMORY_ID", "")
        if not memory_id:
            return lambda: []

        spec_project_id = self.spec.project_id
        run_id = self.run_id
        region_name = resolve_aws_region(required=False)
        client_cell: dict[str, Any] = {}

        def _factory() -> list[Any]:
            client: Any = client_cell.get("client")
            if client is None:
                # Lazy import — keeps `bedrock_agentcore` off the critical
                # import path for offline validation and the MEMORY_ID-unset case.
                from bedrock_agentcore.memory import MemoryClient

                client = MemoryClient(region_name=resolve_aws_region(region_name))
                client_cell["client"] = client
            # Lazy import too — `MemoryHookProvider` pulls in `strands.hooks`,
            # which is only present under the `agents` extra (the deployed
            # image). Deferring it here (only when MEMORY_ID is set, exactly
            # where MemoryClient is already imported) keeps the Orchestrator
            # importable for offline validation / the cloud-dispatch client without the
            # full agents stack. Behavior is unchanged in production.
            from .memory.hooks import MemoryHookProvider

            return [
                MemoryHookProvider(
                    client=client,
                    memory_id=memory_id,
                    actor_id=spec_project_id,
                    session_id=run_id,
                )
            ]

        return _factory

    def _run_iteration(self, iteration: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        # Research phase
        self._start_task(f"R{iteration}", "R1_FRAMING", iteration, "research")
        with self.trace.span(f"research_iter{iteration}", "agent"):
            r_raw = self._call_lead(self.research_lead, self._research_prompt(iteration), "R", iteration)
        hypothesis = _parse_json_or_raise(str(r_raw), phase="R", iteration=iteration)
        self.kanban.complete(f"R{iteration}")

        # FE phase
        self._start_task(f"FE{iteration}", "D1_FE", iteration, "feature engineering")
        # N1: surface the concrete train/test/target paths the FE Lead prompt
        # expects (prompts/leads/fe_lead.md calls
        # data_integrity(train_path, test_path, target, ...)). Without these the
        # D1 agent guesses data/ai4i/train.csv and retreats (phase14a CW 12:56:59).
        fe_msg: dict[str, Any] = {"hypothesis": hypothesis, "iteration": iteration}
        if self.data_paths:
            fe_msg["train_path"] = self.data_paths.get("train_path")
            fe_msg["test_path"] = self.data_paths.get("test_path")
            fe_msg["target"] = self.data_paths.get("target", self.spec.target)
            fe_msg["data_paths"] = dict(self.data_paths)
        with self.trace.span(f"fe_iter{iteration}", "agent"):
            fe_raw = self._call_lead(self.fe_lead, json.dumps(fe_msg), "D1", iteration)
        fe_output = _parse_json_or_raise(str(fe_raw), phase="D1", iteration=iteration)
        if fe_output.get("status") == "blocked":
            reason = _extract_reason(fe_output, "data integrity block")
            self.kanban.abandon(f"FE{iteration}", reason=reason)
            raise _IterationRetreat(reason, "D1")
        self.kanban.complete(f"FE{iteration}")

        # ML Execution phase
        self._start_task(f"ML{iteration}", "D2_ML_EXEC", iteration, "ml execution")
        ml_msg: dict[str, Any] = {
            "hypothesis": hypothesis,
            "iteration": iteration,
            "feature_manifest_path": fe_output.get("feature_manifest_path"),
            "augmentation_manifest": fe_output.get("augmentation_manifest"),
        }
        if self.data_paths:
            # N1: the ML Lead trains on the staged train split + predicts on the
            # staged test split; pass the concrete paths so it never guesses.
            ml_msg["data_paths"] = dict(self.data_paths)
        with self.trace.span(f"ml_iter{iteration}", "agent"):
            ml_raw = self._call_lead(self.ml_lead, json.dumps(ml_msg), "D2", iteration)
        ml_output = _parse_json_or_raise(str(ml_raw), phase="D2", iteration=iteration)
        if ml_output.get("status") == "failed":
            reason = _extract_reason(ml_output, "ml execution failed")
            self.kanban.abandon(f"ML{iteration}", reason=reason)
            raise _IterationRetreat(reason, "D2")
        self.kanban.complete(f"ML{iteration}")

        return hypothesis, fe_output, ml_output

    def _call_lead(self, lead: Callable[[str], Any], arg: str, phase: str, iteration: int) -> Any:
        """FIX-15: invoke a Lead, honoring the run's wall-clock deadline.

        The blocking Lead call runs DIRECTLY on the caller thread (the whole
        ``run()`` executes on one consistent worker thread — see ``app.py``, which
        drives ``orch.run()`` off the asyncio event loop so it can emit heartbeat
        events and enforce the hard per-run deadline while the stream stays alive).

        We deliberately do NOT run each Lead on its own background thread: the
        deployed Leads nest Strands + Code-Interpreter sandbox execution, and the
        HARD per-run deadline + the terminal-event guarantee are owned by the
        ``app.py`` entrypoint generator (the only place that can keep the AgentCore
        SSE stream alive). Here we only enforce a CHEAP pre-call check: if the
        wall-clock budget is already spent, raise :class:`BudgetExhaustedError`
        BEFORE starting another multi-minute Lead call, so ``run()`` returns a
        terminal result promptly. ``_ABSOLUTE_CAP_MINUTES`` still belts a runaway.
        """
        if self._deadline and time.monotonic() >= self._deadline:
            raise BudgetExhaustedError(
                f"wall_clock exceeded before {phase} lead call (iteration {iteration})"
            )
        return lead(arg)

    def _research_prompt(self, iteration: int) -> str:
        abandoned_reasons = [a.abandoned_reason for a in self.kanban.abandoned]
        payload: dict[str, Any] = {
            "task_spec": self.spec.model_dump(mode="json"),
            "iteration": iteration,
            "abandoned_reasons_so_far": abandoned_reasons,
        }
        if self.data_paths:
            # N1: the framing sub-agent distils the problem card from this payload;
            # carrying the concrete data_paths means the card downstream agents
            # consume already knows where the real train/test files live.
            payload["data_paths"] = dict(self.data_paths)
        return json.dumps(payload)

    def _start_task(self, task_id: str, phase: str, iteration: int, title: str) -> None:
        self.kanban.start(KanbanTask(id=task_id, phase=phase, iteration=iteration, title=title))  # type: ignore[arg-type]

    def _maybe_grade_durable_ypred(
        self,
        iteration: int,
        hypothesis: dict[str, Any] | None,
        phase_log: list[dict[str, Any]],
        t0: float,
    ) -> Any:
        """FIX-20b: grade the DURABLE y_pred mirror after an ML-Lead retreat, best-effort.

        A mid-run kernel reprovision can reset the ML Lead so it retreats WITHOUT handing back
        a ``y_pred_path`` — yet the model may have ALREADY produced a real prediction that
        FIX-20 mirrored to the process fs. Grade it so the produced prediction is never lost:
          * returns an ``OrchestratorResult(status="accepted", ...)`` if the grade passes
            (the caller returns it → the run finalizes on the recovered prediction);
          * else returns the ``ArbiterReport`` (the caller sets ``last_report`` so even a later
            ``budget_exhausted`` terminal carries a NON-null ``arbiter_report`` — G-METRIC/G-A3);
          * returns ``None`` when there is no durable y_pred or grading is not possible.
        NEVER raises (a failed recovery must not break the retreat path).
        """
        path = _durable_ypred_path()
        if not path:
            return None
        try:
            with self.trace.span(f"arbiter_durable_iter{iteration}", "phase"):
                y_true = self.y_true_loader()
                y_pred = self.y_pred_loader(path)
                report = self._grade(
                    spec=self.spec, y_true=y_true, y_pred=y_pred, phase_log=phase_log
                )
        except Exception as e:  # best-effort — a failed recovery must not break the retreat
            print(f"FIX20B_DURABLE_GRADE_SKIPPED: iter={iteration} {type(e).__name__}: {e}", flush=True)
            return None
        print(
            f"ARBITER_GRADED: iter={iteration} overall_score={report.overall_score} "
            f"direction={report.direction} passes_acceptance={report.passes_acceptance} "
            f"(FIX20B durable-mirror recovery after ml_lead retreat)",
            flush=True,
        )
        phase_log.append(
            {
                "iteration": iteration,
                "score": report.overall_score,
                "direction": report.direction,
                "passes": report.passes_acceptance,
                "durable_recovery": True,
            }
        )
        if report.passes_acceptance:
            self._finalize(iteration, status="accepted", report=report, hypothesis=hypothesis)
            return OrchestratorResult(
                status="accepted",
                arbiter_report=report,
                iterations=iteration,
                kanban=self.kanban,
                final_hypothesis=hypothesis,
                phase_log=phase_log,
                elapsed_seconds=time.monotonic() - t0,
            )
        return report

    def _record_arbiter_retreat(self, iteration: int, reason: str) -> None:
        """Arbiter rejected this iteration — log an 'arbiter' entry in abandoned lane."""
        tid = f"arbiter_{iteration}"
        self.kanban.start(KanbanTask(id=tid, phase="D3_ARBITER", iteration=iteration, title="arbiter check"))
        self.kanban.abandon(tid, reason=reason)
        self.kanban.retreat(iteration + 1)
        self._save_kanban()

    def _record_phase_retreat(self, iteration: int, reason: str, phase: Phase) -> None:
        """Phase failure outside _run_iteration (e.g., missing y_pred_path) — synthesise abandon."""
        phase_map = {"R": "R1_FRAMING", "D1": "D1_FE", "D2": "D2_ML_EXEC", "ARBITER": "D3_ARBITER"}
        kphase = phase_map.get(phase, "R1_FRAMING")
        tid = f"phase_retreat_{iteration}_{phase}"
        self.kanban.start(KanbanTask(id=tid, phase=kphase, iteration=iteration, title=f"retreat at {phase}"))  # type: ignore[arg-type]
        self.kanban.abandon(tid, reason=reason)
        self.kanban.retreat(iteration + 1)
        self._save_kanban()

    def _finalize(
        self,
        iteration: int,
        *,
        status: str,
        report: ArbiterReport,
        hypothesis: dict[str, Any] | None,
    ) -> None:
        (self.run_dir / "arbiter_report.json").write_text(report.model_dump_json(indent=2))
        if hypothesis is not None:
            (self.run_dir / "hypothesis.json").write_text(json.dumps(hypothesis, indent=2))
        self._save_kanban()

    def _save_kanban(self) -> None:
        self.kanban.save(self.run_dir / "kanban.json")


@dataclass
class _IterationRetreat(Exception):
    reason: str
    phase: Phase


def _parse_json_or_raise(text: str, *, phase: str, iteration: int) -> dict[str, Any]:
    """Parse an agent final message as JSON. Accept optional prose with a fenced block."""
    text = text.strip()
    # Try direct parse
    try:
        direct: dict[str, Any] = json.loads(text)
        return direct
    except json.JSONDecodeError:
        pass
    # Try fenced ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        fenced: dict[str, Any] = json.loads(m.group(1))
        return fenced
    # Try naïvely finding the outermost {...}
    first, last = text.find("{"), text.rfind("}")
    if first != -1 and last > first:
        try:
            naive: dict[str, Any] = json.loads(text[first : last + 1])
            return naive
        except json.JSONDecodeError:
            pass
    raise _IterationRetreat(
        reason=f"{phase} lead returned non-JSON final message (iteration {iteration}): {text[:200]!r}",
        phase=phase,  # type: ignore[arg-type]
    )


def _extract_reason(d: dict[str, Any], default: str) -> str:
    # Check top-level reason keys first
    for key in ("reason", "acceptance_reason", "iteration_notes"):
        v = d.get(key)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, list) and v:
            return str(v[0])
    # Recurse into known nested blocks (FE block puts reason under data_integrity)
    for nested_key in ("data_integrity", "arbiter_report"):
        nested = d.get(nested_key)
        if isinstance(nested, dict):
            for key in ("reason", "acceptance_reason"):
                v = nested.get(key)
                if isinstance(v, str) and v:
                    return v
    return default


def _mk_run_id(project_id: str) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    short = uuid.uuid4().hex[:8]
    # AgentCore naming constraint ≥33 chars for A2A session IDs — pad with project_id
    return f"run_{ts}_{short}_{project_id}"


# Convenience model for surfacing the orchestrator-level state to hooks / tests


class OrchestratorState(BaseModel):
    """Snapshot of orchestrator state — for inspection and external observers."""

    run_id: str
    project_id: str
    iteration: int
    phase: Phase | None = None
    passes_acceptance: bool = False
    retreat_count: int = 0
