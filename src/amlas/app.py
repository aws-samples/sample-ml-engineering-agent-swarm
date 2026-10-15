"""AgentCore Runtime entry point.

Hosts the Orchestrator behind `@app.entrypoint`. Payload contract:

    { "spec_path":  "<path to task_spec.md inside the container>",     # one-of
      "spec_text":  "<inline task_spec.md content>",                   # one-of
      "stage_data": true,                                              # Phase 14b N1/N2
      "max_rows":   <int, optional cap for a cheap verification slice>,
      "y_true_path":"<local path OR s3:// uri to a .npy of ground truth>",
      "prompt":     "<optional free-form override — rarely used>"
    }

Phase 14b (N1/N2/B3/B5):
  * N1 — when `stage_data` is truthy, the runtime resolves the spec's
    `golden_dataset` (S3 or local) to concrete train/test files on the container
    filesystem via `cloud.staging.stage_dataset(...)` and threads those paths
    into the Orchestrator's problem card (so D1 `data_integrity` runs on REAL
    files instead of guessing `data/ai4i/train.csv`).
  * N2 — the SAME stage produces an aligned `y_true.npy`; if the operator passes
    an explicit `y_true_path` (incl. an `s3://` uri) it is resolved via
    `cloud.staging.load_y_true_npy(...)` instead.
  * B3 — `cloud.secrets.resolve_secret_into_env()` injects `TAVILY_API_KEY` from
    Secrets Manager (graceful no-op until Phase 14c populates the secret).
  * B5 — this `app.py` path builds the real Bedrock Lead hierarchy + agentic
    Orchestrator (research_search/Tavily + leads/subagents), NOT the offline
    deterministic `MLPipeline` KB.

The Runtime session exits when the entrypoint returns. For the async long-training
pathway (Phase K), the FineTune sub-agent submits a SageMaker job and the Runtime
exits; EventBridge → Lambda re-invokes with runtime_session_id preserved.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from .config.aws_region import resolve_aws_region

app = BedrockAgentCoreApp()

# FIX-26 (operator directive: NO arbitrary caps on the work the system does): there is NO
# absolute wall-clock ceiling. A run executes to its NATURAL completion. A hard deadline is
# applied ONLY when the operator explicitly asks for one — either a positive
# ``wall_clock_minutes`` in the spec, or the ``AMLAS_MAX_RUN_SECONDS`` ops escape hatch (also
# used by offline callers to drive a sub-minute deadline deterministically). Absent both, the run
# is UNBOUNDED. The heartbeat + terminal-event guarantee below are INFRA scaffolding (keep-alive
# so the AgentCore SSE stream never idle-closes without a terminal) — NOT a cap on the work.


def _env_float(name: str, default: float) -> float:
    """Parse a float env var, falling back to ``default`` on unset/invalid."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@app.entrypoint
async def agent_invocation(payload: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
    """Entry point invoked by AgentCore Runtime.

    Streams orchestrator status events. On completion yields the final ArbiterReport
    (or budget_exhausted summary).
    """
    # Deferred imports — Runtime cold-start latency matters; imports only fire on first invoke.

    # Layered-config: overlay the AgentCore/Strands runtime ``.env.agentcore``
    # (runtime ARN, OTEL/STRANDS telemetry toggles, per-role model tiers, boto
    # retry/timeout) on top of the shared-base root ``.env`` BEFORE the Lead
    # hierarchy / Bedrock models are built below. Idempotent + shell-wins (see
    # ``config/env_loader``); best-effort so it never blocks a Runtime invoke.
    try:
        from .config.env_loader import load_layered_env

        load_layered_env("agentcore")
    except Exception:  # nosec B110
        pass

    # : point the process-wide error surface at a real sink at entrypoint start so
    # the 7-agent backend's errors are FUNNELED (local JSONL always + CloudWatch mirror in
    # this managed mode) instead of only printed. Best-effort — never blocks a Runtime invoke.
    try:
        from .telemetry.error_log import init_error_log

        init_error_log()
    except Exception:  # nosec B110
        pass

    from .cloud.secrets import resolve_secret_into_env

    # The three ``build_*_lead`` imports are gone: ``harness.leads.lead_for`` resolves the
    # local builder through the lead registry, so importing them here would be a second path
    # to the same factory.
    from .orchestrator import Orchestrator
    from .spec import load_spec_from_path, load_spec_from_text

    region = resolve_aws_region()

    # B3 mechanism: pull TAVILY_API_KEY from Secrets Manager into the env BEFORE
    # the research agents are built. No-op (and no failure) until Phase 14c
    # populates the secret → research degrades to arxiv + Semantic Scholar.
    try:
        applied = resolve_secret_into_env(region=region)
        # Phase 17 instrumentation: the streamed `tavily_configured` event is
        # buffered by AgentCore until the (multi-minute) generator finishes, so
        # it is NOT visible in CloudWatch mid-run. Emit a flush-print marker
        # (same pattern as N5's ARBITER_GRADED) so the key pickup is provable
        # in-cloud. NEVER print the key — only configured/not + a masked prefix.
        _tav = os.environ.get("TAVILY_API_KEY", "")
        print(
            f"TAVILY_SECRET: configured={bool(_tav)} newly_applied={applied}",
            flush=True,
        )
        if applied:
            yield {"event": "tavily_configured", "source": "secrets_manager"}
    except Exception as e:  # never let secret resolution break a run
        print(f"TAVILY_SECRET: error {type(e).__name__}", flush=True)
        yield {"event": "tavily_unconfigured", "detail": f"{type(e).__name__}: {e}"}

    if "spec_path" in payload:
        spec = load_spec_from_path(payload["spec_path"])
    elif "spec_text" in payload:
        spec = load_spec_from_text(payload["spec_text"])
    else:
        yield {"event": "error", "error": "payload must include 'spec_path' or 'spec_text'"}
        return

    yield {"event": "spec_loaded", "project_id": spec.project_id}

    # N1/N2: stage the dataset + materialize an aligned y_true on the container.
    data_paths: dict[str, Any] | None = None
    staged_y_true_path: str | None = None
    if payload.get("stage_data"):
        from .cloud.staging import StagingError, stage_dataset

        try:
            staged = stage_dataset(
                spec,
                region=region,
                max_rows=payload.get("max_rows"),
            )
        except StagingError as e:
            yield {"event": "error", "error": f"data staging failed: {e}"}
            return
        data_paths = {
            "train_path": staged.train_path,
            "test_path": staged.test_path,
            "target": staged.target,
        }
        staged_y_true_path = staged.y_true_path
        # Publish the staged pair so ``dispatch_preflight``'s staged_test_alignment check can find
        # it with no argument from the agent. y_true is materialised once here and never
        # regenerated, so the check exists to catch a later in-place rewrite of test.csv before a
        # job is dispatched against it.
        os.environ["AMLAS_STAGED_TEST_PATH"] = staged.test_path
        os.environ["AMLAS_STAGED_Y_TRUE_PATH"] = staged.y_true_path
        yield {
            "event": "data_staged",
            "train_path": staged.train_path,
            "test_path": staged.test_path,
            "y_true_path": staged.y_true_path,
            "target": staged.target,
            "n_train": staged.n_train,
            "n_test": staged.n_test,
            "source": staged.source,
        }

        # N1.b: the runtime PROCESS fs (where data_integrity's pandas tools read)
        # and the Code Interpreter SANDBOX fs (where tabular_fe/trainer_tuner run
        # via execute_python_tool) are SEPARATE. stage_dataset wrote only the
        # process fs, so the sandbox sees "no data file" (CW 2026-05-29 13:43).
        # Mirror the staged CSVs into the hoisted sandbox session at the SAME
        # paths so the agents' execute_python_tool calls find them. Best-effort:
        # if the sandbox can't be reached we still yield an event and let D1
        # decide — never crash the run.
        try:
            from .cloud.sandbox_staging import stage_files_into_sandbox

            sandbox_result = stage_files_into_sandbox(
                {staged.train_path: staged.train_path, staged.test_path: staged.test_path},
                region=region,
            )
            yield {
                "event": "data_staged_sandbox",
                "ok": all(sandbox_result.values()),
                "results": sandbox_result,
            }
        except Exception as e:
            yield {"event": "data_staged_sandbox", "ok": False, "detail": f"{type(e).__name__}: {e}"}

    # Build the Lead hierarchy. ``lead_for`` routes each Lead to its managed AgentCore harness
    # when that Lead is opted in via ``AMLAS_HARNESS_LEADS`` AND its harness ARN resolves
    # (ADR-0007 phase 1); otherwise it builds exactly the same in-process Agent as before, with
    # real Bedrock models. With the flag unset this is byte-for-byte the previous behaviour —
    # the cutover is a decision, never a side effect of HarnessStack being deployed.
    #
    # ``actor_id`` is the managed-memory scope: facts namespace under
    # ``/actors/{actorId}/facts/``, so passing the project id is what makes long-term memory
    # carry over per project instead of being unscoped. ``run_id`` is deliberately NOT passed:
    # the Orchestrator owns run-id generation and is constructed below, and each adapter is
    # created per invocation anyway, so its session ledger is already per-run.
    from .harness.leads import HarnessLead, lead_for

    research_lead = lead_for("research_lead", actor_id=spec.project_id)
    fe_lead = lead_for("fe_lead", actor_id=spec.project_id)
    ml_lead = lead_for("ml_execution_lead", actor_id=spec.project_id)
    # Emit which Leads are harness-hosted, so a run's transcript records where each one ran
    # rather than leaving it to be inferred from the environment afterwards.
    for _lead in (research_lead, fe_lead, ml_lead):
        if isinstance(_lead, HarnessLead):
            yield {"event": "lead_harness_hosted", "lead": _lead.lead_name}

    # N2: resolve y_true — prefer the staged (aligned) one; else the operator's
    # explicit path (local OR s3://). The Arbiter needs this for grading.
    y_true_arg = payload.get("y_true_path") or staged_y_true_path
    if not y_true_arg:
        yield {"event": "error", "error": "payload must include 'y_true_path' (or set stage_data=true)"}
        return

    from .cloud.sandbox_staging import load_predictions
    from .cloud.staging import load_y_true_npy

    def _y_true_loader() -> Any:
        # N2 contract: local path OR s3:// URI. FIX-19b (Subtask-13): route BOTH branches
        # through ``load_y_true_npy`` — its ``_load_npy_secure`` recovers an object-dtype
        # STRING-label y_true (string-label classification ``fault_class``) via the SAME restricted unpickler as
        # y_pred, instead of a raw ``np.load(allow_pickle=False)`` that raised
        # "Object arrays cannot be loaded" and FAILED the whole run before grading (run #5).
        return load_y_true_npy(str(y_true_arg), region=region)

    def _y_pred_loader(path: str) -> Any:
        # N5: the ML Lead's trainer_tuner writes y_pred INSIDE the Code
        # Interpreter SANDBOX fs (e.g. /tmp/amlas/y_pred_test.csv), but this
        # loader runs in the runtime PROCESS fs — two separate filesystems
        # (CW 2026-05-29 14:04:44 FileNotFoundError). load_predictions egresses
        # the file OUT of the sandbox (reverse of N1.b) when it's absent on the
        # process fs, and is format-agnostic (.npy via np.load, .csv via a robust
        # parser) so the Arbiter can grade whatever the agent emitted.
        # Instrumentation (N5 verification): these prints land in CloudWatch so
        # the sandbox→process bridge + Arbiter grading are observable server-side.
        print(f"N5_YPRED_LOADER: loading y_pred from {path!r}", flush=True)
        preds = load_predictions(path, egress=True, region=region)
        try:
            n = len(preds)
        except TypeError:
            n = -1
        print(f"N5_YPRED_LOADED: ok n={n} dtype={getattr(preds, 'dtype', '?')}", flush=True)
        return preds

    orch = Orchestrator(
        spec=spec,
        research_lead=research_lead,
        fe_lead=fe_lead,
        ml_lead=ml_lead,
        y_true_loader=_y_true_loader,
        y_pred_loader=_y_pred_loader,
        data_paths=data_paths,  # N1: thread concrete train/test/target into the problem card
    )
    yield {"event": "run_started", "run_id": orch.run_id}

    # FIX-15 (Subtask-5 durable server-side termination guard) — the decisive fix.
    #
    # Root cause of the Subtask-4 stall AND the Subtask-5 v2 idle-close: this
    # entrypoint used to call the SYNCHRONOUS ``orch.run()`` inline, so the async
    # generator yielded NOTHING between ``run_started`` and the final
    # ``run_completed``. On a long agent run that gap spans many minutes; AgentCore
    # then (a) never sees a terminal event if the loop hangs (Subtask-4: client
    # blocks to read_timeout with 0 usable events) or (b) closes the idle SSE
    # stream (Subtask-5 v2: client got a clean EOF at ~128s, no terminal event,
    # while the server was still working).
    #
    # The fix runs ``orch.run()`` on a background worker thread (via the event
    # loop's default executor — a SINGLE consistent thread for the whole run, so
    # the nested Strands + Code-Interpreter sandbox execution keeps its thread
    # context, exactly as when run() ran inline) while the generator:
    #   1. emits periodic ``heartbeat`` events so the SSE stream never goes idle;
    #   2. ALWAYS yields a terminal ``run_completed`` — when run() finishes with
    #      its real result, OR when the HARD wall-clock deadline elapses first
    #      (status=budget_exhausted, reason=wall_clock_exceeded), then returns.
    # The worker is a daemon thread; the Runtime session ends when this generator
    # returns, so an overrun run() cannot keep the container alive.
    #
    # IMPLEMENTATION NOTE: we deliberately use a DAEMON ``threading.Thread`` (not
    # ``loop.run_in_executor`` — whose default ThreadPoolExecutor threads are
    # NON-daemon and would make ``asyncio.run()`` / the process BLOCK on join
    # against a hung ``orch.run()`` at exit, keeping the container alive forever).
    # A daemon thread lets the generator return and the Runtime session end even
    # when the run overran. We poll the result box with ``asyncio.sleep`` so the
    # event loop stays free to emit heartbeats.
    import asyncio
    import threading
    import time as _time

    # FIX-26: the deadline is OPT-IN, not a default cap. Precedence:
    #   1. AMLAS_MAX_RUN_SECONDS (ops escape hatch + deterministic sub-minute test deadline);
    #   2. else a POSITIVE spec ``wall_clock_minutes`` the operator explicitly set;
    #   3. else UNBOUNDED — the run goes to natural completion (``hard_deadline = None``).
    # NO absolute ceiling clamps the operator's number. The heartbeat + terminal-event
    # guarantee are infra scaffolding (SSE keep-alive), independent of any work cap.
    run_start = _time.monotonic()
    _explicit_secs = _env_float("AMLAS_MAX_RUN_SECONDS", 0.0)
    spec_minutes = float(getattr(spec.compute_budget, "wall_clock_minutes", 0) or 0)
    if _explicit_secs > 0:
        max_run_seconds: float | None = _explicit_secs
    elif spec_minutes > 0:
        max_run_seconds = spec_minutes * 60.0
    else:
        max_run_seconds = None  # unbounded — run to natural completion
    hard_deadline = (run_start + max_run_seconds) if max_run_seconds is not None else None
    heartbeat_interval = _env_float("AMLAS_HEARTBEAT_SECONDS", 15.0)
    poll_interval = min(heartbeat_interval, 1.0)
    print(
        f"FIX26_TERMINATION_GUARD: wall_clock_minutes={spec.compute_budget.wall_clock_minutes} "
        f"max_run_seconds={'unbounded' if max_run_seconds is None else f'{max_run_seconds:.0f}'} "
        f"heartbeat_seconds={heartbeat_interval}",
        flush=True,
    )

    run_box: dict[str, Any] = {}

    def _run_worker() -> None:
        try:
            run_box["result"] = orch.run()
        except BaseException as e:  # capture so the generator can report it as a terminal error
            run_box["error"] = e
        finally:
            run_box["done"] = True

    worker = threading.Thread(target=_run_worker, name=f"orch-run-{orch.run_id}", daemon=True)
    worker.start()

    def _terminal_from_result(res: Any) -> dict[str, Any]:
        return {
            "event": "run_completed",
            "status": res.status,
            "iterations": res.iterations,
            "arbiter_report": (res.arbiter_report.model_dump(mode="json") if res.arbiter_report else None),
            "kanban_path": str(orch.run_dir / "kanban.json"),
        }

    def _last_chance_arbiter_report() -> dict[str, Any] | None:
        """FIX-20c (Subtask-13): grade the DURABLE y_pred mirror when the run is abandoned.

        On the HARD wall-clock deadline (or a run-level error) the daemon ``orch.run()`` is
        abandoned mid-iteration, so the orchestrator's own durable-mirror grade (FIX-20b) never
        runs and the terminal event would carry ``arbiter_report: null`` even though the agent
        already produced a real y_pred that FIX-20 mirrored to the process fs (string-label classification run #6
        on v8: CV f1_macro=0.7942, y_pred mirrored x2, yet arbiter_report=null because the 50-min
        deadline fired while run() was still executing). Here — from the entrypoint, on the SAME
        durable mirror — we grade that y_pred so the terminal event carries a REAL arbiter_report
        (G-METRIC/G-A3). Best-effort: NEVER raises, returns ``None`` when there is no durable
        y_pred or grading is not possible (the guard's terminal event is emitted regardless)."""
        try:
            from .orchestrator import _durable_ypred_path

            path = _durable_ypred_path()
            if not path:
                return None
            y_true = _y_true_loader()
            y_pred = _y_pred_loader(path)
            # orch._grade is the deterministic module ``grade`` (ADR-003) — same decision the
            # in-loop arbiter would make; identical, non-delegated.
            report = orch._grade(spec=orch.spec, y_true=y_true, y_pred=y_pred, phase_log=[])
            print(
                f"FIX20C_LASTCHANCE_GRADED: overall_score={report.overall_score} "
                f"direction={report.direction} passes_acceptance={report.passes_acceptance} "
                f"(graded the durable y_pred mirror on the abandoned-run terminal path)",
                flush=True,
            )
            return report.model_dump(mode="json")  # type: ignore[no-any-return]
        except Exception as e:  # pragma: no cover — last-chance grade must never break the terminal
            print(f"FIX20C_LASTCHANCE_SKIPPED: {type(e).__name__}: {e}", flush=True)
            return None

    heartbeats = 0
    last_heartbeat = run_start
    terminal_emitted = False
    while True:
        if run_box.get("done"):
            # run() finished (or raised) within budget — emit its REAL outcome.
            if "error" in run_box:
                err = run_box["error"]
                print(f"FIX15_RUN_ERROR: {type(err).__name__}: {err}", flush=True)
                # : funnel the terminal run error into the centralized surface
                # (local JSONL + CloudWatch mirror), classified by the taxonomy so the
                # terminal ``reason`` carries a stable error code. Best-effort.
                try:
                    from .errors import classify
                    from .telemetry.error_log import capture_error

                    classified = classify(err, source="agents.orchestrator")
                    capture_error(
                        err if isinstance(err, BaseException) else Exception(str(err)),
                        source="agents.orchestrator",
                        error_type=classified.code,
                        run_id=str(getattr(orch, "run_id", "") or ""),
                        context={"category": str(classified.category), "retryable": classified.retryable},
                    )
                except Exception:  # nosec B110
                    pass
                yield {
                    "event": "run_completed",
                    "status": "failed",
                    "reason": f"{type(err).__name__}: {err}",
                    "iterations": 0,
                    # FIX-20c: even on a run-level error, if the agent already produced a
                    # y_pred that FIX-20 mirrored durably, grade it so the terminal carries
                    # a REAL arbiter_report instead of null.
                    "arbiter_report": _last_chance_arbiter_report(),
                    "kanban_path": str(orch.run_dir / "kanban.json"),
                }
            else:
                yield _terminal_from_result(run_box["result"])
            terminal_emitted = True
            break

        now = _time.monotonic()
        if hard_deadline is not None and now >= hard_deadline:
            # HARD wall-clock deadline hit while run() is still executing. Emit a
            # graded terminal event so the client ALWAYS receives a terminal event
            # (never a hung/idle-closed stream). The daemon worker is abandoned;
            # the Runtime session ends when this generator returns.
            print(
                f"FIX15_WALL_CLOCK_EXCEEDED: emitting terminal budget_exhausted after "
                f"{heartbeats} heartbeats (run still executing)",
                flush=True,
            )
            # FIX-20c: the daemon run() is abandoned mid-iteration, so the orchestrator's
            # own durable-mirror grade (FIX-20b) never runs. Grade the durable y_pred mirror
            # HERE so the terminal event carries a REAL arbiter_report (G-METRIC/G-A3) when the
            # agent already produced a y_pred that FIX-20 mirrored to the process fs.
            yield {
                "event": "run_completed",
                "status": "budget_exhausted",
                "reason": "wall_clock_exceeded",
                "iterations": orch.spec.compute_budget.max_retreats + 1,
                "arbiter_report": _last_chance_arbiter_report(),
                "kanban_path": str(orch.run_dir / "kanban.json"),
            }
            terminal_emitted = True
            break

        # Emit a heartbeat when the interval elapses (keeps the SSE stream alive).
        if now - last_heartbeat >= heartbeat_interval:
            heartbeats += 1
            last_heartbeat = now
            yield {"event": "heartbeat", "elapsed_seconds": round(now - run_start, 1), "n": heartbeats}

        # Yield control back to the event loop briefly, then re-check. Unbounded runs
        # (no deadline) just sleep the poll interval between heartbeat/result checks.
        if hard_deadline is None:
            await asyncio.sleep(poll_interval)
        else:
            await asyncio.sleep(min(poll_interval, max(0.0, hard_deadline - _time.monotonic())))

    # Defensive belt: guarantee a terminal event was emitted (should always be).
    if not terminal_emitted:  # pragma: no cover — unreachable by construction
        yield {
            "event": "run_completed",
            "status": "budget_exhausted",
            "reason": "guard_fallthrough",
            "iterations": 0,
            "arbiter_report": None,
            "kanban_path": str(orch.run_dir / "kanban.json"),
        }


if __name__ == "__main__":  # pragma: no cover — entered only in AgentCore Runtime container
    app.run()
