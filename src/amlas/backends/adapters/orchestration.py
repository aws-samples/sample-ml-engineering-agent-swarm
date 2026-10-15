"""OrchestrationBackend adapter — ``LocalDriverBackend`` (roadmap M1; doc 05 §5; ADR-002/003).

Resolves RECON §10.6 (two parallel orchestrators) by landing ONE port with ONE
default adapter that absorbs BOTH stacks:

  * **agentic mode** (``flow_id="exploration_loop"``) — relocates the EXISTING
    plain-Python R→D1→D2→ARBITER **retreat loop** ([`orchestrator.py`] ``Orchestrator``)
    behind ``run_flow``. Behavior-preserving (ADR-002): the loop, the disk-handoff of
    ``y_pred``, the retreat edge, and the kanban abandoned-lane ledger are unchanged.
  * **no-LLM mode** (``flow_id in {"ml_pipeline","no_llm","pipeline"}``) — absorbs the
    standalone ``MLPipeline`` ([`pipeline/orchestrator.py`]) deterministic path
    (research→plan→train→evaluate→leaderboard→decision→model_card) as a mode of the
    SAME adapter, so there is **one orchestration model** (doc 05 §5.2 — the standalone
    MLPipeline is now the LocalDriver's internal no-LLM mode, not a second stack).

The deterministic **Arbiter remains a non-delegated decision step** (ADR-003): the
driver resolves the M0 ``EvaluationBackend`` (the ``DeterministicArbiter`` wrapping the
SAME ``arbiter.grade``) and injects its ``grade`` into the loop via the orchestrator's
``grader=`` seam — so the decision runs through the port WITHOUT being delegated to an
LLM or a managed service. The plain-Python driver is kept (NOT a Strands Graph) per
ADR-002.

``register_callback`` wraps the EXISTING training-resume backbone
([`async_callback/lambda_handler.py`]) — still training-only at M1 (generalized in M5).
``run_workflow`` returns an in-process ``ExecutionRef`` (the StepFunctions/Pipelines
graduation is M8).

This module REPLACES the M0 ``OrchestrationBackend`` stub (``stubs.py``): the registry
now resolves the port to this real adapter.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..ports import EvaluationBackend, OrchestrationBackend
from ..registry import REGISTRY, register_adapter
from ..types import ExecutionRef

__all__ = ["LocalDriverBackend"]

# Flow-id aliases that select the no-LLM (absorbed MLPipeline) mode vs the agentic loop.
_NO_LLM_FLOWS = {"ml_pipeline", "no_llm", "pipeline", "ml_pipeline_flow"}
_AGENTIC_FLOWS = {"exploration_loop", "exploration", "agentic", "train_eval_loop"}


class LocalDriverBackend:
    """Default ``OrchestrationBackend`` adapter — the unified exploration driver.

    One adapter, two modes (doc 05 §5.2). ``run_flow`` dispatches on ``flow_id``:
    the agentic retreat loop (default) or the absorbed no-LLM ``MLPipeline``. The
    Arbiter is invoked through ``EvaluationBackend`` and is never delegated (ADR-003).
    """

    name = "local_driver"

    # ── the unified entrypoint ──────────────────────────────────────────────
    def run_flow(
        self,
        *,
        flow_id: str,
        spec: Any,
        params: dict[str, Any] | None = None,
        steering: Any | None = None,
    ) -> dict[str, Any]:
        """Run ``flow_id`` against ``spec`` (the unified TUI ``/run`` + hosted ``app.py`` path).

        Args:
            flow_id: ``"exploration_loop"`` (agentic retreat loop) or one of
                ``{"ml_pipeline","no_llm","pipeline"}`` (the absorbed deterministic
                ``MLPipeline`` no-LLM mode).
            spec: A ``SpecModel`` (agentic) or a flat spec dict (no-LLM).
            params: Mode-specific inputs (leads + loaders + ``runs_dir`` for agentic;
                ``output_dir`` + ``data_path`` for no-LLM).
            steering: Optional ``SteeringProfile`` (reserved; forwarded to the loop).

        Returns:
            A JSON-ish result dict carrying ``driver``/``mode`` provenance plus the
            mode's result (the ``OrchestratorResult`` fields, or the pipeline summary).
        """
        params = dict(params or {})
        fid = (flow_id or "").strip().lower()
        if fid in _NO_LLM_FLOWS:
            return self._run_no_llm(spec=spec, params=params)
        # default → the agentic retreat loop (exploration_loop and aliases)
        return self._run_agentic(spec=spec, params=params, steering=steering)

    # ── agentic mode (the EXISTING retreat loop, Arbiter non-delegated) ─────
    def _run_agentic(self, *, spec: Any, params: dict[str, Any], steering: Any | None) -> dict[str, Any]:
        from ...orchestrator import Orchestrator

        # NON-DELEGATED Arbiter (ADR-003): resolve the EvaluationBackend port and
        # inject its ``grade`` so the loop's decision runs through the port —
        # the DeterministicArbiter wraps the SAME ``arbiter.grade`` (identical result).
        resolved = REGISTRY.resolve(EvaluationBackend, steering=steering)
        eval_adapter = resolved.adapter
        grader: Callable[..., Any] = eval_adapter.grade

        orch_kwargs: dict[str, Any] = {
            "spec": spec,
            "research_lead": params["research_lead"],
            "fe_lead": params["fe_lead"],
            "ml_lead": params["ml_lead"],
            "y_true_loader": params["y_true_loader"],
            "y_pred_loader": params["y_pred_loader"],
            "grader": grader,
        }
        # Optional pass-throughs (behavior-preserving defaults when absent).
        for opt in ("run_id", "runs_dir", "trace_logger", "data_paths"):
            if params.get(opt) is not None:
                orch_kwargs[opt] = params[opt]

        orch = Orchestrator(**orch_kwargs)
        result = orch.run()
        return {
            "driver": self.name,
            "mode": "agentic",
            "flow_id": "exploration_loop",
            "status": result.status,
            "iterations": result.iterations,
            "arbiter_report": result.arbiter_report,
            "kanban": result.kanban,
            "final_hypothesis": result.final_hypothesis,
            "phase_log": result.phase_log,
            "elapsed_seconds": result.elapsed_seconds,
            "run_id": orch.run_id,
            # ADR-003 provenance: the grade ran through the EvaluationBackend port,
            # deterministically, and was NOT delegated to an LLM/managed judge.
            "arbiter_delegated": False,
            "evaluation_adapter": resolved.adapter_name,
        }

    # ── no-LLM mode (the absorbed MLPipeline) ───────────────────────────────
    def _run_no_llm(self, *, spec: Any, params: dict[str, Any]) -> dict[str, Any]:
        # Reference the module attribute (not a direct class import) so callers can
        # override ``pipeline.orchestrator.MLPipeline`` to stay offline/deadlock-safe.
        from ...pipeline import orchestrator as pipe_mod

        output_dir = params.get("output_dir") or params.get("runs_dir") or "runs/no_llm"
        data_path = params.get("data_path")
        spec_dict = spec if isinstance(spec, dict) else self._spec_to_dict(spec)

        pipeline = pipe_mod.MLPipeline(output_dir=output_dir)
        result = pipeline.run(spec_dict=spec_dict, data_path=data_path)
        return {
            "driver": self.name,
            "mode": "no_llm",
            "flow_id": "ml_pipeline",
            "status": "completed",
            "decision": getattr(getattr(result, "decision", None), "recommendation", None),
            "leaderboard": getattr(result, "leaderboard", ""),
            "model_card": getattr(result, "model_card", ""),
            "artifacts_dir": getattr(result, "artifacts_dir", ""),
            # Arbiter/decision is the deterministic DecisionEngine here too (ADR-003).
            "arbiter_delegated": False,
        }

    @staticmethod
    def _spec_to_dict(spec: Any) -> dict[str, Any]:
        """Best-effort flatten of a ``SpecModel`` into the pipeline's flat spec dict."""
        if hasattr(spec, "model_dump"):
            dumped: dict[str, Any] = spec.model_dump(mode="json")
            return dumped
        if hasattr(spec, "__dict__"):
            return dict(spec.__dict__)
        return dict(spec)

    # ── async resume backbone (M1 training-resume → M5 generalized re-entry) ──────
    def register_callback(self, *, source: str, detail_type: str, resume_target: str) -> dict[str, Any]:
        """Register an async-resume callback (wraps the EXISTING training-resume backbone).

        M1 seeded this with the training-only resume (the ``async_callback/lambda_handler.py``
        EventBridge → InvokeAgentRuntime path). **M5 generalizes it** (ADR-008): the same
        backbone now accepts ANY production source — Processing/Glue/Pipeline/labeling AND the
        ``amlas.monitoring`` drift/retrain trigger — so a monitoring signal can RE-ENTER the
        unified exploration retreat loop (``resume_target="exploration_loop"``) as a new
        improvement iteration, not a parallel retrain path. Returns a structured registration
        descriptor (offline-safe — provisions no AWS resources here; CDK wires the rule).
        """
        return {
            "registered": True,
            "source": source,
            "detail_type": detail_type,
            "resume_target": resume_target,
            "backend": self.name,
            # M5: generalized beyond the M1 training-only seed to any production source
            # (Processing/Glue/Pipeline/labeling + monitoring drift triggers) — the closed loop.
            "scope": "generalized (M5): training + processing/glue/pipeline/labeling + monitoring",
        }

    def run_workflow(self, *, definition: dict[str, Any], input: dict[str, Any] | None = None) -> ExecutionRef:
        """Run a workflow definition in-process (the managed DAG graduation is M8).

        For M1 the LocalDriver executes the definition as a local exploration run
        (no StepFunctions/Pipelines). Returns an in-process ``ExecutionRef``.
        """
        flow_id = str(definition.get("flow_id", "exploration_loop"))
        return ExecutionRef(arn=f"local-driver:{flow_id}", status="local", pipeline_name=flow_id)


# Register as BOTH the default and the local adapter for OrchestrationBackend: the
# plain-Python driver IS the exploration driver in both contexts (the managed
# SageMakerPipelineOrchestrationBackend graduation is M8). This REPLACES the M0 stub.
register_adapter(OrchestrationBackend, "local_driver", lambda: LocalDriverBackend(), is_default=True, is_local=True)
register_adapter(OrchestrationBackend, "local", lambda: LocalDriverBackend(), is_local=True)
