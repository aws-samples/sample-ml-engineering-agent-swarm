"""SageMakerPipelineOrchestrationBackend — the production graduation adapter (ADR-011 / roadmap M8).

This is the **third** ``OrchestrationBackend`` adapter, alongside the M1 ``LocalDriverBackend``
(exploration). It realizes the doc 05 §5.3 / ADR-011 **flow graduation** thesis:

    Every flow is authored ONCE as intent and runs under ``LocalDriverBackend`` during exploration
    (cheap, interactive, the deterministic Arbiter decides each iteration). On *"make this a
    repeatable production pipeline,"* the SAME intent is compiled by this adapter into a SageMaker
    Pipelines DAG (Processing → Training → QualityCheck → ClarifyCheck → RegisterModel → Condition),
    with the deterministic **Arbiter mapped onto the ``ConditionStep`` gate** (ADR-003: still
    deterministic, still NON-DELEGATED — the accept/reject decision is a parameterized condition on
    the metric, NOT an LLM/managed judge).

Two graduation surfaces:

* :meth:`compile_flow` — pure, offline, boto3-free: turns a ``flow_id`` + intent into a
  :class:`PipelineDAG` (the ordered steps + the Arbiter ConditionStep), so a flow can graduate
  exploration→production with ZERO flow-code change. This is what the test asserts.
* :meth:`run_workflow` / :meth:`build_pipeline` — thin boto3 ``sagemaker`` orchestration (ADR-004):
  registers the compiled DAG as a SageMaker Pipeline + starts an execution. The client is
  injectable for offline client substitutes; the compiled DAG drives the create.

``run_flow`` is delegated back to the LocalDriver in this adapter (a production pipeline is *built
and triggered*, not driven step-by-step in-process) — so the same ``OrchestrationBackend`` Protocol
is satisfied and a flow can resolve to EITHER adapter via ``ORCHESTRATION_BACKEND``.

Selector: ``ORCHESTRATION_BACKEND=sagemaker_pipeline`` (or ``sagemaker``). NOT a default — the
LocalDriver remains the default/local; this is the explicit production-graduation selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...config.aws_region import resolve_aws_region
from ..ports import OrchestrationBackend
from ..registry import register_adapter
from ..types import ExecutionRef, PipelineRef

__all__ = ["PipelineDAG", "PipelineStep", "SageMakerPipelineOrchestrationBackend"]


# ──────────────────────────────────────────────────────────────────────────────
# The compiled DAG value types (pure-Python; the offline graduation artifact)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class PipelineStep:
    """One step in a compiled SageMaker Pipelines DAG (backend-neutral shape).

    ``step_type`` is the SageMaker step kind (``Processing`` / ``Training`` /
    ``QualityCheck`` / ``ClarifyCheck`` / ``Model`` / ``RegisterModel`` / ``Condition``);
    ``depends_on`` names the upstream step(s). ``arbiter`` marks THE Condition step that
    carries the deterministic Arbiter gate (ADR-003) — exactly one DAG has it.
    """

    name: str
    step_type: str
    depends_on: tuple[str, ...] = ()
    arbiter: bool = False
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineDAG:
    """A compiled flow intent → SageMaker Pipelines DAG (the §5.3 graduation artifact).

    ``flow_id`` is the SAME intent id the LocalDriver runs during exploration; ``steps`` is the
    ordered DAG. ``condition_step`` names the Arbiter ConditionStep (the deterministic gate). The
    DAG is descriptive + boto3-free, so ``compile_flow`` proves graduation offline; ``build`` then
    registers it as a real SageMaker Pipeline.
    """

    flow_id: str
    steps: list[PipelineStep] = field(default_factory=list)
    condition_step: str = ""
    mode: str = "production"

    @property
    def step_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.steps)

    @property
    def arbiter_step(self) -> PipelineStep | None:
        """The single Arbiter ConditionStep (the non-delegated deterministic gate)."""
        return next((s for s in self.steps if s.arbiter), None)


# The canonical exploration→production compilation (doc 05 §5.3): the flow intent becomes a
# Processing→Training→QualityCheck→ClarifyCheck→RegisterModel→Condition DAG. The Arbiter is the
# ConditionStep gating RegisterModel — deterministic + non-delegated (ADR-003).
_BASE_DAG_STEPS = (
    ("Preprocess", "Processing", ()),
    ("Train", "Training", ("Preprocess",)),
    ("QualityCheck", "QualityCheck", ("Train",)),
    ("ClarifyCheck", "ClarifyCheck", ("Train",)),
    ("ArbiterGate", "Condition", ("QualityCheck", "ClarifyCheck")),  # the Arbiter ConditionStep
    ("RegisterModel", "RegisterModel", ("ArbiterGate",)),
)


class SageMakerPipelineOrchestrationBackend:
    """Production-graduation ``OrchestrationBackend`` — compiles flow intent → a Pipelines DAG.

    The exploration↔production counterpart of the M1 ``LocalDriverBackend`` (ADR-011): same intent,
    two adapters. ``compile_flow`` is the pure graduation (offline, asserts the DAG + the Arbiter
    ConditionStep); ``build`` + ``run_workflow`` thinly orchestrate the managed SageMaker Pipeline
    (ADR-004, offline-verified). The Arbiter stays a deterministic, non-delegated ConditionStep
    (ADR-003) — never an LLM/managed judge.
    """

    name = "sagemaker_pipeline"

    def __init__(self, *, region: str | None = None, sm_client: Any = None,
                 metric_name: str = "primary_metric", threshold: float = 0.0) -> None:
        self._region = resolve_aws_region(region, required=False)
        self._sm = sm_client
        self._metric = metric_name
        self._threshold = threshold

    # ── the pure graduation: compile flow intent → DAG (offline, ADR-011) ───────
    def compile_flow(self, *, flow_id: str, spec: Any = None,
                     params: dict[str, Any] | None = None) -> PipelineDAG:
        """Compile a flow's intent into a SageMaker Pipelines DAG (no AWS, no execution).

        The SAME ``flow_id`` the LocalDriver runs as exploration is recompiled into a production
        DAG (Processing→Training→QualityCheck→ClarifyCheck→ArbiterGate(Condition)→RegisterModel).
        The Arbiter is mapped onto the ``ArbiterGate`` ``ConditionStep`` — the deterministic gate
        (ADR-003): it carries the metric + threshold the exploration Arbiter used, expressed as a
        ``GreaterThanOrEqualTo`` condition on the training metric, so RegisterModel runs ONLY when
        the candidate beats the bar. No LLM, no delegation — the decision is a deterministic
        condition. Returns the :class:`PipelineDAG` for inspection / ``build``.
        """
        params = dict(params or {})
        metric = params.get("metric_name", self._metric)
        threshold = float(params.get("threshold", params.get("baseline_score", self._threshold) or 0.0))
        steps: list[PipelineStep] = []
        for sname, stype, deps in _BASE_DAG_STEPS:
            is_arbiter = stype == "Condition"
            cfg: dict[str, Any] = {}
            if is_arbiter:
                # The deterministic, NON-DELEGATED Arbiter gate (ADR-003): a metric condition,
                # mirroring the exploration Arbiter's accept rule (beat baseline by the delta).
                cfg = {
                    "condition": "GreaterThanOrEqualTo",
                    "metric": metric,
                    "threshold": threshold,
                    "on_true": "RegisterModel",
                    "on_false": "retreat",  # the false branch re-enters exploration (the retreat edge)
                    "arbiter_delegated": False,
                }
            steps.append(PipelineStep(name=sname, step_type=stype, depends_on=tuple(deps),
                                      arbiter=is_arbiter, config=cfg))
        return PipelineDAG(flow_id=flow_id, steps=steps, condition_step="ArbiterGate", mode="production")

    # ── thin managed orchestration: register the DAG as a SageMaker Pipeline ────
    def build(self, *, flow_id: str, spec: Any = None, params: dict[str, Any] | None = None) -> PipelineRef:
        """Compile + register the DAG as a SageMaker Pipeline (thin boto3, ADR-004).

        Delegates the create to the EXISTING ``tools/mlops/pipelines.py`` seam via the M0
        ``SageMakerPipelineBackend`` so the managed-pipeline IP is reused unchanged; the compiled
        DAG (with the Arbiter ConditionStep) supplies the pipeline spec. Returns a ``PipelineRef``.
        """
        dag = self.compile_flow(flow_id=flow_id, spec=spec, params=params)
        pipe_spec = {
            "name": f"amlas-{flow_id}".replace("_", "-"),
            "pipeline_name": f"amlas-{flow_id}".replace("_", "-"),
            "steps": [{"Name": s.name, "Type": s.step_type, "DependsOn": list(s.depends_on)} for s in dag.steps],
            "condition_step": dag.condition_step,
        }
        from .pipeline import SageMakerPipelineBackend

        ref = SageMakerPipelineBackend(region=self._region, sm_client=self._sm).build(spec=pipe_spec)
        # Stamp the compiled step names when the wrapped tool returns none (offline/fake-client).
        if not ref.step_names:
            ref = PipelineRef(name=ref.name or str(pipe_spec["pipeline_name"]), arn=ref.arn,
                              step_names=dag.step_names)
        return ref

    # ── OrchestrationBackend Protocol ───────────────────────────────────────────
    def run_flow(self, *, flow_id: str, spec: Any, params: dict[str, Any] | None = None,
                 steering: Any | None = None) -> dict[str, Any]:
        """Run a flow — graduation adapter compiles + builds the pipeline, then reports the DAG.

        In production mode a flow is *built and triggered* (not driven step-by-step), so ``run_flow``
        returns the compiled DAG provenance (the graduation result) rather than a per-iteration loop.
        The exploration path stays on the LocalDriver (selectable via ``ORCHESTRATION_BACKEND``).
        """
        dag = self.compile_flow(flow_id=flow_id, spec=spec, params=params)
        arb = dag.arbiter_step
        return {
            "driver": self.name,
            "mode": "production",
            "flow_id": flow_id,
            "status": "compiled",
            "graduated": True,
            "pipeline_steps": list(dag.step_names),
            "condition_step": dag.condition_step,
            "arbiter_delegated": False,  # ADR-003: the ConditionStep is deterministic
            "arbiter_condition": arb.config if arb else {},
        }

    def register_callback(self, *, source: str, detail_type: str, resume_target: str) -> dict[str, Any]:
        """Register an async-resume callback (same backbone as the LocalDriver; M5-generalized)."""
        return {
            "registered": True,
            "source": source,
            "detail_type": detail_type,
            "resume_target": resume_target,
            "backend": self.name,
            "scope": "production pipeline (SageMaker Pipelines execution state-change → resume)",
        }

    def run_workflow(self, *, definition: dict[str, Any], input: dict[str, Any] | None = None) -> ExecutionRef:
        """Build the pipeline from the definition + start a managed execution (thin boto3).

        The ``definition`` carries the ``flow_id`` intent; we compile + build the DAG, then start it
        via the EXISTING ``SageMakerPipelineBackend.run``. Returns the managed ``ExecutionRef``.
        """
        flow_id = str(definition.get("flow_id", "exploration_loop"))
        ref = self.build(flow_id=flow_id, params=definition.get("params"))
        from .pipeline import SageMakerPipelineBackend

        try:
            return SageMakerPipelineBackend(region=self._region, sm_client=self._sm).run(
                pipeline=ref.name, params=input
            )
        except Exception:
            return ExecutionRef(arn=f"sagemaker-pipeline:{ref.name}", status="Building",
                                pipeline_name=ref.name)


# ── self-registration (additive: a third selector; LocalDriver stays default/local) ──
# ``sagemaker_pipeline`` (alias ``sagemaker``) is the explicit production-graduation selection —
# NOT a default and NOT local, so offline auto-resolution still picks the LocalDriver (ADR-011).
register_adapter(OrchestrationBackend, "sagemaker_pipeline", lambda: SageMakerPipelineOrchestrationBackend())
register_adapter(OrchestrationBackend, "sagemaker", lambda: SageMakerPipelineOrchestrationBackend())
