"""PipelineBackend adapters (doc 05 §4.2 G4 / roadmap M0 work-item 8).

* ``SageMakerPipelineBackend`` — **default**; wraps EXISTING
  ``tools/mlops/pipelines.py`` (``build_sagemaker_pipeline`` / ``run_sagemaker_pipeline``).
* ``LocalMLPipelineBackend`` — **local**; wraps the EXISTING deterministic
  ``pipeline/orchestrator.py`` ``MLPipeline`` (the no-LLM offline pipeline the TUI
  ``/run`` drives). M0 keeps ``build`` as a no-AWS definition assembly (via the
  EXISTING pure ``build_pipeline_definition``) and ``run`` delegates to ``MLPipeline``;
  the full orchestration unification graduates in M1 (doc 05 §5).
"""

from __future__ import annotations

import tempfile
from typing import Any

from ..ports import PipelineBackend
from ..registry import register_adapter
from ..types import ExecutionRef, PipelineRef
from ._util import raw_tool


class SageMakerPipelineBackend:
    """Default PipelineBackend — wraps EXISTING ``tools/mlops/pipelines.py``."""

    def __init__(self, *, region: str | None = None, sm_client: Any = None) -> None:
        self._region = region
        self._sm = sm_client

    def build(self, *, spec: dict[str, Any]) -> PipelineRef:
        """Build + register the DAG via EXISTING ``build_sagemaker_pipeline`` (unchanged)."""
        from ...tools.mlops.pipelines import build_sagemaker_pipeline

        out = raw_tool(build_sagemaker_pipeline)(dict(spec or {}), region=self._region, sm_client=self._sm)
        if out.get("success") is False:
            raise RuntimeError(out.get("error", "build_sagemaker_pipeline failed"))
        return PipelineRef(
            name=str(out.get("pipeline_name", "")),
            arn=str(out.get("pipeline_arn", "")),
            step_names=tuple(out.get("step_names", [])),
        )

    def run(self, *, pipeline: str, params: dict[str, Any] | None = None) -> ExecutionRef:
        """Start an execution via EXISTING ``run_sagemaker_pipeline`` (unchanged)."""
        from ...tools.mlops.pipelines import run_sagemaker_pipeline

        out = raw_tool(run_sagemaker_pipeline)(pipeline, params, region=self._region, sm_client=self._sm)
        if out.get("success") is False:
            raise RuntimeError(out.get("error", "run_sagemaker_pipeline failed"))
        return ExecutionRef(
            arn=str(out.get("execution_arn", "")),
            status=str(out.get("status", "")),
            pipeline_name=str(out.get("pipeline_name", pipeline)),
        )

    def status(self, *, execution: str) -> dict[str, Any]:
        """Describe an execution status (best-effort, via the injected client)."""
        from ...tools.mlops._common import make_sm_client

        sm = make_sm_client(self._region, self._sm)
        try:
            d = sm.describe_pipeline_execution(PipelineExecutionArn=execution)
            return {"status": d.get("PipelineExecutionStatus", ""), "raw": d}
        except Exception as e:
            return {"status": "Unknown", "error": str(e)}


class LocalMLPipelineBackend:
    """Local PipelineBackend — wraps EXISTING ``pipeline/orchestrator.py`` MLPipeline.

    ``build`` assembles a definition with the EXISTING pure ``build_pipeline_definition``
    (no AWS); ``run`` executes the deterministic ``MLPipeline`` offline. The full
    orchestration unification (LocalDriver absorbing MLPipeline) is M1 (doc 05 §5).
    """

    def build(self, *, spec: dict[str, Any]) -> PipelineRef:
        """Assemble a local pipeline definition via EXISTING ``build_pipeline_definition``."""
        from ...tools.mlops.pipelines import build_pipeline_definition

        definition = build_pipeline_definition(dict(spec or {}))
        name = str(spec.get("name", spec.get("pipeline_name", "amlas-local-pipeline")))
        return PipelineRef(name=name, arn="", step_names=tuple(s["Name"] for s in definition.get("Steps", [])))

    def run(self, *, pipeline: str, params: dict[str, Any] | None = None) -> ExecutionRef:
        """Run the deterministic MLPipeline offline (the EXISTING no-LLM path)."""
        from ...pipeline.orchestrator import MLPipeline

        p = dict(params or {})
        spec_dict = p.get("spec_dict") or p.get("spec")
        data_path = p.get("data_path", "")
        if spec_dict is None:
            raise ValueError("LocalMLPipelineBackend.run requires params['spec_dict'] + params['data_path']")
        # R-B-003 / AMLAS-SEC-B-003 (CWE-377 / bandit B108): default to a per-run unique 0700
        # tempdir via tempfile.mkdtemp() instead of the predictable shared ``/tmp/amlas_local_pipeline``
        # (which invited symlink/clobber races on multi-user hosts). A caller-supplied ``output_dir``
        # still wins; only the default is now non-predictable + private.
        #
        # The mkdtemp prefix uses HYPHENS (``amlas-local-pipeline-``), NOT the underscore
        # form of the forbidden literal. On a Linux runner the tempdir root is ``/tmp``, so an
        # underscore prefix ``amlas_local_pipeline_`` would produce a path that still CONTAINS
        # the predictable shared substring ``/tmp/amlas_local_pipeline`` the R-B-003 hardening
        # test forbids (on macOS the root is ``/var/folders`` so the substring never appeared,
        # hiding the leak locally). The random mkdtemp suffix already guarantees per-run
        # uniqueness/unpredictability; the hyphenated prefix ensures the exact predictable
        # literal is never reconstructed as a substring either.
        output_dir = p.get("output_dir") or tempfile.mkdtemp(prefix="amlas-local-pipeline-")
        # Run for side effects (artifacts under output_dir); the ExecutionRef below
        # is the adapter's return contract, so the frame itself isn't retained.
        MLPipeline(output_dir=output_dir).run(spec_dict=spec_dict, data_path=data_path)
        return ExecutionRef(arn="local", status="Completed", pipeline_name=pipeline)

    def status(self, *, execution: str) -> dict[str, Any]:
        """Local executions complete synchronously."""
        return {"status": "Completed", "execution": execution}


register_adapter(PipelineBackend, "sagemaker", lambda: SageMakerPipelineBackend(), is_default=True)
register_adapter(PipelineBackend, "local", lambda: LocalMLPipelineBackend(), is_local=True)
