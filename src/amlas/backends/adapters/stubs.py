"""Future-milestone port adapters — Protocols defined now, behavior stubbed.

A port whose behavior does NOT yet exist gets its Protocol defined (in ``ports.py``)
plus an adapter STUB here that raises ``NotImplementedError`` naming the milestone
that fills it — so the registry can resolve every one of the 25 ports and the
Protocol-conformance tests have a concrete class to check.

**Only THREE stubs remain**: ``ModelBackend``, ``SandboxBackend`` and
``RuntimeBackend`` (all "EXISTING seam, concrete wrap deferred" — see Group 5). The
original M0 set was larger; each graduation is recorded as a "NO LONGER stubbed here"
note below, and ``validation coverage`` pins both counts so the
header cannot drift from the body again. Two earlier counts in this header were stale by
eleven graduations and by six ports respectively, while the body directly beneath recorded
every graduation — which is exactly how a prose count rots unnoticed.

Milestone map — the ``M*`` labels below are roadmap milestone tags. **The roadmap
document they came from ("doc 05") is not tracked in this repo**, so the labels are
historical markers only: they record the intended sequence, and cannot be validated
against anything. See ``docs/adr/README.md`` for the provenance situation.
  * ``TuningBackend``            → M3 (Optuna→AMT)
  * ``AutoMLBackend``            → M3 (Autopilot generate-only)
  * ``ModelHubBackend``          → M6 (JumpStart / Bedrock FM)
  * ``ETLBackend``               → DONE in M2 (``adapters/etl.py``: InProcessPandas /
    ProcessingJob / GlueETL); NO LONGER stubbed here.
  * ``DataQualityBackend``       → DONE in M2 (``adapters/dataquality.py``: InProcessDQ /
    DataBrew / GlueDataQuality); NO LONGER stubbed here.
  * ``QueryBackend``             → M6 (Athena / DuckDB)
  * ``LabelingBackend``          → M6 (Ground Truth)
  * ``DeploymentBackend``        → M4 (THE priority GAP — Endpoints)
  * ``OrchestrationBackend``     → DONE in M1 (``adapters/orchestration.py``
    ``LocalDriverBackend`` — the retreat loop + absorbed MLPipeline no-LLM mode);
    NO LONGER stubbed here.
  * ``GovernanceBackend``        → M4/M6 (Model Cards / lineage / LakeFormation)
  * ``GuardrailBackend``         → M4 (Bedrock Guardrails)
  * agent-runtime (Model/Memory/Sandbox/Runtime) → EXISTING seams, wrapped M1/M5

Each stub is registered as BOTH the default and local selector for its port so the
registry resolves consistently; ``__getattr__`` raises the milestone TODO for any
port method called.
"""

from __future__ import annotations

from typing import Any

from .. import ports as _ports
from ..registry import register_adapter


class _MilestoneStub:
    """A port adapter whose behavior lands in a later milestone.

    Constructing it is cheap (the registry can resolve + Protocol-check it); calling
    ANY method raises ``NotImplementedError`` with the milestone + port named, so a
    flow that reaches future behavior fails loudly with guidance rather than silently.
    """

    def __init__(self, *, port: str, milestone: str, note: str = "") -> None:
        self._port = port
        self._milestone = milestone
        self._note = note

    def __getattr__(self, name: str) -> Any:
        # Only invoked for attributes not found normally (so _port/_milestone are safe).
        def _raise(*_a: Any, **_k: Any) -> Any:
            extra = f" {self._note}" if self._note else ""
            raise NotImplementedError(
                f"{self._port}.{name}() is not implemented at M0 — it lands in {self._milestone}.{extra}"
            )

        return _raise


def _register_stub(port: type, milestone: str, *, note: str = "") -> None:
    """Register a milestone stub as both ``default`` and ``local`` for ``port``."""
    pname = getattr(port, "__name__", str(port))

    def _factory() -> _MilestoneStub:
        return _MilestoneStub(port=pname, milestone=milestone, note=note)

    register_adapter(port, "default", _factory, is_default=True)
    register_adapter(port, "local", _factory, is_local=True)


# ── Group 1 — ML-compute (ModelHub) ───────────────────────────────────────────
# NOTE: ``TuningBackend`` + ``AutoMLBackend`` are NO LONGER stubbed here — M3 landed the
# real ``adapters/tuning.py`` (Optuna/SageMakerAMT) and ``adapters/automl.py``
# (LocalSweep/AutopilotV2), imported BEFORE ``stubs`` and registered as their own
# default/local. Re-stubbing would overwrite the real defaults (idempotent
# ``register_adapter`` last-write-wins), so the stubs are removed.
# NOTE: ``ModelHubBackend`` is NO LONGER stubbed here — M6 landed the real
# ``adapters/modelhub.py`` (LocalModelHubBackend static catalog / JumpStartBedrockModelHubBackend),
# imported BEFORE ``stubs`` and registered as its own default/local. The stub is removed.

# ── Group 2 — data/feature (Query/Labeling) ───────────────────────────────────
# NOTE: ``ETLBackend`` + ``DataQualityBackend`` are NO LONGER stubbed here — M2 landed
# the real ``adapters/etl.py`` (InProcessPandas/ProcessingJob/GlueETL) and
# ``adapters/dataquality.py`` (InProcessDQ/DataBrew/GlueDataQuality), imported BEFORE
# ``stubs`` and registered as their own default/local. Re-stubbing would overwrite the
# real defaults (idempotent ``register_adapter`` last-write-wins), so the stubs are removed.
# NOTE: ``QueryBackend`` (M6 ``adapters/query.py`` DuckDB/Athena) + ``LabelingBackend``
# (M6 ``adapters/labeling.py`` Stub/GroundTruth) + ``RetrievalBackend`` (M6
# ``adapters/retrieval.py`` Local/BedrockKB — the 20th port, ADR-017) are NO LONGER
# stubbed here — they landed real in M6, imported BEFORE ``stubs``. The stubs are removed.

# ── Group 3 — deploy (Deployment) — the priority GAP ──────────────────────────
# NOTE: ``DeploymentBackend`` is NO LONGER stubbed here — M4 landed the real
# ``adapters/deployment.py`` (``LocalServeBackend`` in-process server +
# ``SageMakerDeploymentBackend`` managed: model/endpoint-config/endpoint/serverless/
# async/MME/transform/canary/recommender/Neo). It is imported BEFORE ``stubs`` and
# registered as its own default(sagemaker)/local. Re-stubbing would overwrite the real
# defaults (idempotent last-write-wins), so the stub is removed (the GAP is closed).

# ── Group 4 — governance / guardrail ──────────────────────────────────────────
# NOTE: ``OrchestrationBackend`` is NO LONGER stubbed here — M1 landed the real
# ``LocalDriverBackend`` (``adapters/orchestration.py``), which the adapters package
# imports BEFORE ``stubs`` and registers as default+local. Re-stubbing it here would
# overwrite the real adapter (idempotent ``register_adapter`` last-write-wins), so the
# stub is removed.
# NOTE: ``GovernanceBackend`` + ``GuardrailBackend`` are NO LONGER stubbed here — M4 landed
# the minimal-real ``adapters/governance.py`` (Local/SageMaker Governance + InProcess/Bedrock
# Guardrail) for the Stage-9 readiness gates, imported BEFORE ``stubs`` and registered as
# their own default/local. Full breadth (Model Cards/lineage/LakeFormation, Bedrock Guardrails
# create/bind/monitor) graduates in M6/M8; the stubs are removed.

# ── Group 5 — agent-runtime (EXISTING seams; concrete wrap graduates later) ───
# These already work today via other modules (Strands provider seam, memory/hooks,
# code_interpreter, cloud/dispatch). A stub is recorded so the registry can resolve all
# 25 ports; the concrete wrapping is still deferred. These three are the ONLY stubs left.
_register_stub(_ports.ModelBackend, "M1", note="EXISTING Strands provider seam (_build.py).")
# NOTE: ``MemoryBackend`` is NO LONGER stubbed here — M5 landed the real
# ``LocalMemoryBackend`` (``adapters/monitoring.py``: the offline AgentCore-Memory
# carry-over fallback that closes the improvement loop), imported BEFORE ``stubs`` and
# registered as default+local. Re-stubbing would overwrite the real default (idempotent
# last-write-wins), so the stub is removed. The managed AgentCore adapter layers on top (P1).
_register_stub(_ports.SandboxBackend, "M1", note="EXISTING tools/code_interpreter.py.")
_register_stub(_ports.RuntimeBackend, "M1", note="EXISTING cloud/dispatch.py InvokeAgentRuntime.")
