"""The 25 backend **ports** — domain-typed ``Protocol`` interfaces (doc 05 §4.2).

This is the central new construct (the fix for RECON §10.1 "no backend-abstraction
layer"). Every ``@tool`` and opinionated flow depends on a **port**, never boto3
directly (doc 05 §4.4 before/after). A port speaks the *domain* (algo, data,
budget, plan, handle, ref from ``types.py``); the boto3 specifics live in each
adapter under ``backends/<port>/``. ADR-001 / ADR-012.

The five port groups (doc 05 §4.1), totalling **25** formal ports:
  * Group 1 — ML-compute (4): Training, Tuning, AutoML, ModelHub.
  * Group 2 — data/feature (7): ETL, DataQuality, FeatureStore, Lake, Query, Labeling, Retrieval.
  * Group 3 — deploy/serve/eval (3): Deployment, Explainability, Evaluation.
  * Group 4 — MLOps/gov/orchestration (6 + Guardrail): ModelRegistry, Monitoring,
    ExperimentTracking, Pipeline, Orchestration, Governance (+ GuardrailBackend,
    counted within group 4 per doc 06 Part 3 note).
  * Group 5 — agent-runtime (4, mostly pre-existing): Model, Memory, Sandbox, Runtime.

All are ``typing.Protocol`` (structural typing): an adapter "is a" port simply by
implementing the methods — no inheritance required, so wrapping EXISTING code as
an adapter never touches the existing module. ``@runtime_checkable`` is applied so
the registry + callers can assert conformance via ``isinstance`` (note: that checks
method *presence*, not signatures — the conformance tests assert call shape).

**No boto3 imports in this module** (the seam's whole point). Only ``typing`` +
the domain ``types``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from .types import (
    CatalogView,
    Chunk,
    DatasetRef,
    DQScore,
    DriftReport,
    EndpointRef,
    ExecutionRef,
    ExplainReport,
    IndexRef,
    JobHandle,
    JobStatus,
    ModelRef,
    MonitorReport,
    PackageRef,
    PipelineRef,
    ProfileReport,
    RetrievalResult,
    ScheduleRef,
    SizingRec,
)

__all__ = [
    # introspection helpers
    "ALL_PORTS",
    "AutoMLBackend",
    "DataQualityBackend",
    # Group 3 — deploy / serve / eval
    "DeploymentBackend",
    # Group 2 — data / feature
    "ETLBackend",
    "EvaluationBackend",
    "ExperimentTrackingBackend",
    "ExplainabilityBackend",
    "FeatureStoreBackend",
    "GovernanceBackend",
    "GuardrailBackend",
    "LabelingBackend",
    "LakeBackend",
    "MemoryBackend",
    # Group 5 — agent-runtime (pre-existing seams)
    "ModelBackend",
    "ModelHubBackend",
    # Group 4 — MLOps / gov / orchestration (+ Guardrail)
    "ModelRegistryBackend",
    "MonitoringBackend",
    "OrchestrationBackend",
    "PipelineBackend",
    "QueryBackend",
    "RetrievalBackend",
    "RuntimeBackend",
    "SandboxBackend",
    # Group 1 — ML-compute
    "TrainingBackend",
    "TuningBackend",
    "port_key",
]


# ══════════════════════════════════════════════════════════════════════════════
# Group 1 — ML-compute ports (4)
# ══════════════════════════════════════════════════════════════════════════════
@runtime_checkable
class TrainingBackend(Protocol):
    """Submit + track a training run (default SageMaker; local sklearn for small).

    M0 adapters: ``SageMakerTrainingBackend`` (wraps ``tools/sagemaker.py`` 3-step
    DDB submit) + ``LocalSklearnBackend`` (wraps ``train_model``). doc 05 §4.2 G1.
    """

    def submit(
        self,
        *,
        algo: str,
        train_uri: str,
        val_uri: str | None = None,
        target: str | None = None,
        hyperparams: dict[str, Any] | None = None,
        resources: dict[str, Any] | None = None,
        budget: dict[str, Any] | None = None,
    ) -> JobHandle: ...

    def status(self, handle: JobHandle) -> JobStatus: ...

    def metrics(self, handle: JobHandle) -> dict[str, Any]: ...

    def artifact(self, handle: JobHandle) -> ModelRef: ...


@runtime_checkable
class TuningBackend(Protocol):
    """Hyperparameter search (default SageMaker AMT; local Optuna). doc 05 §4.2 G1.

    M2/M3 fills the adapters; M0 defines the Protocol + stub only.
    """

    def search(
        self,
        *,
        base_training: dict[str, Any],
        ranges: dict[str, Any],
        objective: str,
        strategy: str = "bayesian",
        max_jobs: int = 10,
        max_parallel: int = 1,
        warm_start_from: str | None = None,
    ) -> dict[str, Any]: ...


@runtime_checkable
class AutoMLBackend(Protocol):
    """AutoML baseline generation (default Autopilot; local FLAML). doc 05 §4.2 G1.

    M3 fills the adapters; M0 defines the Protocol + stub only.
    """

    def run(
        self,
        *,
        train_uri: str,
        target: str,
        problem_type: str,
        mode: str = "ensembling",
        max_candidates: int = 10,
        generate_only: bool = False,
    ) -> dict[str, Any]: ...


@runtime_checkable
class ModelHubBackend(Protocol):
    """Foundation-model / pretrained-model hub (default JumpStart). doc 05 §4.2 G1.

    M6 fills the adapters; M0 defines the Protocol + stub only.
    """

    def list_models(self, *, task: str | None = None, query: str | None = None) -> list[dict[str, Any]]: ...

    def deploy(self, model_id: str, **opts: Any) -> EndpointRef: ...

    def finetune(
        self, model_id: str, train_uri: str, val_uri: str | None, hyperparams: dict[str, Any] | None
    ) -> JobHandle: ...


# ══════════════════════════════════════════════════════════════════════════════
# Group 2 — data / feature ports (6)
# ══════════════════════════════════════════════════════════════════════════════
@runtime_checkable
class ETLBackend(Protocol):
    """Transforms + crawls (default Glue; local in-process pandas). doc 05 §4.2 G2.

    M2 fills the managed adapters; M0 defines the Protocol + stub only.
    """

    def run_transform(
        self, *, input_uri: str, transform_spec: dict[str, Any], output: str, resources: dict[str, Any] | None = None
    ) -> dict[str, Any]: ...

    def run_crawl(self, *, s3_target: str) -> dict[str, Any]: ...


@runtime_checkable
class DataQualityBackend(Protocol):
    """Profile / validate / **impute** (default DataBrew+GlueDQ; local 6-dim).

    The "swappable module" worked example (doc 05 §4.5). M2 fills the managed
    adapters; M0 defines the Protocol + stub only.
    """

    def profile(self, *, dataset_uri: str) -> ProfileReport: ...

    def evaluate(self, *, dataset_uri: str, ruleset: Any) -> DQScore: ...

    def impute(self, *, dataset_uri: str, plan: dict[str, Any]) -> DatasetRef: ...


@runtime_checkable
class FeatureStoreBackend(Protocol):
    """Online/offline feature store + PIT (default SageMaker FS). doc 05 §4.2 G2.

    M0 adapters: ``SageMakerFeatureStoreBackend`` (wraps ``feature_store_ingest``/
    ``feature_store_get``) + ``GoldTableBackend``. PIT (``pit_dataset``) is the
    M3 extension — defined now, stubbed in the M0 adapters.
    """

    def upsert(self, *, group: str, records: Any) -> int: ...

    def get(self, *, group: str, record_id: str, online: bool = True) -> dict[str, Any]: ...

    def pit_dataset(self, *, group: str, entity_ids: list[str], as_of: str) -> DatasetRef: ...


@runtime_checkable
class LakeBackend(Protocol):
    """Medallion read/write/catalog + the bronze-write rejection (ADR-009).

    M0 adapters: ``S3MedallionBackend`` (wraps ``tools/lake/``) + ``LocalFSBackend``.
    doc 05 §4.2 G2 / §6.
    """

    def read_slice(self, *, name_or_uri: str, columns: list[str] | None = None,
                   filters: dict[str, Any] | None = None, max_rows: int | None = None) -> DatasetRef: ...

    def write_dataset(self, *, local_path: str, dest_uri: str, tier: str = "silver",
                      lineage: dict[str, Any] | None = None, profile: dict[str, Any] | None = None) -> str: ...

    def list_catalog(self, *, database: str | None = None) -> CatalogView: ...


@runtime_checkable
class QueryBackend(Protocol):
    """NL→SQL over the lake (default Athena; local DuckDB). doc 05 §4.2 G2.

    M6 fills the adapters; M0 defines the Protocol + stub only.
    """

    def run_sql(self, *, query: str, database: str | None = None) -> dict[str, Any]: ...


@runtime_checkable
class LabelingBackend(Protocol):
    """Data labeling (default Ground Truth; stub for isolated validation). doc 05 §4.2 G2.

    M6 fills the adapters; M0 defines the Protocol + stub only.
    """

    def create_job(
        self, *, input_manifest: str, task_type: str, categories: list[str],
        workteam: str, ui_template: str | None = None, auto_label: bool = True
    ) -> JobHandle: ...

    def progress(self, handle: JobHandle) -> dict[str, Any]: ...


@runtime_checkable
class RetrievalBackend(Protocol):
    """RAG / vector-retrieval seam — the candidate **20th port** (GAP-PORT-1; ADR-017).

    The genuine multi-implementation seam doc 05 §4.6 / roadmap M6 resolves: index +
    chunk + embed + query + retrieve over a research/knowledge corpus. M6 adapters:
    ``LocalRetrievalBackend`` (offline default — an in-process TF-IDF/hash index over the
    research corpus + ``memory-bank/research/``; no heavy embedding deps that risk the
    macOS-aarch64 deadlock) + ``BedrockKnowledgeBasesBackend`` (managed default — thin
    boto3 over Bedrock Knowledge Bases / OpenSearch Serverless, offline-verified). The
    RAG tools T-RAG-01..06 call this port. doc 05 §4.2 G2 (folds in as the 20th port).
    """

    def chunk(self, *, text: str, chunk_size: int = 512, overlap: int = 64,
              source: str = "", metadata: dict[str, Any] | None = None) -> list[Chunk]:
        """T-RAG-02 — split ``text`` into overlapping :class:`Chunk`s with provenance."""
        ...

    def build_index(self, *, name: str, chunks: list[Chunk]) -> IndexRef:
        """T-RAG-05 — build/replace a retrieval index ``name`` from ``chunks``."""
        ...

    def index_corpus(self, *, name: str, documents: list[dict[str, Any]],
                     chunk_size: int = 512, overlap: int = 64) -> IndexRef:
        """T-RAG-01 — chunk + index a corpus of ``{text, source, metadata}`` docs."""
        ...

    def query(self, *, name: str, query: str, top_k: int = 5,
              filters: dict[str, Any] | None = None) -> list[RetrievalResult]:
        """T-RAG-06 — retrieve the ``top_k`` most relevant chunks for ``query``."""
        ...


# ══════════════════════════════════════════════════════════════════════════════
# Group 3 — deploy / serve / evaluate ports (3)
# ══════════════════════════════════════════════════════════════════════════════
@runtime_checkable
class DeploymentBackend(Protocol):
    """Package + deploy + invoke + batch + canary + rollback (THE priority GAP, ADR-007).

    M4 fills the adapters (``LocalServeBackend`` offline / ``SageMakerDeploymentBackend``
    managed). The port speaks the domain: a ``PackageRef`` artifact + a serving ``mode``
    in → an ``EndpointRef`` out; ``invoke`` returns a prediction; ``rollback``/``delete``
    are the irreversible ops the M1 approval gate wraps (doc 05 §8.8). doc 05 §4.2 G3.
    """

    def package(self, *, model_ref: ModelRef, image: str | None = None,
                inference_code: str | None = None, env: dict[str, Any] | None = None) -> PackageRef:
        """T-DEPLOY-09 — package a trained artifact into a deployable container ref."""
        ...

    def deploy(self, *, model_ref: ModelRef, mode: str = "realtime",
               resources: dict[str, Any] | None = None, autoscaling: dict[str, Any] | None = None) -> EndpointRef:
        """T-DEPLOY-01 — deploy as realtime/serverless/async/multimodel; returns the endpoint."""
        ...

    def invoke(self, *, endpoint: EndpointRef, payload: Any, target_model: str | None = None) -> dict[str, Any]:
        """T-DEPLOY-02 — invoke the endpoint with ``payload``; returns ``{"prediction": ...}``."""
        ...

    def update_canary(self, *, endpoint: EndpointRef, new_config: dict[str, Any],
                      canary_pct: int, alarms: list[str] | None = None, mode: str = "canary") -> dict[str, Any]:
        """T-DEPLOY-03 — shift traffic to ``new_config`` via canary / blue-green."""
        ...

    def batch_score(self, *, model_ref: ModelRef, input_uri: str, output_uri: str,
                    resources: dict[str, Any] | None = None) -> JobHandle:
        """T-DEPLOY-04 — run a batch-transform scoring job over ``input_uri``."""
        ...

    def rollback(self, *, endpoint: EndpointRef, to_version: str | None = None) -> dict[str, Any]:
        """T-DEPLOY-07 — roll the endpoint back to a prior (or named) config version."""
        ...

    def delete(self, *, endpoint: EndpointRef) -> dict[str, Any]:
        """T-DEPLOY-08 — tear the endpoint down (irreversible)."""
        ...

    def recommend_config(self, *, model_ref: ModelRef, sample: str, sla: dict[str, Any] | None = None) -> list[SizingRec]:
        """T-DEPLOY-11 — recommend an inference instance/sizing config (Inference Recommender)."""
        ...

    def deploy_edge(self, *, model_ref: ModelRef, compile_output: dict[str, Any] | None = None,
                    target_device: str = "jetson_nano", greengrass_target: str | None = None,
                    **opts: Any) -> dict[str, Any]:
        """T-DEPLOY-10b (G3) — compile (Neo) → package (Edge Manager) → optionally deploy (Greengrass V2).

        Additive edge/embedded path that consumes a Neo ``compile`` output and packages/deploys it
        to an edge target. ``LocalServeBackend`` realizes it as an on-disk compiled bundle; the
        managed adapter submits the Neo compilation + edge-packaging jobs and (when a
        ``greengrass_target`` + client are configured) the Greengrass V2 deployment. Returns a
        provider-neutral dict (``compilation_job``/``edge_packaging_job``/``component_name``/
        ``deployed``/…) so the two adapters are interchangeable.
        """
        ...


@runtime_checkable
class ExplainabilityBackend(Protocol):
    """Bias + SHAP (default Clarify; local in-process SHAP). doc 05 §4.2 G3.

    M0 adapter: ``ClarifyBackend`` (wraps ``run_clarify``). doc 06 Part 3 row 12.
    """

    def bias_and_shap(self, *, dataset_uri: str, target: str,
                      sensitive_cols: list[str] | None = None, model_ref: ModelRef | None = None) -> ExplainReport: ...


@runtime_checkable
class EvaluationBackend(Protocol):
    """The deterministic Arbiter grade — the decision is **never delegated** (ADR-003).

    M0 adapter: ``DeterministicArbiter`` wraps ``arbiter/`` verbatim and is BOTH
    the default and the local adapter (it IS deterministic by design). Returns the
    rich ``arbiter.schema.ArbiterReport`` unchanged (wrap, don't reshape).
    """

    def grade(self, *, spec: Any, y_true: Any, y_pred: Any,
              baseline_score: float | None = None, **kwargs: Any) -> Any: ...


# ══════════════════════════════════════════════════════════════════════════════
# Group 4 — MLOps / governance / orchestration ports (6 + Guardrail)
# ══════════════════════════════════════════════════════════════════════════════
@runtime_checkable
class ModelRegistryBackend(Protocol):
    """Register / approve / list model packages (default SageMaker Registry).

    M0 adapters: ``SageMakerRegistryBackend`` (wraps ``tools/mlops/registry.py``)
    + ``JSONFileRegistryBackend`` (tests). doc 05 §4.2 G4.
    """

    def register(self, *, group: str, artifact: str, image: str | None = None,
                 metrics: dict[str, Any] | None = None, approval: str = "PendingManualApproval") -> PackageRef: ...

    def approve(self, *, package: str, status: str = "Approved", reason: str | None = None) -> str: ...

    def list(self, *, group: str, status: str | None = None) -> list[PackageRef]: ...


@runtime_checkable
class MonitoringBackend(Protocol):
    """Drift / monitor setup + inspect + the full Stage-11 observability surface.

    M0 adapters: ``SageMakerModelMonitorBackend`` (wraps ``tools/mlops/monitor.py``)
    + ``OTELTraceBackend``. M5 completes the surface: the parameterized ``analyze_drift``
    (T-MON-08, 6 drift types), CloudWatch-style ``put_custom_metric`` / ``configure_alert``
    / ``build_dashboard`` / ``get_recent_logs`` / ``track_spend`` / ``compile_report``
    (T-MON-03..10). The local adapter does GENUINE in-process drift detection; the managed
    adapter is thin boto3 orchestration (ADR-004). doc 05 §4.2 G4; ADR-008.
    """

    def setup(self, *, endpoint: str, baseline_uri: str, monitor_types: list[str] | None = None,
              schedule: str = "hourly") -> ScheduleRef: ...

    def inspect(self, *, schedule: str) -> MonitorReport: ...

    def metrics(self, *, namespace: str, metric: str, dims: dict[str, Any] | None = None,
                period: int = 300) -> dict[str, Any]: ...

    # ── M5 extension (T-MON-03..10; the parameterized 6-drift tool + observability) ──
    def analyze_drift(self, *, drift_type: str = "data", reference: Any = None, current: Any = None,
                      threshold: float | None = None, **kwargs: Any) -> DriftReport: ...

    def put_custom_metric(self, *, namespace: str, metric: str, value: float,
                          dims: dict[str, Any] | None = None) -> dict[str, Any]: ...

    def configure_alert(self, *, metric: str, threshold: float, comparison: str = "GreaterThanThreshold",
                        action: str = "", namespace: str = "Amlas/ML") -> dict[str, Any]: ...

    def build_dashboard(self, *, endpoint: str, panels: list[str] | None = None) -> dict[str, Any]: ...

    def get_recent_logs(self, *, log_group: str, filter: str = "", since: str = "1h") -> dict[str, Any]: ...

    def track_spend(self, *, endpoint: str, window: str = "7d") -> dict[str, Any]: ...

    def compile_report(self, *, endpoint: str, window: str = "7d") -> dict[str, Any]: ...


@runtime_checkable
class ExperimentTrackingBackend(Protocol):
    """Experiment run logging and search through the local leaderboard store."""

    def log_run(self, *, experiment: str, params: dict[str, Any], metrics: dict[str, Any],
                artifacts: dict[str, Any] | None = None) -> str: ...

    def search(self, *, experiment: str, filter: str | None = None, order_by: str | None = None) -> list[dict[str, Any]]: ...


@runtime_checkable
class PipelineBackend(Protocol):
    """Build / run / status a managed pipeline DAG (default SageMaker Pipelines).

    M0 adapters: ``SageMakerPipelineBackend`` (wraps ``tools/mlops/pipelines.py``)
    + ``LocalMLPipelineBackend`` (wraps ``pipeline/orchestrator.py`` MLPipeline).
    doc 05 §4.2 G4.
    """

    def build(self, *, spec: dict[str, Any]) -> PipelineRef: ...

    def run(self, *, pipeline: str, params: dict[str, Any] | None = None) -> ExecutionRef: ...

    def status(self, *, execution: str) -> dict[str, Any]: ...


@runtime_checkable
class OrchestrationBackend(Protocol):
    """The loop driver + async-resume seam (default LocalDriver). doc 05 §5.

    M1 fills the adapters (relocates ``orchestrator.py`` retreat loop); M0 defines
    the Protocol + stub only.
    """

    def run_flow(self, *, flow_id: str, spec: Any, params: dict[str, Any] | None = None,
                 steering: Any | None = None) -> dict[str, Any]: ...

    def register_callback(self, *, source: str, detail_type: str, resume_target: str) -> dict[str, Any]: ...

    def run_workflow(self, *, definition: dict[str, Any], input: dict[str, Any] | None = None) -> ExecutionRef: ...


@runtime_checkable
class GovernanceBackend(Protocol):
    """Model cards + lineage + access checks (default SageMaker Gov + LakeFormation).

    M4/M6 fill the managed adapters; M0 defines the Protocol + stub only (the
    local-markdown card already exists in ``decisions/model_card.py``).
    """

    def model_card(self, *, model_ref: ModelRef, content: dict[str, Any]) -> dict[str, Any]: ...

    def lineage(self, *, ref: str) -> dict[str, Any]: ...

    def check_access(self, *, principal: str, resource: str) -> dict[str, Any]: ...


@runtime_checkable
class GuardrailBackend(Protocol):
    """Content-safety check (default Bedrock Guardrails; in-process filter for isolated validation).

    Distinct from the hooks-as-guardrails *mechanism* (``hooks.py``): this port is
    a *content* guardrail (text policy), counted within group 4 (doc 06 Part 3).
    M4 fills the adapters; M0 defines the Protocol + stub only.
    """

    def check(self, *, text: str, policy: dict[str, Any] | None = None) -> dict[str, Any]: ...


# ══════════════════════════════════════════════════════════════════════════════
# Group 5 — agent-runtime ports (4, mostly pre-existing)
# ══════════════════════════════════════════════════════════════════════════════
# These four seams ALREADY EXIST in the codebase under other names (doc 05 §4.2
# Group 5): the Strands provider seam (``_build.py`` ModelConfig.for_role), the
# AgentCore Memory hooks (``memory/hooks.py``), the Code Interpreter
# (``tools/code_interpreter.py``), and the AgentCore Runtime dispatch
# (``cloud/dispatch.py``). M0 records them as Protocols for completeness /
# traceability; their concrete wrapping graduates in M1/M5 (they are NOT among the
# 8 ports M0 wraps with new adapters — they already work today).
@runtime_checkable
class ModelBackend(Protocol):
    """The Strands model-provider seam (BedrockModel/OpenAIModel/custom).

    Already provided by Strands' own provider abstraction (``_build.py``
    ``ModelConfig.for_role``); there is no ``@tool`` surface — it is the LLM
    provider seam, recorded here for the 25-port set. doc 05 §4.2 G5.
    """

    def stream(self, *, messages: Any, tool_specs: Any | None = None,
               system_prompt: str | None = None) -> Iterable[Any]: ...


@runtime_checkable
class MemoryBackend(Protocol):
    """AgentCore 3-strategy memory carry-over (EXISTING ``memory/hooks.py``).

    M5 lands the concrete ``LocalMemoryBackend`` carry-over fallback (the offline feed
    into C13 improvement, doc 05 §8.6): ``persist`` writes a learning record (production
    failure analyses + drift findings) keyed by ``actor_id=project_id``; ``carry_over``
    reads them into the next iteration's Research prompt. The managed AgentCore adapter
    (``MemoryHookProvider``) layers on top in prod. doc 05 §4.2 G5; ADR-008.
    """

    def carry_over(self, *, actor_id: str, session_id: str) -> list[Any]: ...

    def persist(self, *, actor_id: str, learnings: dict[str, Any], session_id: str = "") -> str: ...


@runtime_checkable
class SandboxBackend(Protocol):
    """Code Interpreter sandbox (EXISTING ``tools/code_interpreter.py``).

    State preserved across calls. doc 05 §4.2 G5.
    """

    def execute(self, *, code: str, files: dict[str, Any] | None = None) -> dict[str, Any]: ...


@runtime_checkable
class RuntimeBackend(Protocol):
    """AgentCore Runtime invoke (EXISTING ``cloud/dispatch.py`` InvokeAgentRuntime).

    In-process for local/tests; AgentCore for hosted. doc 05 §4.2 G5.
    """

    def invoke(self, *, payload: dict[str, Any], session_id: str) -> Iterable[Any]: ...


# ══════════════════════════════════════════════════════════════════════════════
# Introspection — the canonical port set + the env-key derivation
# ══════════════════════════════════════════════════════════════════════════════
# The 25 formal ports (doc 05 §4.6): 4 ML-compute + 7 data/feature + 3
# deploy/serve/eval + 7 MLOps/gov/orchestration (GuardrailBackend counted within
# group 4) + 4 agent-runtime. The registry resolves any of these.
ALL_PORTS: tuple[type, ...] = (
    # Group 1
    TrainingBackend,
    TuningBackend,
    AutoMLBackend,
    ModelHubBackend,
    # Group 2
    ETLBackend,
    DataQualityBackend,
    FeatureStoreBackend,
    LakeBackend,
    QueryBackend,
    LabelingBackend,
    RetrievalBackend,
    # Group 3
    DeploymentBackend,
    ExplainabilityBackend,
    EvaluationBackend,
    # Group 4 (+ Guardrail)
    ModelRegistryBackend,
    MonitoringBackend,
    ExperimentTrackingBackend,
    PipelineBackend,
    OrchestrationBackend,
    GovernanceBackend,
    GuardrailBackend,
    # Group 5 — agent-runtime
    ModelBackend,
    MemoryBackend,
    SandboxBackend,
    RuntimeBackend,
)


# Explicit, stable env-selector keys per port (doc 05 §8.5). Derived names would
# be brittle (``DataQualityBackend`` → ``DATAQUALITY`` vs the operator-friendly
# ``DQ``), so the mapping is explicit and is the single source the registry +
# ``.env.*.example`` agree on.
_PORT_ENV_KEYS: dict[str, str] = {
    "TrainingBackend": "TRAINING_BACKEND",
    "TuningBackend": "TUNING_BACKEND",
    "AutoMLBackend": "AUTOML_BACKEND",
    "ModelHubBackend": "MODELHUB_BACKEND",
    "ETLBackend": "ETL_BACKEND",
    "DataQualityBackend": "DQ_BACKEND",
    "FeatureStoreBackend": "FEATURE_STORE_BACKEND",
    "LakeBackend": "LAKE_BACKEND",
    "QueryBackend": "QUERY_BACKEND",
    "LabelingBackend": "LABELING_BACKEND",
    "RetrievalBackend": "RETRIEVAL_BACKEND",
    "DeploymentBackend": "DEPLOYMENT_BACKEND",
    "ExplainabilityBackend": "EXPLAINABILITY_BACKEND",
    "EvaluationBackend": "EVALUATION_BACKEND",
    "ModelRegistryBackend": "MODEL_REGISTRY_BACKEND",
    "MonitoringBackend": "MONITORING_BACKEND",
    "ExperimentTrackingBackend": "EXPERIMENT_TRACKING_BACKEND",
    "PipelineBackend": "PIPELINE_BACKEND",
    "OrchestrationBackend": "ORCHESTRATION_BACKEND",
    "GovernanceBackend": "GOVERNANCE_BACKEND",
    "GuardrailBackend": "GUARDRAIL_BACKEND",
    "ModelBackend": "MODEL_BACKEND",
    "MemoryBackend": "MEMORY_BACKEND",
    "SandboxBackend": "SANDBOX_BACKEND",
    "RuntimeBackend": "RUNTIME_BACKEND",
    # Control-plane FACET (not one of the 25 formal ports above, so NOT in
    # ``ALL_PORTS``): the cost-governance CostAdvisor facet (doc 05 §8.3/§8.8 /
    # roadmap M8). It is registered + resolved through the same registry, so it
    # needs a stable selector key — but it is intentionally absent from
    # ``ALL_PORTS`` to keep the formal-port count at 25 (the M0 conformance
    # contract). The class itself lives in ``control_plane/cost_governance.py``;
    # keyed here by name only so ``ports.py`` stays import-cycle-free.
    "CostAdvisorBackend": "COST_ADVISOR_BACKEND",
}


def port_key(port: type) -> str:
    """Return the env-selector key for a port (e.g. ``TrainingBackend`` → ``TRAINING_BACKEND``).

    Raises ``KeyError`` for an unknown port so a typo fails fast rather than
    silently resolving to a default (doc 05 §4.3 fail-fast contract).
    """
    name = getattr(port, "__name__", str(port))
    if name not in _PORT_ENV_KEYS:
        raise KeyError(f"unknown port {name!r}; not one of the {len(_PORT_ENV_KEYS)} registered ports")
    return _PORT_ENV_KEYS[name]
