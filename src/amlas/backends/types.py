"""Domain value types crossing the backend-port seam (doc 05 §4.2 / roadmap M0 work-item 1).

These are the **technology-agnostic** value objects the 25 ports (``ports.py``)
speak. They carry the *domain* (algo, data, budget, plan, handle, ref) and never
leak boto3 types (no ``AlgorithmSpecification``, no ``TrainingJobArn`` shapes) —
that translation lives in each adapter (doc 05 §4.1 "Ports speak the domain").

Design rules (ADR-001 / ADR-012):
  * Pure-Python dataclasses + enums; **no boto3 / strands imports** here so the
    types are directly verifiable without AWS creds (same constraint as
    ``config/runtime.py``).
  * ``frozen=True`` where the value is an immutable handle/ref; mutable result
    bags stay plain dataclasses.
  * ``backend`` / ``adapter`` provenance fields let the traceability matrix
    (doc 05 §8.1) record *which* adapter produced a handle/ref — the
    ``port → adapter (+ reason)`` span the registry stamps on resolution.

M0 scope: only the value types the 8 wrapped ports (Training, Evaluation,
ModelRegistry, Lake, Pipeline, Monitoring, Explainability, FeatureStore) plus the
registry/hooks need. Later milestones extend this module (e.g. ``EndpointRef`` is
defined now but only *consumed* by the M4 DeploymentBackend).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


# ──────────────────────────────────────────────────────────────────────────────
# Status / result enums
# ──────────────────────────────────────────────────────────────────────────────
class JobState(StrEnum):
    """Backend-neutral lifecycle state of an async job (training/processing/etc.).

    Maps onto each provider's native states in the adapter (e.g. SageMaker
    ``InProgress``/``Completed``/``Failed``/``Stopped`` → these), so flows never
    branch on a vendor's enum strings.
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"
    UNKNOWN = "UNKNOWN"


class Tier(StrEnum):
    """Medallion lake tiers (doc 05 §6.3). ``BRONZE`` is read-only (ADR-009)."""

    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


# ──────────────────────────────────────────────────────────────────────────────
# Handles & references (immutable provenance-carrying value objects)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class JobHandle:
    """An opaque handle to a submitted async job (training/tuning/processing).

    ``id`` is the provider job name/id; ``backend`` records which adapter created
    it (``"sagemaker"`` / ``"local"`` …) so ``status``/``metrics`` route back to
    the same adapter and the trace span can attribute the run (doc 05 §8.1).
    ``extra`` carries adapter-specific provenance (e.g. ``runtime_session_id``,
    ``training_job_arn``) without widening the typed surface.
    """

    id: str
    backend: str = "local"
    arn: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JobStatus:
    """The resolved state of a job plus an optional human-readable detail."""

    state: JobState
    detail: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelRef:
    """A reference to a trained model artifact (the ``ModelDataUrl`` analog)."""

    uri: str
    backend: str = "local"
    algo: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetRef:
    """A reference to a dataset slice/artifact crossing the seam.

    ``uri`` is the canonical location (``s3://…`` or a local path); ``tier``
    records the medallion tier when known; ``rows``/``schema`` carry the cheap
    profile a flow needs without re-reading the data.
    """

    uri: str
    tier: str = ""
    rows: int = 0
    schema: dict[str, str] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PackageRef:
    """A reference to a registered model package (Model Registry version)."""

    arn: str
    group: str
    version: int = 1
    approval_status: str = "PendingManualApproval"
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScheduleRef:
    """A reference to a monitoring schedule (Model Monitor)."""

    name: str
    arn: str = ""
    monitor_type: str = "data_quality"
    schedule: str = ""


@dataclass(frozen=True)
class EndpointRef:
    """A reference to a deployed inference endpoint (consumed by M4 Deployment)."""

    name: str
    arn: str = ""
    mode: str = "realtime"
    backend: str = "local"


@dataclass(frozen=True)
class PipelineRef:
    """A reference to a built pipeline definition (Pipeline port)."""

    name: str
    arn: str = ""
    step_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionRef:
    """A reference to a pipeline/workflow execution."""

    arn: str
    status: str = ""
    pipeline_name: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Result bags (richer reports the ports return)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class ProfileReport:
    """A data-quality / profiling report (DataQualityBackend.profile)."""

    rows: int = 0
    columns: int = 0
    score: float = 0.0
    dimensions: dict[str, float] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class DQScore:
    """A data-quality validation score (DataQualityBackend.evaluate)."""

    passed: bool = True
    score: float = 0.0
    violations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ExplainReport:
    """A bias + SHAP explainability report (ExplainabilityBackend)."""

    bias_report_uri: str = ""
    shap_uri: str = ""
    processing_job: str = ""
    analysis: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitorReport:
    """A monitoring inspection report (MonitoringBackend.inspect)."""

    schedule_name: str = ""
    status: str = ""
    last_execution_status: str = ""
    violations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DriftReport:
    """A drift-detection result crossing the seam (MonitoringBackend.analyze_drift; M5).

    The backend-neutral shape doc 06 T-MON-08 returns
    (``{drift_detected, magnitude, affected}``) extended with the ``drift_type`` and the
    ``threshold`` it was judged against. ``drift_type`` is one of the 6 the parameterized
    tool serves (``label`` | ``data`` | ``feature`` | ``model`` | ``bias`` | ``attribution``);
    the local adapter computes ``magnitude`` genuinely (PSI / metric-delta / parity-delta /
    importance-L1), the managed adapter maps Model-Monitor / Clarify violations onto it.
    ``severity`` buckets the magnitude (``none`` | ``low`` | ``moderate`` | ``severe``) so the
    retrain trigger (T-ORCH-02) can decide without re-deriving thresholds.
    """

    drift_type: str = "data"
    drift_detected: bool = False
    magnitude: float = 0.0
    threshold: float = 0.2
    severity: str = "none"
    affected: list[str] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrainDecision:
    """The closed-loop retrain trigger decision (T-ORCH-02 ``evaluate_retrain_trigger``; M5/ADR-008).

    ``trigger`` is the go/no-go on re-entering the unified exploration retreat loop;
    ``target_stage`` names where the improvement iteration re-enters (default
    ``"research"`` — the retreat edge feeds the next Research prompt, analogous to how
    ``abandoned_reasons`` feed it today); ``reasons`` is the human-readable rationale; and
    ``carry_context`` is the drift/failure findings carried into that next iteration
    (the AgentCore Memory carry-over payload).
    """

    trigger: bool = False
    reason: str = ""
    target_stage: str = "research"
    severity: str = "none"
    signals: dict[str, Any] = field(default_factory=dict)
    carry_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class CatalogView:
    """A view over the data catalog (LakeBackend.list_catalog)."""

    datasets: list[dict[str, Any]] = field(default_factory=list)
    count: int = 0


@dataclass
class SizingRec:
    """A single inference-config sizing recommendation (Deployment/CostAdvisor)."""

    instance_type: str
    est_cost_per_hour: float = 0.0
    est_latency_ms: float = 0.0
    est_throughput: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# M6 value types — RetrievalBackend (ADR-017) + ModelHub / Query / Labeling
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Chunk:
    """A retrieval chunk — a unit of indexed text + its source provenance (T-RAG-02).

    ``text`` is the chunk body; ``source`` is the document path/uri it came from;
    ``chunk_id`` is stable within an index; ``metadata`` carries filterable tags
    (e.g. ``{"stage": "0", "kind": "arxiv"}``) the metadata-filter tool (T-RAG-04)
    narrows on. Backend-neutral: a local TF-IDF chunk and a Bedrock-KB chunk share
    this shape so the RAG tools never branch on the adapter.
    """

    chunk_id: str
    text: str
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IndexRef:
    """A reference to a built retrieval index (RetrievalBackend.build_index; T-RAG-05).

    ``name`` is the index id the query side resolves; ``backend`` records the adapter
    (``"local"`` / ``"bedrock_kb"`` / ``"opensearch"``); ``doc_count``/``chunk_count``
    are the cheap stats a digest needs without re-reading the corpus; ``arn`` carries
    the managed resource id (KB id / collection arn) when applicable.
    """

    name: str
    backend: str = "local"
    arn: str = ""
    doc_count: int = 0
    chunk_count: int = 0
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """A ranked retrieval hit (RetrievalBackend.query / retrieve; T-RAG-06).

    ``chunk`` is the matched :class:`Chunk`; ``score`` is the relevance score (cosine /
    BM25-ish / KB relevance), higher = better; ``rank`` is the 0-based position in the
    returned list. The tool layer flattens these to plain dicts for the agent.
    """

    chunk: Chunk
    score: float = 0.0
    rank: int = 0


@dataclass
class ModelCard:
    """A foundation/pretrained model-hub card (ModelHubBackend.model_card; T-HUB row).

    The backend-neutral model descriptor JumpStart / Bedrock FM / a local static
    catalog all map onto: ``model_id`` (the hub id), ``task`` (e.g. ``"tabular-classification"``),
    ``provider`` (``"jumpstart"`` / ``"bedrock"`` / ``"huggingface"``), plus cheap
    fit signals (``modalities``, ``fine_tunable``, ``description``, ``license``).
    """

    model_id: str
    task: str = ""
    provider: str = ""
    description: str = ""
    fine_tunable: bool = False
    modalities: list[str] = field(default_factory=list)
    license: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryResult:
    """A NL→SQL query result crossing the seam (QueryBackend.run_sql; T-QRY-01).

    ``columns`` + ``rows`` are the tabular result (rows as lists aligned to columns);
    ``row_count`` is the cheap count; ``query`` echoes the executed SQL; ``engine``
    records the adapter (``"athena"`` / ``"duckdb"``). ``execution_id`` carries the
    Athena QueryExecutionId when managed.
    """

    columns: list[str] = field(default_factory=list)
    rows: list[list[Any]] = field(default_factory=list)
    row_count: int = 0
    query: str = ""
    engine: str = "duckdb"
    execution_id: str = ""


@dataclass(frozen=True)
class LabelingJobRef:
    """A reference to a labeling job (LabelingBackend.create_job; T-LBL-01).

    ``name`` is the Ground-Truth labeling-job name; ``arn`` the managed job arn;
    ``task_type`` the annotation task (``"text-classification"`` etc.); ``status`` the
    last-known job status; ``backend`` the adapter (``"groundtruth"`` / ``"stub"``).
    """

    name: str
    arn: str = ""
    task_type: str = ""
    status: str = "InProgress"
    backend: str = "stub"
    detail: dict[str, Any] = field(default_factory=dict)


# ``Verdict`` is the generic accept/reject value some hooks/gates emit. The
# Arbiter's own rich report stays ``arbiter.schema.ArbiterReport`` (the
# EvaluationBackend returns that verbatim — wrap, don't reshape; ADR-003).
@dataclass(frozen=True)
class Verdict:
    """A deterministic allow/deny decision with a reason (hooks/gates)."""

    allowed: bool
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)
