"""MLOps tool subpackage (Capability Group D — Phase 10).

Agentic-over-SageMaker layer (master plan §8): each SageMaker MLOps building block
is a ``@tool`` the SageMaker Ops sub-agent / MLOps Lead orchestrate:

  * :mod:`pipelines`     — ``build_sagemaker_pipeline`` (Processing→Training→
    QualityCheck→ClarifyCheck→RegisterModel→Condition DAG) + ``run_sagemaker_pipeline``.
  * :mod:`registry`      — ``register_model_sm`` / ``approve_model_sm`` /
    ``list_model_packages_sm`` (versioning + approval gate).
  * :mod:`monitor`       — ``setup_model_monitor`` / ``inspect_model_monitor``
    (baseline → schedule → drift/violation reports).
  * :mod:`clarify`       — ``run_clarify`` (pre/post bias metrics + SHAP).
  * :mod:`feature_store` — ``feature_store_ingest`` / ``feature_store_get``
    (feature group + PIT online retrieval).
  * :mod:`kpis`          — ``compute_mlops_kpis`` (the §8 D7 MLOps scorecard).

Conventions (match ``tools/imputation`` + ``tools/synthetic``):
  - Tools are ``@tool``-decorated, return a JSON-serializable dict, fail soft
    with ``{"success": False, "error": "..."}``.
  - boto3/sagemaker are imported lazily; every tool takes an injectable client
    (``sm_client`` / ``fs_runtime_client``) so tests drive them against REAL AWS
    in the selected region — Pipelines, Registry, Clarify, Feature
    Store, and Model Monitor all run real (heavy provisioning gated ``expensive``).
    Reuses the SM-01 image/instance selector from ``tools/sagemaker.py`` (no
    duplication).
"""

from __future__ import annotations

from .clarify import build_clarify_config, run_clarify
from .feature_store import (
    feature_store_get,
    feature_store_ingest,
    infer_feature_definitions,
)
from .kpis import compute_mlops_kpis, maturity_level, mlops_kpi_scorecard
from .monitor import (
    build_monitoring_schedule_config,
    inspect_model_monitor,
    setup_model_monitor,
)
from .pipelines import (
    PIPELINE_STEP_SEQUENCE,
    build_pipeline_definition,
    build_sagemaker_pipeline,
    run_sagemaker_pipeline,
)
from .registry import (
    approve_model_sm,
    list_model_packages_sm,
    register_model_sm,
)

__all__ = [
    "PIPELINE_STEP_SEQUENCE",
    "approve_model_sm",
    "build_clarify_config",
    "build_monitoring_schedule_config",
    "build_pipeline_definition",
    # pipelines
    "build_sagemaker_pipeline",
    # kpis
    "compute_mlops_kpis",
    "feature_store_get",
    # feature store
    "feature_store_ingest",
    "infer_feature_definitions",
    "inspect_model_monitor",
    "list_model_packages_sm",
    "maturity_level",
    "mlops_kpi_scorecard",
    # registry
    "register_model_sm",
    # clarify
    "run_clarify",
    "run_sagemaker_pipeline",
    # monitor
    "setup_model_monitor",
]
