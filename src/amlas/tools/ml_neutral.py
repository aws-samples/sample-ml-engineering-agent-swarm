"""Backend-neutral Stage 6–8 tools — feature / tuning / AutoML / tracking / eval (M3).

The doc 05 §4.4 EVOLVE pattern extended to the M3 ports (doc 06 T-FS-*, T-TUNE-*,
T-AML-*, T-EXP-*, T-EVAL-*): each tool resolves a PORT via the ``BackendRegistry`` and
delegates to whatever adapter the registry binds (local offline/under test, managed AWS
in prod). NEW M3 tools, ADDITIVE — the existing tools/agents keep working untouched, so
the suite stays green; flows + the control plane adopt these port-calling tools.

Stage 6 (FeatureStore): ``feature_store_ingest_via_port`` (T-FS-01),
``feature_store_get_via_port`` (T-FS-02), ``feature_pit_dataset`` (T-FS-03).
Stage 8 (Tuning/AutoML/Tracking/Eval): ``run_hpo`` (T-TUNE-01), ``run_automl`` (T-AML-01),
``log_experiment_run`` (T-EXP-01), ``search_runs`` (T-EXP-02),
``build_model_comparison_table`` (T-EXP-05/07), ``evaluate_offline_metrics`` (T-EVAL-04
threshold/offline metrics — distinct from the deterministic Arbiter grade, T-EVAL-01,
which lives in ``backend_neutral.grade_submission`` and is NEVER delegated, ADR-003).

Spend-incurring tools (``run_hpo`` / ``run_automl``) classify as EXPENSIVE_JOB in the
approval policy (``control_plane/approvals.py``); the flows wrap them with
``plan_approve_execute`` so a managed AMT/Autopilot run is plan-previewed + approved
before it spends (doc 05 §8.8; ADR-014).
"""

from __future__ import annotations

from typing import Any

from strands import tool

from ..backends.ports import (
    AutoMLBackend,
    EvaluationBackend,
    ExperimentTrackingBackend,
    FeatureStoreBackend,
    TuningBackend,
)
from ..backends.registry import REGISTRY
from ..errors import tool_error_boundary

__all__ = [
    "build_model_comparison_table",
    "evaluate_offline_metrics",
    "feature_pit_dataset",
    "feature_store_get_via_port",
    "feature_store_ingest_via_port",
    "log_experiment_run",
    "run_automl",
    "run_hpo",
    "search_runs",
]


# ══════════════════════════════════════════════════════════════════════════════
# Stage 6 — FeatureStore (T-FS-01/02/03)
# ══════════════════════════════════════════════════════════════════════════════
@tool
@tool_error_boundary(source="tools.ml_neutral")
def feature_store_ingest_via_port(
    group: str,
    records: Any,
    scale: int | str | None = None,
) -> dict[str, Any]:
    """Ingest feature ``records`` into ``group`` via :class:`FeatureStoreBackend` (T-FS-01).

    Resolves the feature-store port (SageMaker FS managed by default, gold-table local
    offline) and upserts. Returns the ingested count + the bound adapter.
    """
    try:
        resolved = REGISTRY.resolve(FeatureStoreBackend, scale=scale)
        n = resolved.adapter.upsert(group=group, records=records)
        return {"ingested": n, "group": group, "adapter": resolved.adapter_name, "reason": resolved.reason}
    except Exception as e:
        return {"success": False, "error": f"feature_store_ingest_via_port failed: {type(e).__name__}"}


@tool
@tool_error_boundary(source="tools.ml_neutral")
def feature_store_get_via_port(group: str, record_id: str, online: bool = True) -> dict[str, Any]:
    """Retrieve the latest feature record for ``record_id`` (T-FS-02).

    Resolves :class:`FeatureStoreBackend` and reads the (online) record — the latest
    known value for the entity.
    """
    try:
        resolved = REGISTRY.resolve(FeatureStoreBackend)
        rec = resolved.adapter.get(group=group, record_id=record_id, online=online)
        return {"record": rec, "group": group, "adapter": resolved.adapter_name}
    except Exception as e:
        return {"success": False, "error": f"feature_store_get_via_port failed: {type(e).__name__}"}


@tool
@tool_error_boundary(source="tools.ml_neutral")
def feature_pit_dataset(
    group: str,
    entity_ids: list[str],
    as_of: str,
    scale: int | str | None = None,
) -> dict[str, Any]:
    """Assemble a **point-in-time** training dataset at ``as_of`` (T-FS-03, no leakage).

    Resolves :class:`FeatureStoreBackend` (gold-table in-process PIT offline; SageMaker
    offline-store + Athena ``as_of`` join managed) and returns a PIT-correct dataset:
    per entity, only the latest feature value known at or before ``as_of`` — strictly no
    future leakage (the project PIT rule). Returns the dataset URI + row count + lineage.
    """
    try:
        resolved = REGISTRY.resolve(FeatureStoreBackend, scale=scale)
        ref = resolved.adapter.pit_dataset(group=group, entity_ids=entity_ids, as_of=as_of)
        return {
            "dataset_uri": ref.uri,
            "rows": ref.rows,
            "tier": ref.tier,
            "lineage": ref.lineage,
            "adapter": resolved.adapter_name,
            "reason": resolved.reason,
        }
    except Exception as e:
        return {"success": False, "error": f"feature_pit_dataset failed: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
# Stage 8 — Tuning (T-TUNE-01) — spend-incurring (gated by plan_approve_execute)
# ══════════════════════════════════════════════════════════════════════════════
@tool
@tool_error_boundary(source="tools.ml_neutral")
def run_hpo(
    base_training: dict[str, Any],
    ranges: dict[str, Any],
    objective: str,
    strategy: str = "bayesian",
    max_jobs: int = 10,
    max_parallel: int = 1,
    warm_start_from: str | None = None,
    scale: int | str | None = None,
) -> dict[str, Any]:
    """Run hyperparameter search over :class:`TuningBackend` (T-TUNE-01).

    Resolves the tuning port (Optuna local offline, SageMaker AMT managed for scale) and
    runs the search, returning the best params + score + trial log. **Spend-incurring** —
    the ``hpo_flow`` wraps this with ``plan_approve_execute`` so a managed AMT run is
    plan-previewed + approved first (doc 05 §8.8 EXPENSIVE_JOB).
    """
    try:
        resolved = REGISTRY.resolve(TuningBackend, scale=scale)
        out: dict[str, Any] = resolved.adapter.search(
            base_training=base_training, ranges=ranges, objective=objective,
            strategy=strategy, max_jobs=max_jobs, max_parallel=max_parallel,
            warm_start_from=warm_start_from,
        )
        out.setdefault("adapter", resolved.adapter_name)
        out.setdefault("reason", resolved.reason)
        return out
    except Exception as e:
        return {"success": False, "error": f"run_hpo failed: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
# Stage 8 — AutoML (T-AML-01) — spend-incurring (gated)
# ══════════════════════════════════════════════════════════════════════════════
@tool
@tool_error_boundary(source="tools.ml_neutral")
def run_automl(
    train_uri: str,
    target: str,
    problem_type: str,
    mode: str = "ensembling",
    max_candidates: int = 10,
    generate_only: bool = False,
    scale: int | str | None = None,
) -> dict[str, Any]:
    """Run AutoML candidate generation over :class:`AutoMLBackend` (T-AML-01).

    Resolves the AutoML port (local quick-sweep offline, SageMaker Autopilot managed) and
    runs candidate generation / ensembling. ``generate_only`` produces a cheap candidate
    PLAN without training (US-3.06). **Spend-incurring** when not generate-only — the flow
    wraps it with ``plan_approve_execute``.
    """
    try:
        resolved = REGISTRY.resolve(AutoMLBackend, scale=scale)
        out: dict[str, Any] = resolved.adapter.run(
            train_uri=train_uri, target=target, problem_type=problem_type,
            mode=mode, max_candidates=max_candidates, generate_only=generate_only,
        )
        out.setdefault("adapter", resolved.adapter_name)
        out.setdefault("reason", resolved.reason)
        return out
    except Exception as e:
        return {"success": False, "error": f"run_automl failed: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
# Stage 8 — ExperimentTracking (T-EXP-01/02/05/07)
# ══════════════════════════════════════════════════════════════════════════════
@tool
@tool_error_boundary(source="tools.ml_neutral")
def log_experiment_run(
    experiment: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Log an experiment run via :class:`ExperimentTrackingBackend` (T-EXP-01).

    Resolves the local leaderboard tracking port and records
    params/metrics/artifacts. Returns the run id.
    """
    try:
        resolved = REGISTRY.resolve(ExperimentTrackingBackend)
        run_id = resolved.adapter.log_run(experiment=experiment, params=params, metrics=metrics, artifacts=artifacts)
        return {"run_id": run_id, "experiment": experiment, "adapter": resolved.adapter_name}
    except Exception as e:
        return {"success": False, "error": f"log_experiment_run failed: {e}"}


@tool
@tool_error_boundary(source="tools.ml_neutral")
def search_runs(experiment: str, filter: str | None = None, order_by: str | None = None) -> dict[str, Any]:
    """Search logged runs for ``experiment`` via :class:`ExperimentTrackingBackend` (T-EXP-02)."""
    try:
        resolved = REGISTRY.resolve(ExperimentTrackingBackend)
        runs = resolved.adapter.search(experiment=experiment, filter=filter, order_by=order_by)
        return {"runs": runs, "count": len(runs), "experiment": experiment, "adapter": resolved.adapter_name}
    except Exception as e:
        return {"success": False, "error": f"search_runs failed: {e}"}


@tool
@tool_error_boundary(source="tools.ml_neutral")
def build_model_comparison_table(experiment: str, direction: str = "maximize") -> dict[str, Any]:
    """Build a ranked model-comparison table for ``experiment`` (T-EXP-05/07).

    Resolves :class:`ExperimentTrackingBackend` and delegates to its
    ``build_comparison_table`` (the EXISTING leaderboard ranking IP locally). The
    ``evaluation_flow`` renders this alongside the deterministic Arbiter verdict.
    """
    try:
        resolved = REGISTRY.resolve(ExperimentTrackingBackend)
        adapter = resolved.adapter
        if not hasattr(adapter, "build_comparison_table"):
            return {"success": False, "error": f"adapter {resolved.adapter_name} has no comparison table"}
        table = adapter.build_comparison_table(experiment=experiment, direction=direction)
        return {"comparison_table": table, "rows": len(table), "experiment": experiment, "adapter": resolved.adapter_name}
    except Exception as e:
        return {"success": False, "error": f"build_model_comparison_table failed: {e}"}


# ══════════════════════════════════════════════════════════════════════════════
# Stage 8 — Evaluation offline metrics (T-EVAL-04) — NOT the Arbiter decision (ADR-003)
# ══════════════════════════════════════════════════════════════════════════════
@tool
@tool_error_boundary(source="tools.ml_neutral")
def evaluate_offline_metrics(
    spec: Any,
    y_true: Any,
    y_pred: Any,
    baseline_score: float | None = None,
) -> dict[str, Any]:
    """Compute offline evaluation metrics via the deterministic Arbiter grade (T-EVAL-04).

    This is a thin convenience over :class:`EvaluationBackend` that surfaces the metric
    block from the Arbiter report for a model-comparison view. The accept/reject DECISION
    remains the non-delegated deterministic Arbiter (ADR-003) — this tool only extracts
    the computed metrics; it does NOT make or override the verdict.
    """
    try:
        backend = REGISTRY.resolve(EvaluationBackend).adapter
        report = backend.grade(spec=spec, y_true=y_true, y_pred=y_pred, baseline_score=baseline_score)
        rd = report.model_dump() if hasattr(report, "model_dump") else dict(report)
        return {
            "metrics": rd.get("metrics", rd.get("scores", {})),
            "primary_metric": rd.get("primary_metric", rd.get("metric")),
            "primary_score": rd.get("primary_score", rd.get("score")),
            "verdict": rd.get("verdict", rd.get("decision")),
            "arbiter_report": rd,
        }
    except Exception as e:
        return {"success": False, "error": f"evaluate_offline_metrics failed: {e}"}
