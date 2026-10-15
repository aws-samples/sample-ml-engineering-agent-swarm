"""Strands @tool wrappers for evaluation operations.

Tools:
  - record_experiment_result: Save a training result to the evaluation store
  - get_leaderboard: Retrieve current rankings for a spec
  - compare_experiments: Side-by-side metric comparison of specific runs
  - get_best_model: Return the current best model info for a spec
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from strands import tool

from ..errors import tool_error_boundary


@tool
@tool_error_boundary(source="tools.evaluation_tools")
def record_experiment_result(
    store_path: str,
    run_id: str,
    spec_id: str,
    algorithm: str,
    hyperparameters: str,
    metrics: str,
    primary_metric: str,
    primary_score: float,
    duration_seconds: float,
    artifact_path: str,
    status: str,
    notes: str = "",
) -> dict[str, Any]:
    """Record an experiment result to the evaluation store.

    Saves a training run's result (algorithm, metrics, hyperparameters) to the
    JSON-based evaluation store for leaderboard tracking and comparison.

    Args:
        store_path: Path to the evaluation store JSON file.
        run_id: Unique identifier for this run.
        spec_id: Which spec/project this run belongs to.
        algorithm: Algorithm name (e.g., 'random_forest', 'xgboost').
        hyperparameters: JSON string of hyperparameters used.
        metrics: JSON string of computed metrics (e.g., '{"f1_macro": 0.78}').
        primary_metric: Name of the primary metric for ranking.
        primary_score: Value of the primary metric.
        duration_seconds: Training duration in seconds.
        artifact_path: Path to saved model artifact.
        status: Run status ('success', 'failed', 'error').
        notes: Optional notes about the run.
    """
    from amlas.evaluation.store import EvaluationStore, ExperimentResult

    try:
        params = json.loads(hyperparameters) if isinstance(hyperparameters, str) else hyperparameters
        metrics_dict = json.loads(metrics) if isinstance(metrics, str) else metrics
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON: {e}"}

    result = ExperimentResult(
        run_id=run_id,
        spec_id=spec_id,
        algorithm=algorithm,
        hyperparameters=params,
        metrics=metrics_dict,
        primary_metric=primary_metric,
        primary_score=float(primary_score),
        duration_seconds=float(duration_seconds),
        timestamp=datetime.now(UTC).isoformat(),
        artifact_path=artifact_path,
        status=status,
        notes=notes,
    )

    store = EvaluationStore(store_path=store_path)
    store.record(result)

    return {
        "success": True,
        "run_id": run_id,
        "spec_id": spec_id,
        "algorithm": algorithm,
        "primary_score": primary_score,
        "message": f"Recorded experiment {run_id} ({algorithm}, {primary_metric}={primary_score:.4f})",
    }


@tool
@tool_error_boundary(source="tools.evaluation_tools")
def get_leaderboard(
    store_path: str,
    spec_id: str,
    metric: str = "f1_macro",
    top_n: int = 10,
    direction: str = "maximize",
) -> dict[str, Any]:
    """Retrieve the current leaderboard (ranked experiment results) for a spec.

    Returns experiments ranked by primary metric score, best first.

    Args:
        store_path: Path to the evaluation store JSON file.
        spec_id: Which spec/project to get leaderboard for.
        metric: Metric to rank by.
        top_n: Maximum number of results to return.
        direction: 'maximize' (higher is better) or 'minimize' (lower is better).
    """
    from amlas.evaluation.leaderboard import Leaderboard
    from amlas.evaluation.store import EvaluationStore

    store = EvaluationStore(store_path=store_path)
    lb = Leaderboard(store=store)

    ranked = lb.get_leaderboard(spec_id, metric=metric, direction=direction, top_n=top_n)

    leaderboard_data = []
    for i, r in enumerate(ranked, 1):
        leaderboard_data.append({
            "rank": i,
            "run_id": r.run_id,
            "algorithm": r.algorithm,
            "primary_score": r.primary_score,
            "metrics": r.metrics,
            "duration_seconds": r.duration_seconds,
            "timestamp": r.timestamp,
        })

    markdown = lb.format_leaderboard_markdown(spec_id, metric=metric, direction=direction, top_n=top_n)

    return {
        "success": True,
        "spec_id": spec_id,
        "metric": metric,
        "direction": direction,
        "total_results": len(ranked),
        "leaderboard": leaderboard_data,
        "markdown": markdown,
    }


@tool
@tool_error_boundary(source="tools.evaluation_tools")
def compare_experiments(
    store_path: str,
    run_ids: str,
) -> dict[str, Any]:
    """Compare specific experiment runs side-by-side.

    Retrieves results for the given run IDs and presents them for comparison.

    Args:
        store_path: Path to the evaluation store JSON file.
        run_ids: JSON string list of run IDs to compare (e.g., '["run-001", "run-002"]').
    """
    from amlas.evaluation.store import EvaluationStore

    try:
        ids = json.loads(run_ids) if isinstance(run_ids, str) else run_ids
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"Invalid JSON for run_ids: {e}"}

    store = EvaluationStore(store_path=store_path)
    results = store.get_by_run_ids(ids)

    if not results:
        return {"success": False, "error": f"No results found for run_ids: {ids}"}

    comparison = []
    for r in results:
        comparison.append({
            "run_id": r.run_id,
            "algorithm": r.algorithm,
            "primary_metric": r.primary_metric,
            "primary_score": r.primary_score,
            "metrics": r.metrics,
            "hyperparameters": r.hyperparameters,
            "duration_seconds": r.duration_seconds,
            "status": r.status,
            "timestamp": r.timestamp,
        })

    return {
        "success": True,
        "comparison": comparison,
        "n_experiments": len(comparison),
    }


@tool
@tool_error_boundary(source="tools.evaluation_tools")
def get_best_model(
    store_path: str,
    spec_id: str,
    direction: str = "maximize",
) -> dict[str, Any]:
    """Get the current best model for a spec.

    Returns the highest-scoring (or lowest for minimize) experiment run.

    Args:
        store_path: Path to the evaluation store JSON file.
        spec_id: Which spec/project to find best model for.
        direction: 'maximize' (higher is better) or 'minimize' (lower is better).
    """
    from amlas.evaluation.store import EvaluationStore

    store = EvaluationStore(store_path=store_path)
    best = store.get_best_run(spec_id, direction=direction)

    if best is None:
        return {
            "success": False,
            "error": f"No successful results found for spec_id: {spec_id}",
        }

    return {
        "success": True,
        "best_run_id": best.run_id,
        "best_algorithm": best.algorithm,
        "best_score": best.primary_score,
        "primary_metric": best.primary_metric,
        "metrics": best.metrics,
        "hyperparameters": best.hyperparameters,
        "artifact_path": best.artifact_path,
        "duration_seconds": best.duration_seconds,
        "timestamp": best.timestamp,
    }
