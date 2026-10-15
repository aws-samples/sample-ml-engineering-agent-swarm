"""Strands @tool wrappers for experiment orchestration.

Tools:
  - create_experiment_plan: Given spec + algorithm recommendations → create plan
  - run_experiments: Execute a plan, return results summary
  - get_experiment_report: Generate comparison report for a spec
"""

from __future__ import annotations

import json
from typing import Any

from strands import tool

from ..errors import tool_error_boundary


@tool
@tool_error_boundary(source="tools.experiment_tools")
def create_experiment_plan(
    spec_json: str,
    recommendations_json: str,
    store_path: str,
) -> dict[str, Any]:
    """Create an experiment plan from a spec and research recommendations.

    Converts ranked algorithm recommendations into a plan with multiple
    experiment configs (one per algorithm × hyperparameter combination).

    Args:
        spec_json: JSON string of spec dict with spec_id, task_type, target, primary_metric, data_path.
        recommendations_json: JSON string of recommendations from recommend_approaches tool.
        store_path: Path to the evaluation store JSON file.
    """
    from amlas.evaluation.store import EvaluationStore
    from amlas.experiments.runner import ExperimentRunner

    try:
        spec = json.loads(spec_json)
    except (json.JSONDecodeError, TypeError) as e:
        return {"success": False, "error": f"Invalid spec_json: {e}"}

    try:
        recommendations = json.loads(recommendations_json)
    except (json.JSONDecodeError, TypeError) as e:
        return {"success": False, "error": f"Invalid recommendations_json: {e}"}

    store = EvaluationStore(store_path=store_path)
    runner = ExperimentRunner(store=store)

    plan = runner.create_plan_from_recommendations(spec, recommendations)

    return {
        "success": True,
        "plan_id": plan.plan_id,
        "spec_id": plan.spec_id,
        "n_experiments": len(plan.configs),
        "algorithms": list({c.algorithm for c in plan.configs}),
        "plan": plan.to_dict(),
    }


@tool
@tool_error_boundary(source="tools.experiment_tools")
def run_experiments(
    plan_json: str,
    store_path: str,
) -> dict[str, Any]:
    """Execute an experiment plan and return results summary.

    Runs all experiments in the plan sequentially. Fault-tolerant:
    one failed experiment does not stop others.

    Args:
        plan_json: JSON string of ExperimentPlan (from create_experiment_plan output).
        store_path: Path to the evaluation store JSON file.
    """
    from amlas.evaluation.store import EvaluationStore
    from amlas.experiments.config import ExperimentPlan
    from amlas.experiments.runner import ExperimentRunner

    try:
        plan_data = json.loads(plan_json)
    except (json.JSONDecodeError, TypeError) as e:
        return {"success": False, "error": f"Invalid plan_json: {e}"}

    try:
        plan = ExperimentPlan.from_dict(plan_data)
    except (KeyError, TypeError) as e:
        return {"success": False, "error": f"Invalid plan structure: {e}"}

    store = EvaluationStore(store_path=store_path)
    runner = ExperimentRunner(store=store)

    results = runner.run_plan(plan)

    successful = [r for r in results if r.status == "success"]
    failed = [r for r in results if r.status != "success"]

    # Summary of results
    result_summaries = []
    for r in results:
        result_summaries.append({
            "run_id": r.run_id,
            "algorithm": r.algorithm,
            "status": r.status,
            "primary_score": r.primary_score,
            "duration_seconds": r.duration_seconds,
        })

    return {
        "success": True,
        "total_experiments": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "results": result_summaries,
        "best_algorithm": successful[0].algorithm if successful else None,
        "best_score": max(r.primary_score for r in successful) if successful else 0.0,
    }


@tool
@tool_error_boundary(source="tools.experiment_tools")
def get_experiment_report(
    spec_id: str,
    store_path: str,
) -> dict[str, Any]:
    """Generate a markdown comparison report for all experiments in a spec.

    Produces a formatted report with leaderboard, best model, and
    per-experiment details.

    Args:
        spec_id: The spec ID to generate report for.
        store_path: Path to the evaluation store JSON file.
    """
    from amlas.evaluation.store import EvaluationStore
    from amlas.experiments.config import ExperimentPlan
    from amlas.experiments.runner import ExperimentRunner

    store = EvaluationStore(store_path=store_path)
    runner = ExperimentRunner(store=store)

    # Get results and create a synthetic plan for report generation
    results = store.get_by_spec(spec_id)

    if not results:
        return {
            "success": False,
            "error": f"No results found for spec_id: {spec_id}",
        }

    # Determine primary metric from first result
    primary_metric = results[0].primary_metric

    # Create a synthetic plan for the report
    from amlas.experiments.config import ExperimentConfig

    configs = [
        ExperimentConfig(
            experiment_id=r.run_id,
            spec_id=spec_id,
            algorithm=r.algorithm,
            hyperparameters=r.hyperparameters,
            data_path="",
            target_column="",
            task_type="classification",
            primary_metric=primary_metric,
        )
        for r in results
    ]

    plan = ExperimentPlan(
        plan_id="report-plan",
        spec_id=spec_id,
        configs=configs,
        created_at=results[0].timestamp if results else "",
    )

    report = runner.get_report(plan)

    return {
        "success": True,
        "spec_id": spec_id,
        "report": report,
        "n_experiments": len(results),
        "n_successful": sum(1 for r in results if r.status == "success"),
    }
