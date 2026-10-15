"""Strands @tool wrappers for research discovery.

Tools:
  - search_algorithms: Search algorithm KB by task/data type, return ranked list
  - recommend_approaches: Given a SpecModel, synthesize recommendations
"""

from __future__ import annotations

import json
from typing import Any

from strands import tool

from ..errors import tool_error_boundary
from ..knowledge.algorithms import rank_algorithms_for_task
from .spec_heuristics import GENERATION_TASKS, resolve_task_modality


@tool
@tool_error_boundary(source="tools.research_tools")
def search_algorithms(
    task_type: str,
    data_type: str = "tabular",
    data_size: str = "medium",
) -> dict[str, Any]:
    """Search the algorithm knowledge base for suitable algorithms.

    Returns a ranked list of algorithms with rationale for the given task
    configuration. Algorithms are scored and ranked by suitability.

    Args:
        task_type: Problem type — 'classification' or 'regression'.
        data_type: Data modality — 'tabular', 'text', 'image', 'time-series'.
        data_size: Dataset size — 'small' (<500), 'medium' (500-10K), 'large' (>10K).
    """
    ranked = rank_algorithms_for_task(
        task_type=task_type,
        data_type=data_type,
        data_size=data_size,
    )

    # Format output (include task_types for downstream filtering)
    algorithms = []
    for algo in ranked:
        algorithms.append({
            "name": algo["name"],
            "display_name": algo["display_name"],
            "rank": algo["rank"],
            "category": algo["category"],
            "task_types": algo["task_types"],
            "strengths": algo["strengths"][:3],  # Top 3 strengths
            "weaknesses": algo["weaknesses"][:2],  # Top 2 weaknesses
            "complexity": algo["complexity"],
            "sklearn_class": algo.get("sklearn_class", ""),
            "rationale": algo.get("rationale", ""),
            "handles_missing": algo["data_requirements"].get("handles_missing", False),
        })

    return {
        "algorithms": algorithms,
        "task_type": task_type,
        "data_type": data_type,
        "data_size": data_size,
        "total_found": len(algorithms),
    }


@tool
@tool_error_boundary(source="tools.research_tools")
def recommend_approaches(spec_json: str) -> dict[str, Any]:
    """Synthesize research + algorithm knowledge into ranked recommendations.

    Given a SpecModel (as JSON), uses the algorithm knowledge base to produce a
    comprehensive recommendation for the ML pipeline.

    Args:
        spec_json: JSON string of a SpecModel dict with task_type, target, etc.
    """
    try:
        spec = json.loads(spec_json)
    except (json.JSONDecodeError, TypeError) as e:
        return {"error": f"Invalid spec JSON: {e}"}

    task_type = spec.get("task_type", "classification")
    normalized_task = str(task_type).strip().lower().replace("-", "_")
    if normalized_task in GENERATION_TASKS:
        task_type = "text_generation"
    domain_notes = spec.get("domain_notes", "")

    # Determine data characteristics from spec
    data_type = resolve_task_modality(task_type, spec.get("modality"))
    data_size = "medium"  # Default heuristic

    if task_type == "text_generation":
        return {
            "data_type": data_type,
            "recommended_algorithms": [],
            "suggested_approach": (
                "Research a pinned open language model and compatible SFT recipe. "
                "Verify dataset provenance, model templates and measured resource fit "
                "before managed training. The tabular algorithm catalog does not supply "
                "a generation recipe."
            ),
            "experiment_suggestions": [
                "Prepare supervised text/tool trajectories with grouped splits and leakage checks",
                "Measure resource fit and throughput for the selected model and trainer",
                "Evaluate the untouched base and trained model with matched held-out settings",
            ],
            "needs_research": ["model_revision", "trainer_recipe", "evaluation_recipe"],
        }

    # Get ranked algorithms
    ranked = rank_algorithms_for_task(
        task_type=task_type,
        data_type=data_type,
        data_size=data_size,
    )

    # Build recommended algorithms list
    recommended_algorithms = []
    for algo in ranked[:5]:  # Top 5 recommendations
        sklearn_class = algo.get("sklearn_class", "")
        if task_type == "regression" and "sklearn_class_regression" in algo:
            sklearn_class = algo["sklearn_class_regression"]
        recommended_algorithms.append({
            "name": algo["name"],
            "rank": algo["rank"],
            "rationale": algo.get("rationale", f"Suitable for {task_type} on {data_type} data"),
            "sklearn_class": sklearn_class,
            "hyperparameters": algo.get("hyperparameters", []),
        })

    # Generate suggested approach narrative
    top_algo = ranked[0] if ranked else None
    target = spec.get("target", "target variable")
    metric = spec.get("primary_metric", "accuracy")

    suggested_approach = (
        f"For this {task_type} task predicting '{target}' optimized for {metric}, "
        f"the recommended primary algorithm is {top_algo['display_name'] if top_algo else 'gradient boosting'}. "
        f"This recommendation is based on the {data_type} data modality and {data_size} dataset size. "
    )
    if top_algo and top_algo.get("data_requirements", {}).get("handles_missing"):
        suggested_approach += "The algorithm handles missing values natively, reducing preprocessing needs. "
    suggested_approach += (
        f"Recommended strategy: Start with {ranked[0]['display_name'] if ranked else 'ensemble methods'} "
        f"as the primary model, compare against {ranked[1]['display_name'] if len(ranked) > 1 else 'baseline'} "
        f"and {ranked[2]['display_name'] if len(ranked) > 2 else 'a linear baseline'} for benchmarking. "
        f"Use cross-validation with the '{spec.get('splits', {}).get('strategy', 'stratified')}' split strategy."
    )

    # Generate experiment suggestions
    experiment_suggestions = [
        f"Train {ranked[0]['display_name'] if ranked else 'primary model'} with default hyperparameters as baseline",
        f"Run hyperparameter optimization using Optuna with {spec.get('compute_budget', {}).get('trials', 50)} trials",
        f"Compare top 3 algorithms ({', '.join(a['display_name'] for a in ranked[:3])}) with same feature set",
    ]
    if domain_notes:
        experiment_suggestions.append(
            "Apply domain-specific feature engineering based on domain notes"
        )
    experiment_suggestions.extend([
        "Evaluate feature importance and remove low-signal features",
        "Test imputation strategies if missing values present",
        f"Validate final model with {metric} on held-out test set",
    ])

    return {
        "recommended_algorithms": recommended_algorithms,
        "suggested_approach": suggested_approach,
        "experiment_suggestions": experiment_suggestions[:7],
    }
