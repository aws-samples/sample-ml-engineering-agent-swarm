"""Decision Rules — deterministic rule checks for the Decision Engine.

Pure Python, no LLM needed. Each function checks a specific condition
on experiment results and returns a boolean or string recommendation.
"""

from __future__ import annotations

from ..evaluation.store import ExperimentResult

# ─── Algorithm Family Classification ────────────────────────────────────────

# Map algorithm names to broad "families" for diversity checking
ALGORITHM_FAMILIES: dict[str, str] = {
    "logistic_regression": "linear",
    "linear_regression": "linear",
    "ridge": "linear",
    "lasso": "linear",
    "elastic_net": "linear",
    "svm": "kernel",
    "svc": "kernel",
    "svr": "kernel",
    "decision_tree": "tree",
    "random_forest": "ensemble_bagging",
    "extra_trees": "ensemble_bagging",
    "gradient_boosting": "ensemble_boosting",
    "xgboost": "ensemble_boosting",
    "lightgbm": "ensemble_boosting",
    "catboost": "ensemble_boosting",
    "adaboost": "ensemble_boosting",
    "knn": "instance_based",
    "k_neighbors": "instance_based",
    "naive_bayes": "probabilistic",
    "gaussian_nb": "probabilistic",
    "neural_network": "neural",
    "mlp": "neural",
}


def _get_family(algorithm: str) -> str:
    """Get the algorithm family for diversity checking."""
    return ALGORITHM_FAMILIES.get(algorithm.lower(), algorithm.lower())


# ─── Rule Check Functions ────────────────────────────────────────────────────


def check_target_met(results: list[ExperimentResult], threshold: float) -> bool:
    """Check if the best result meets or exceeds the target threshold.

    Args:
        results: List of experiment results.
        threshold: Target metric threshold to meet.

    Returns:
        True if best primary_score >= threshold.
    """
    if not results:
        return False
    successful = [r for r in results if r.status == "success"]
    if not successful:
        return False
    best_score = max(r.primary_score for r in successful)
    return best_score >= threshold


def check_diminishing_returns(results: list[ExperimentResult]) -> bool:
    """Check if improvements are getting smaller across iterations.

    Looks at successive improvements in primary_score (ordered by timestamp).
    Returns True if improvements are consistently decreasing.

    Args:
        results: List of experiment results (at least 3 needed).

    Returns:
        True if diminishing returns pattern detected.
    """
    if len(results) < 3:
        return False

    # Sort by timestamp to get chronological order
    sorted_results = sorted(results, key=lambda r: r.timestamp)
    scores = [r.primary_score for r in sorted_results]

    # Calculate improvements between successive results
    improvements = [scores[i + 1] - scores[i] for i in range(len(scores) - 1)]

    # Filter positive improvements only
    positive_improvements = [imp for imp in improvements if imp > 0]

    if len(positive_improvements) < 2:
        return False

    # Check if improvements are decreasing
    decreasing_count = sum(
        1
        for i in range(len(positive_improvements) - 1)
        if positive_improvements[i + 1] < positive_improvements[i]
    )

    # Diminishing if most improvements are getting smaller
    return decreasing_count >= len(positive_improvements) - 1


def check_model_diversity(results: list[ExperimentResult]) -> bool:
    """Check if enough different algorithm families have been tried.

    Args:
        results: List of experiment results.

    Returns:
        True if 3+ distinct algorithm families have been tried.
    """
    if not results:
        return False
    families = set(_get_family(r.algorithm) for r in results)
    return len(families) >= 3


def check_overfitting(results: list[ExperimentResult]) -> bool:
    """Check if there's evidence of overfitting (high train-test gap).

    Looks for a train score metric in the results and compares with test score.
    Overfitting defined as train-test gap > 0.15.

    Args:
        results: List of experiment results.

    Returns:
        True if overfitting pattern detected in any result.
    """
    if not results:
        return False

    for r in results:
        if r.status != "success":
            continue
        # Look for train score in metrics
        train_key = f"train_{r.primary_metric}"
        if train_key in r.metrics:
            train_score = r.metrics[train_key]
            test_score = r.primary_score
            gap = train_score - test_score
            if gap > 0.15:
                return True
    return False


def check_all_poor(results: list[ExperimentResult], threshold: float) -> bool:
    """Check if ALL results are below 60% of the threshold.

    This indicates fundamental approach problems.

    Args:
        results: List of experiment results.
        threshold: Target metric threshold.

    Returns:
        True if all successful results < 60% of threshold.
    """
    if not results:
        return True

    successful = [r for r in results if r.status == "success"]
    if not successful:
        return True

    poor_threshold = threshold * 0.60
    return all(r.primary_score < poor_threshold for r in successful)


def check_models_close(
    results: list[ExperimentResult],
    tolerance: float = 0.02,
) -> bool:
    """Check if top models are within tolerance of each other.

    This suggests an ensemble might help.

    Args:
        results: List of experiment results.
        tolerance: Maximum spread between top models (default 2%).

    Returns:
        True if top 3 (or all if fewer) models are within tolerance.
    """
    successful = [r for r in results if r.status == "success"]
    if len(successful) < 2:
        return False

    # Sort by score descending
    sorted_results = sorted(successful, key=lambda r: r.primary_score, reverse=True)
    top = sorted_results[: min(3, len(sorted_results))]

    # Check spread
    spread = top[0].primary_score - top[-1].primary_score
    return spread <= tolerance


def suggest_scaling_strategy(result: ExperimentResult) -> str:
    """Suggest deployment scaling strategy based on model type.

    Args:
        result: The best model result to suggest scaling for.

    Returns:
        Human-readable scaling strategy description.
    """
    family = _get_family(result.algorithm)

    if family == "linear":
        return (
            "Direct deploy with low resource requirements. "
            "Linear models have minimal inference latency and memory footprint. "
            "Standard CPU instance sufficient."
        )
    elif family in ("ensemble_bagging", "ensemble_boosting"):
        if result.algorithm in ("xgboost", "lightgbm", "catboost"):
            return (
                "Optimize inference with model serialization (ONNX/treelite). "
                "Consider batch prediction for throughput. "
                "Monitor memory usage with large forests."
            )
        return (
            "Standard deployment with moderate resource requirements. "
            "Ensemble models need more memory than linear models. "
            "Consider model compression if latency-sensitive."
        )
    elif family == "neural":
        return (
            "Optimize inference with model quantization or distillation. "
            "GPU instance may be needed for real-time serving. "
            "Implement batching for throughput optimization."
        )
    else:
        return (
            "Standard deployment. Evaluate inference latency requirements "
            "and scale resources accordingly."
        )
