"""Decision Engine — rules-based recommendation system.

Analyzes experiment results against spec requirements and produces
structured Decision recommendations. Pure Python, no LLM needed (v1).

The engine applies a priority-ordered set of rules:
  1. Target met → deploy
  2. Overfitting detected → more_data
  3. All approaches poor → different_approach
  4. Models close together → ensemble
  5. Close to threshold → tune_hyperparams
  6. Default fallback → different_approach (low confidence)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..evaluation.store import ExperimentResult
from .rules import (
    check_all_poor,
    check_diminishing_returns,
    check_model_diversity,
    check_models_close,
    check_overfitting,
    check_target_met,
    suggest_scaling_strategy,
)


@dataclass
class Decision:
    """Structured recommendation from the Decision Engine.

    Attributes:
        recommendation: One of "deploy", "retrain", "more_data",
            "different_approach", "tune_hyperparams", "ensemble".
        confidence: "high", "medium", or "low".
        rationale: Human-readable explanation of why this recommendation.
        next_steps: Specific actionable steps to take.
        risk_level: "low", "medium", or "high".
    """

    recommendation: str
    confidence: str
    rationale: str
    next_steps: list[str] = field(default_factory=list)
    risk_level: str = "medium"


class DecisionEngine:
    """Rules-based decision engine for ML experiment analysis.

    Analyzes experiment results against spec requirements and produces
    a Decision with recommendation, confidence, rationale, and next steps.

    Usage:
        engine = DecisionEngine()
        decision = engine.analyze(results=results, spec=spec_dict)
    """

    VALID_RECOMMENDATIONS: ClassVar[set[str]] = {
        "deploy",
        "retrain",
        "more_data",
        "different_approach",
        "tune_hyperparams",
        "ensemble",
    }

    def analyze(
        self,
        results: list[ExperimentResult],
        spec: dict[str, Any],
    ) -> Decision:
        """Analyze results against spec requirements and recommend next action.

        Args:
            results: List of ExperimentResult from the evaluation store.
            spec: Dict with threshold, baseline_score, direction, primary_metric.

        Returns:
            Decision with recommendation, confidence, rationale, and next_steps.
        """
        threshold = float(spec.get("threshold", 0.75))
        baseline_score = float(spec.get("baseline_score", 0.0))

        # Filter to successful results only
        successful = [r for r in results if r.status == "success"]

        # Handle empty or all-failed results
        if not successful:
            return Decision(
                recommendation="different_approach",
                confidence="low",
                rationale="No successful experiment results available. All experiments failed.",
                next_steps=[
                    "Review error logs from failed experiments",
                    "Check data quality and preprocessing pipeline",
                    "Try simpler algorithms as a starting point",
                ],
                risk_level="high",
            )

        best_result = max(successful, key=lambda r: r.primary_score)
        best_score = best_result.primary_score

        # ── Rule 1: Target Met → Deploy ──────────────────────────────────
        if check_target_met(successful, threshold):
            delta = best_score - baseline_score
            confidence = self._compute_deploy_confidence(best_score, threshold, delta)
            scaling = suggest_scaling_strategy(best_result)

            return Decision(
                recommendation="deploy",
                confidence=confidence,
                rationale=(
                    f"Best model ({best_result.algorithm}) achieves {best_result.primary_metric}="
                    f"{best_score:.4f}, exceeding threshold {threshold:.4f} "
                    f"(delta vs baseline: +{delta:.4f})."
                ),
                next_steps=[
                    f"Register {best_result.algorithm} model in model registry",
                    "Deploy to inference endpoint with monitoring",
                    "Set up model performance monitoring and drift detection",
                    "Create A/B test plan for production validation",
                    f"Scaling strategy: {scaling}",
                ],
                risk_level="low" if confidence == "high" else "medium",
            )

        # ── Rule 2: Overfitting → More Data ─────────────────────────────
        if check_overfitting(successful):
            return Decision(
                recommendation="more_data",
                confidence="medium",
                rationale=(
                    "High train-test performance gap detected, indicating overfitting. "
                    f"Best test score: {best_score:.4f} vs threshold: {threshold:.4f}. "
                    "Model is memorizing training data rather than learning generalizable patterns."
                ),
                next_steps=[
                    "Collect more training data to reduce overfitting",
                    "Apply stronger regularization (L1/L2, dropout)",
                    "Reduce model complexity (fewer features, simpler architecture)",
                    "Use cross-validation to get more reliable estimates",
                    "Try data augmentation techniques",
                ],
                risk_level="medium",
            )

        # ── Rule 3: All Poor → Different Approach ────────────────────────
        if check_all_poor(successful, threshold):
            return Decision(
                recommendation="different_approach",
                confidence="medium",
                rationale=(
                    f"All {len(successful)} experiments performed below 60% of threshold "
                    f"({threshold * 0.60:.4f}). Best score: {best_score:.4f}. "
                    "Current approach family is fundamentally insufficient."
                ),
                next_steps=[
                    "Re-examine problem formulation and feature engineering",
                    "Try different algorithm families (neural nets, SVMs, etc.)",
                    "Investigate data quality — possible label noise or missing features",
                    "Consider problem reformulation (e.g., regression → classification)",
                    "Review domain literature for proven approaches",
                ],
                risk_level="high",
            )

        # ── Rule 4: Models Close → Ensemble ──────────────────────────────
        if check_models_close(successful, tolerance=0.02) and len(successful) >= 2:
            top_models = sorted(successful, key=lambda r: r.primary_score, reverse=True)[:3]
            model_names = [r.algorithm for r in top_models]

            return Decision(
                recommendation="ensemble",
                confidence="medium",
                rationale=(
                    f"Top {len(top_models)} models are within 2% of each other: "
                    f"{', '.join(f'{r.algorithm}={r.primary_score:.4f}' for r in top_models)}. "
                    "Ensemble combination likely to exceed individual models."
                ),
                next_steps=[
                    f"Create voting ensemble from: {', '.join(model_names)}",
                    "Try stacking with a meta-learner",
                    "Experiment with weighted averaging based on validation scores",
                    "Evaluate ensemble on held-out test set",
                ],
                risk_level="low",
            )

        # ── Rule 5: Close to Threshold → Tune Hyperparams ───────────────
        gap = threshold - best_score
        threshold_5_pct = threshold * 0.05
        if gap <= threshold_5_pct and gap > 0:
            return Decision(
                recommendation="tune_hyperparams",
                confidence="medium",
                rationale=(
                    f"Best model ({best_result.algorithm}) at {best_score:.4f} is within "
                    f"5% of threshold {threshold:.4f} (gap: {gap:.4f}). "
                    "Fine-tuning hyperparameters likely to close the gap."
                ),
                next_steps=[
                    f"Run hyperparameter search on {best_result.algorithm}",
                    "Try Bayesian optimization or random search with more trials",
                    "Experiment with learning rate schedules",
                    "Increase ensemble size or tree depth",
                    "Consider feature selection to reduce noise",
                ],
                risk_level="low",
            )

        # ── Fallback: Low Confidence Different Approach ──────────────────
        return Decision(
            recommendation="different_approach",
            confidence="low",
            rationale=(
                f"No clear pattern identified. Best score: {best_score:.4f}, "
                f"threshold: {threshold:.4f}, gap: {gap:.4f}. "
                "Results are mediocre — not terrible, not close. "
                "Consider fundamentally different approaches."
            ),
            next_steps=[
                "Try a different algorithm family",
                "Revisit feature engineering",
                "Check if problem formulation is correct",
                "Consider collecting domain expert input",
            ],
            risk_level="medium",
        )

    def _compute_deploy_confidence(
        self,
        best_score: float,
        threshold: float,
        delta: float,
    ) -> str:
        """Compute confidence level for a deploy recommendation.

        High: score significantly above threshold (>10% margin)
        Medium: score meets threshold but not by much
        Low: borderline case
        """
        margin = best_score - threshold
        if margin > threshold * 0.10:
            return "high"
        elif margin > threshold * 0.02:
            return "medium"
        else:
            return "low"

    def _check_target_met(self, results: list[ExperimentResult], spec: dict[str, Any]) -> bool:
        """Is the best result meeting the spec's success threshold?"""
        threshold = float(spec.get("threshold", 0.75))
        return check_target_met(results, threshold)

    def _check_diminishing_returns(self, results: list[ExperimentResult]) -> bool:
        """Are improvements getting smaller across iterations?"""
        return check_diminishing_returns(results)

    def _check_model_diversity(self, results: list[ExperimentResult]) -> bool:
        """Have we tried enough different approaches?"""
        return check_model_diversity(results)

    def _suggest_scaling_strategy(self, best_result: ExperimentResult) -> str:
        """Based on best model, suggest scaling approach."""
        return suggest_scaling_strategy(best_result)
