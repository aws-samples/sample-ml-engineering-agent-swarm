"""Experiment Runner — orchestrates multiple training experiments.

Pure Python orchestrator (no LLM needed). Takes a spec + research
recommendations as input, creates experiment configs, runs each
experiment via the training tools, records results, and produces
a comparison report with leaderboard.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from ..evaluation.leaderboard import Leaderboard
from ..evaluation.store import EvaluationStore, ExperimentResult, training_result_to_experiment
from ..tools.training_tools import train_model
from .config import ExperimentConfig, ExperimentPlan
from .hyperparams import DEFAULT_CONFIGS

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Orchestrates running multiple training experiments and tracking results.

    Usage:
        store = EvaluationStore("path/to/store.json")
        runner = ExperimentRunner(store=store)
        plan = runner.create_plan_from_recommendations(spec, recommendations)
        results = runner.run_plan(plan)
        report = runner.get_report(plan)
    """

    def __init__(self, store: EvaluationStore) -> None:
        self._store = store
        self._leaderboard = Leaderboard(store=store)

    def create_plan_from_recommendations(
        self,
        spec: dict[str, Any],
        recommendations: dict[str, Any],
    ) -> ExperimentPlan:
        """Convert research recommendations into an experiment plan.

        For each recommended algorithm, creates experiment configs using
        the default hyperparameter grid. If the algorithm isn't in the
        default grid, uses a single empty-params config.

        Args:
            spec: Dict with spec_id, task_type, target, primary_metric, data_path.
            recommendations: Output from recommend_approaches tool.

        Returns:
            ExperimentPlan with configs for all algorithm × hyperparameter combos.
        """
        spec_id = spec.get("spec_id", str(uuid.uuid4()))
        task_type = spec.get("task_type", "classification")
        target = spec.get("target", "target")
        primary_metric = spec.get("primary_metric", "f1_macro")
        data_path = spec.get("data_path", "")

        recommended_algorithms = recommendations.get("recommended_algorithms", [])

        configs: list[ExperimentConfig] = []

        for algo_rec in recommended_algorithms:
            algo_name = algo_rec["name"]
            # Get default hyperparameter variants for this algorithm
            param_variants = DEFAULT_CONFIGS.get(algo_name, [{}])

            for params in param_variants:
                config = ExperimentConfig(
                    experiment_id=str(uuid.uuid4()),
                    spec_id=spec_id,
                    algorithm=algo_name,
                    hyperparameters=dict(params),  # Copy to avoid mutation
                    data_path=data_path,
                    target_column=target,
                    task_type=task_type,
                    primary_metric=primary_metric,
                )
                configs.append(config)

        plan = ExperimentPlan(
            plan_id=str(uuid.uuid4()),
            spec_id=spec_id,
            configs=configs,
            created_at=datetime.now(UTC).isoformat(),
        )

        return plan

    def run_single(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment: load → preprocess → train → evaluate → record.

        Uses the train_model tool which handles preprocessing internally.
        Records the result to the evaluation store.

        Args:
            config: ExperimentConfig defining the experiment to run.

        Returns:
            ExperimentResult with metrics and status.
        """
        try:
            # Call training tool (handles load, preprocess, train, evaluate)
            training_result = train_model(
                data_path=config.data_path,
                target_column=config.target_column,
                algorithm=config.algorithm,
                params=config.hyperparameters,
                seed=42,
                test_size=0.2,
            )

            # Convert training result to ExperimentResult
            result = training_result_to_experiment(
                training_result=training_result,
                spec_id=config.spec_id,
                primary_metric=config.primary_metric,
            )

        except Exception as e:
            # Catch unexpected errors and create a failed result
            logger.error(f"Experiment {config.experiment_id} ({config.algorithm}) failed: {type(e).__name__}")
            result = ExperimentResult(
                run_id=str(uuid.uuid4()),
                spec_id=config.spec_id,
                algorithm=config.algorithm,
                hyperparameters=config.hyperparameters,
                metrics={},
                primary_metric=config.primary_metric,
                primary_score=0.0,
                duration_seconds=0.0,
                timestamp=datetime.now(UTC).isoformat(),
                artifact_path="",
                status="failed",
                notes=str(e),
            )

        # Record to store
        self._store.record(result)

        return result

    def run_plan(self, plan: ExperimentPlan) -> list[ExperimentResult]:
        """Run all experiments in a plan sequentially.

        Fault-tolerant: one failed experiment doesn't stop the others.
        All results (including failures) are recorded and returned.

        Args:
            plan: ExperimentPlan with configs to execute.

        Returns:
            List of ExperimentResult for all experiments (successes + failures).
        """
        results: list[ExperimentResult] = []

        for config in plan.configs:
            logger.info(
                f"Running experiment {config.experiment_id}: "
                f"{config.algorithm} with {config.hyperparameters}"
            )
            result = self.run_single(config)
            results.append(result)
            logger.info(
                f"  → {result.status}: {result.primary_metric}={result.primary_score:.4f}"
            )

        return results

    def get_report(self, plan: ExperimentPlan) -> str:
        """Generate a markdown comparison report for a plan's results.

        Includes:
          - Summary header with plan info
          - Leaderboard table ranked by primary metric
          - Best model highlight
          - Per-experiment details with hyperparameters

        Args:
            plan: ExperimentPlan to report on.

        Returns:
            Markdown-formatted report string.
        """
        spec_id = plan.spec_id
        results = self._store.get_by_spec(spec_id)
        successful = [r for r in results if r.status == "success"]
        failed = [r for r in results if r.status != "success"]

        # Determine primary metric from first config
        primary_metric = plan.configs[0].primary_metric if plan.configs else "f1_macro"

        lines: list[str] = []

        # Header
        lines.append(f"# Experiment Report: {spec_id}")
        lines.append("")
        lines.append(f"**Plan ID**: {plan.plan_id}")
        lines.append(f"**Total Experiments**: {len(results)}")
        lines.append(f"**Successful**: {len(successful)}")
        lines.append(f"**Failed**: {len(failed)}")
        lines.append(f"**Primary Metric**: {primary_metric}")
        lines.append("")

        # Leaderboard
        if successful:
            # Sort by primary score descending
            ranked = sorted(successful, key=lambda r: r.primary_score, reverse=True)

            lines.append("## Leaderboard")
            lines.append("")
            lines.append(f"| Rank | Algorithm | {primary_metric} | Hyperparameters | Duration (s) |")
            lines.append("|------|-----------|" + "-" * (len(primary_metric) + 2) + "|-----------------|--------------|")

            for i, r in enumerate(ranked, 1):
                score = f"{r.primary_score:.4f}"
                params_str = json.dumps(r.hyperparameters) if r.hyperparameters else "{}"
                if len(params_str) > 40:
                    params_str = params_str[:37] + "..."
                duration = f"{r.duration_seconds:.2f}"
                lines.append(f"| {i} | {r.algorithm} | {score} | {params_str} | {duration} |")

            lines.append("")

            # Best model highlight
            best = ranked[0]
            lines.append("## Best Model")
            lines.append("")
            lines.append(f"**Algorithm**: {best.algorithm}")
            lines.append(f"**{primary_metric}**: {best.primary_score:.4f}")
            lines.append(f"**Hyperparameters**: {json.dumps(best.hyperparameters, indent=2)}")
            lines.append(f"**Duration**: {best.duration_seconds:.2f}s")
            lines.append("")
        else:
            lines.append("## Leaderboard")
            lines.append("")
            lines.append("_No successful experiments to rank._")
            lines.append("")

        # Failed experiments
        if failed:
            lines.append("## Failed Experiments")
            lines.append("")
            for r in failed:
                lines.append(f"- **{r.algorithm}**: {r.notes}")
            lines.append("")

        return "\n".join(lines)


