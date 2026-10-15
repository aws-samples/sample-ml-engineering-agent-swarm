"""ML Pipeline Orchestrator — wires all components end-to-end.

Pure Python orchestrator (no LLM needed) that chains:
  Spec → Research → Experiment Plan → Train → Evaluate → Leaderboard → Decision → Model Card

Usage:
    pipeline = MLPipeline(output_dir="/tmp/pipeline_run")
    result = pipeline.run(spec_dict=spec, data_path="repos/adsa/train.csv")
    print(result.model_card)
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..decisions.engine import Decision, DecisionEngine
from ..decisions.model_card import generate_model_card
from ..evaluation.leaderboard import Leaderboard
from ..evaluation.store import EvaluationStore, ExperimentResult
from ..experiments.config import ExperimentPlan
from ..experiments.runner import ExperimentRunner
from ..tools.research_tools import recommend_approaches

logger = logging.getLogger(__name__)

# Metrics where a LOWER value is better (regression error metrics).
_MINIMIZE_METRICS = {"rmse", "mae", "mse", "mape", "msle", "rmsle", "log_loss"}


def _metric_direction(primary_metric: str, task_type: str) -> str:
    """Return "maximize" or "minimize" for the given metric + task (REG-02).

    Regression error metrics (rmse/mae/mse/...) minimize; r2 and classification
    metrics (f1/accuracy/...) maximize.
    """
    metric = (primary_metric or "").strip().lower()
    if metric in _MINIMIZE_METRICS:
        return "minimize"
    if task_type == "regression" and metric not in {"r2", "r²", "explained_variance"}:
        # Unknown regression metric names default to error-style (minimize).
        return "minimize"
    return "maximize"


def _parse_delta(delta_expr: object) -> float:
    """Parse an acceptance delta like '>= 0.02' / '<= -0.05' into a float magnitude.

    Returns the absolute improvement magnitude (always >= 0). Falls back to 0.0
    on any parse failure.
    """
    if isinstance(delta_expr, (int, float)):
        return abs(float(delta_expr))
    if not isinstance(delta_expr, str):
        return 0.0
    import re

    m = re.search(r"-?\d+(\.\d+)?", delta_expr)
    return abs(float(m.group())) if m else 0.0


def _resolve_pipeline_spec(spec_dict: dict[str, Any]) -> dict[str, Any]:
    """Adapt a spec dict (flat OR nested SpecModel shape) to the flat dict the
    pipeline + DecisionEngine expect (SPEC-01).

    Resolves:
      - primary_metric, task_type, target, project_id, golden_dataset
      - baseline_score: flat `baseline_score`, else nested `baseline[primary_metric]`
      - threshold: flat `threshold`, else derived from baseline ± acceptance delta
                   (baseline + delta for maximize, baseline - delta for minimize)
      - direction: from task_type/metric (REG-02)
    """
    task_type = spec_dict.get("task_type", "classification")
    primary_metric = spec_dict.get("primary_metric", "f1_macro")
    direction = _metric_direction(primary_metric, task_type)

    # Baseline score — prefer flat, fall back to nested baseline[metric].
    baseline_score = spec_dict.get("baseline_score")
    if baseline_score is None:
        baseline = spec_dict.get("baseline")
        if isinstance(baseline, dict):
            baseline_score = baseline.get(primary_metric)
    baseline_score = float(baseline_score) if baseline_score is not None else 0.0

    # Threshold — prefer flat, else derive from baseline + acceptance delta.
    threshold = spec_dict.get("threshold")
    if threshold is None:
        acceptance = spec_dict.get("acceptance")
        delta = 0.0
        if isinstance(acceptance, dict):
            delta = _parse_delta(acceptance.get("primary_metric_delta_vs_baseline"))
        if direction == "minimize":
            threshold = baseline_score - delta
        else:
            threshold = baseline_score + delta
    threshold = float(threshold)

    # golden_dataset — prefer flat key; tolerate nested absence.
    golden_dataset = spec_dict.get("golden_dataset", "")

    return {
        "project_id": spec_dict.get("project_id", spec_dict.get("spec_id", "unknown")),
        "task_type": task_type,
        "target": spec_dict.get("target", "target"),
        "primary_metric": primary_metric,
        "golden_dataset": golden_dataset,
        "threshold": threshold,
        "baseline_score": baseline_score,
        "direction": direction,
    }


@dataclass
class PipelineResult:
    """Full pipeline output containing all artifacts and intermediate results."""

    spec: dict[str, Any]
    recommendations: dict[str, Any]
    plan: ExperimentPlan
    results: list[ExperimentResult]
    leaderboard: str  # Markdown
    decision: Decision
    model_card: str  # Markdown
    artifacts_dir: str
    total_duration_seconds: float


class MLPipeline:
    """Orchestrates the full ML engineering pipeline.

    Wires together: research → experiment plan → training → evaluation →
    leaderboard → decision → model card.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        store_path = self._output_dir / "evaluation_store.json"
        self._store = EvaluationStore(store_path)
        self._runner = ExperimentRunner(store=self._store)
        self._leaderboard = Leaderboard(store=self._store)
        self._decision_engine = DecisionEngine()

    @property
    def store(self) -> EvaluationStore:
        """Access the underlying evaluation store."""
        return self._store

    @property
    def leaderboard(self) -> Leaderboard:
        """Access the leaderboard."""
        return self._leaderboard

    def run(
        self,
        spec_dict: dict[str, Any],
        data_path: str | None = None,
        on_stage: Callable[[str, str], None] | None = None,
    ) -> PipelineResult:
        """Run the complete pipeline: research → plan → experiment → evaluate → decide.

        Args:
            spec_dict: Dict with spec fields (project_id, task_type, target,
                       primary_metric, golden_dataset, threshold, baseline_score).
            data_path: Override data path (uses spec's golden_dataset if not provided).
            on_stage: Optional progress callback ``on_stage(stage_key, human_label)``
                invoked as each pipeline stage begins. : the TUI passes this to
                publish per-lead ``AgentActivity`` to the Dashboard's Agent Logs pane, so
                the seven roster leads light up on a local ``/run`` (the local pipeline
                otherwise writes NO trace file for the trace poller to read). Null-safe +
                best-effort — a callback error never breaks the run.

        Returns:
            PipelineResult with all intermediate and final outputs.
        """
        start_time = time.time()

        def _stage(stage_key: str, label: str) -> None:
            if on_stage is None:
                return
            try:
                on_stage(stage_key, label)
            except Exception:  # best-effort observability — never break the run
                logger.debug("on_stage callback failed for %s", stage_key, exc_info=True)

        # Resolve spec (flat OR nested SpecModel shape) into the flat dict the
        # pipeline + decision layer expect (SPEC-01), incl. optimization
        # direction derived from task_type/metric (REG-02).
        resolved = _resolve_pipeline_spec(spec_dict)
        task_type = resolved["task_type"]
        primary_metric = resolved["primary_metric"]
        direction = resolved["direction"]
        spec_id = resolved["project_id"]

        # Resolve data path
        effective_data_path = data_path or resolved["golden_dataset"]

        # ─── Step 1: Research → Recommendations ────────────────────────────
        _stage("research", "researching approaches")
        spec_json = json.dumps(spec_dict)
        recommendations = recommend_approaches(spec_json=spec_json)

        # ─── Step 2: Create Experiment Plan ────────────────────────────────
        _stage("plan", "building experiment plan")
        plan_spec = {
            "spec_id": spec_id,
            "task_type": task_type,
            "target": resolved["target"],
            "primary_metric": primary_metric,
            "data_path": effective_data_path,
        }
        plan = self._runner.create_plan_from_recommendations(plan_spec, recommendations)

        # ─── Step 3: Run Experiments (train + evaluate) ────────────────────
        _stage("train", "training + evaluating models")
        results = self._runner.run_plan(plan)

        # ─── Step 4: Generate Leaderboard (direction-aware — REG-02) ───────
        _stage("evaluate", "ranking the leaderboard")
        leaderboard_md = self._leaderboard.format_leaderboard_markdown(
            spec_id=spec_id,
            metric=primary_metric,
            direction=direction,
        )

        # ─── Step 5: Decision Engine ──────────────────────────────────────
        _stage("decision", "go / no-go decision")
        decision_spec = {
            "threshold": resolved["threshold"],
            "baseline_score": resolved["baseline_score"],
            "primary_metric": primary_metric,
            "direction": direction,
        }
        successful_results = [r for r in results if r.status == "success"]
        decision = self._decision_engine.analyze(
            results=successful_results,
            spec=decision_spec,
        )

        # ─── Step 6: Model Card for Best Model (direction-aware — REG-02) ──
        best_result = self._store.get_best_run(spec_id=spec_id, direction=direction)
        model_card_md = ""
        if best_result:
            card_spec = {
                "project_id": spec_id,
                "task_type": task_type,
                "target": resolved["target"],
                "primary_metric": primary_metric,
                "threshold": resolved["threshold"],
                "baseline_score": resolved["baseline_score"],
            }
            model_card_md = generate_model_card(result=best_result, spec=card_spec)

        # ─── Save Artifacts ────────────────────────────────────────────────
        artifacts_dir = self._output_dir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Save leaderboard markdown
        (artifacts_dir / "leaderboard.md").write_text(leaderboard_md, encoding="utf-8")

        # Save decision
        decision_data = {
            "recommendation": decision.recommendation,
            "confidence": decision.confidence,
            "rationale": decision.rationale,
            "next_steps": decision.next_steps,
            "risk_level": decision.risk_level,
        }
        (artifacts_dir / "decision.json").write_text(
            json.dumps(decision_data, indent=2), encoding="utf-8"
        )

        # Save model card
        if model_card_md:
            (artifacts_dir / "model_card.md").write_text(model_card_md, encoding="utf-8")

        _stage("complete", "run complete")
        total_duration = time.time() - start_time

        return PipelineResult(
            spec=spec_dict,
            recommendations=recommendations,
            plan=plan,
            results=results,
            leaderboard=leaderboard_md,
            decision=decision,
            model_card=model_card_md,
            artifacts_dir=str(artifacts_dir),
            total_duration_seconds=total_duration,
        )

    def get_summary(self) -> str:
        """Generate a markdown summary of the pipeline's evaluation store."""
        all_results = self._store.get_all()
        if not all_results:
            return "# Pipeline Summary\n\nNo experiments recorded yet."

        successful = [r for r in all_results if r.status == "success"]
        failed = [r for r in all_results if r.status != "success"]

        lines = [
            "# Pipeline Summary",
            "",
            f"**Total Experiments**: {len(all_results)}",
            f"**Successful**: {len(successful)}",
            f"**Failed**: {len(failed)}",
            "",
        ]

        if successful:
            best = max(successful, key=lambda r: r.primary_score)
            lines.extend([
                "## Best Result",
                "",
                f"- **Algorithm**: {best.algorithm}",
                f"- **{best.primary_metric}**: {best.primary_score:.4f}",
                f"- **Duration**: {best.duration_seconds:.2f}s",
                "",
            ])

        return "\n".join(lines)
