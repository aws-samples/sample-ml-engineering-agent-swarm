"""``evaluation_flow`` — Stage 8 evaluate flow over Evaluation + ExperimentTracking ports.

The invariant Stage-8 sequence (doc 06 §5.3 evaluate-flow row):

    **run offline metrics + the deterministic Arbiter verdict → build a model-comparison
    table over ExperimentTracking**

Non-delegated Arbiter (ADR-003): the accept/reject DECISION is the deterministic
``EvaluationBackend.grade`` — NEVER an LLM/managed call. The flow surfaces the Arbiter's
verdict verbatim and ranks candidates via the EXISTING leaderboard comparison IP; it
does not make or override the verdict. Read-only / cheap → not approval-gated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..backends.ports import EvaluationBackend, ExperimentTrackingBackend
from ..backends.registry import REGISTRY

__all__ = ["EvaluationFlowResult", "run_evaluation_flow"]


@dataclass
class EvaluationFlowResult:
    """The invariant output contract of one ``evaluation_flow`` run."""

    status: str  # "completed" | "failed"
    arbiter_verdict: str = ""
    arbiter_report: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    comparison_table: list[dict[str, Any]] = field(default_factory=list)
    best_run: dict[str, Any] = field(default_factory=dict)
    adapter: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


def run_evaluation_flow(
    *,
    spec: Any,
    y_true: Any,
    y_pred: Any,
    baseline_score: float | None = None,
    experiment: str | None = None,
    direction: str = "maximize",
    steering: Any | None = None,
) -> EvaluationFlowResult:
    """Run the evaluate flow: deterministic Arbiter verdict + model-comparison table.

    Fixed steps:
      1. **grade** ``y_pred`` vs ``y_true`` via the deterministic Arbiter
         (``EvaluationBackend.grade`` — NON-DELEGATED, ADR-003) → verdict + metrics;
      2. **compare** — when ``experiment`` is supplied, build the ranked model-comparison
         table over ``ExperimentTrackingBackend`` (the EXISTING leaderboard ranking IP) and
         surface the best run alongside the Arbiter verdict.

    Args:
        spec / y_true / y_pred / baseline_score: The Arbiter grade inputs (unchanged shape).
        experiment: Optional experiment to build the comparison table for.
        direction: Ranking direction for the comparison (``maximize`` / ``minimize``).
        steering: Forwarded to the registry resolution.

    Returns:
        An :class:`EvaluationFlowResult` (Arbiter verdict + metrics + comparison table +
        best run).
    """
    # ── 1. deterministic Arbiter grade (NON-DELEGATED — ADR-003) ────────────────
    try:
        ev = REGISTRY.resolve(EvaluationBackend, steering=steering)
        report = ev.adapter.grade(spec=spec, y_true=y_true, y_pred=y_pred, baseline_score=baseline_score)
        rd = report.model_dump() if hasattr(report, "model_dump") else dict(report)
    except Exception as e:
        return EvaluationFlowResult(status="failed", detail={"stage": "grade", "error": str(e)})

    verdict = str(rd.get("verdict", rd.get("decision", "")))
    metrics = rd.get("metrics", rd.get("scores", {}))

    # ── 2. model-comparison table over ExperimentTracking (EXISTING leaderboard) ─
    comparison: list[dict[str, Any]] = []
    best_run: dict[str, Any] = {}
    adapter_name = ev.adapter_name
    if experiment:
        try:
            trk = REGISTRY.resolve(ExperimentTrackingBackend)
            adapter_name = trk.adapter_name
            if hasattr(trk.adapter, "build_comparison_table"):
                comparison = trk.adapter.build_comparison_table(experiment=experiment, direction=direction)
                best_run = dict(comparison[0]) if comparison else {}
        except Exception as e:
            comparison = []
            best_run = {"comparison_error": str(e)}

    return EvaluationFlowResult(
        status="completed",
        arbiter_verdict=verdict,
        arbiter_report=rd,
        metrics=metrics if isinstance(metrics, dict) else {},
        comparison_table=comparison,
        best_run=best_run,
        adapter=adapter_name,
        detail={"experiment": experiment, "direction": direction},
    )
