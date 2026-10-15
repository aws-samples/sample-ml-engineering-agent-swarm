"""Arbiter schema — Grader + ArbiterReport + InvalidSubmissionError.

Mirrors `openai/mle-bench` `mlebench/grade.py` + `grade_helpers.py` with Amlas extensions:
    - wall_clock_seconds, llm_total_tokens: run-cost surfacing
    - phase_log: per-iteration score trail for kanban / retreat reasoning
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class InvalidSubmissionError(Exception):
    """Raised by a grader when the agent's submission is malformed.

    Signals an agent-fixable error — message is reinjected into the agent loop via
    Strands' native tool-error-as-toolResult path (see PLAN.md §13 gotcha #4).
    """


Direction = Literal["maximize", "minimize"]


@dataclass(frozen=True)
class Grader:
    """Binding of task_type → scoring function.

    Args:
        task_type: key used by the Arbiter registry.
        grade_fn: (y_true, y_pred) -> scalar score. Must raise InvalidSubmissionError on
            submission-side errors (wrong shape, out-of-range values, missing rows).
            Assertions / other exceptions propagate as programmer errors.
        direction: "maximize" means higher is better (F1, AUC, mIoU); "minimize" for RMSE/MAE.
        slice_fn: optional (y_true, y_pred) -> dict[str, float] for per-slice metrics
            (e.g., per-class IoU, per-subgroup F1).
    """

    task_type: str
    grade_fn: Callable[[Any, Any], float]
    direction: Direction
    slice_fn: Callable[[Any, Any], dict[str, float]] | None = None


class ArbiterReport(BaseModel):
    """Canonical Arbiter output. Serialized to runs/<ts>/arbiter_report.json."""

    model_config = ConfigDict(extra="forbid")

    task_type: str
    primary_metric: str
    overall_score: float
    direction: Direction
    baseline_score: float
    delta_vs_baseline: float
    slices: dict[str, float] = Field(default_factory=dict)
    passes_acceptance: bool
    acceptance_reason: str

    # Amlas extensions — run cost + iteration trail
    wall_clock_seconds: float = 0.0
    llm_total_tokens: int = 0
    phase_log: list[dict[str, Any]] = Field(default_factory=list)

    #: How many rows the score was actually computed over, and how many were excluded because they
    #: carry no ground-truth label. Recorded because a score over a SUBSET is a different claim from
    #: a score over the whole staged split, and ADR-0003 is about honest grades — a reader must be
    #: able to tell which one they are looking at. ``n_excluded_unlabelled == 0`` is the normal case.
    n_graded: int = 0
    n_excluded_unlabelled: int = 0


def direction_is_better(direction: Direction, a: float, b: float) -> bool:
    """Return True if `a` is a better score than `b` per the given direction."""
    return a > b if direction == "maximize" else a < b


def evaluate_acceptance(
    *,
    overall: float,
    baseline: float,
    direction: Direction,
    delta_rule: str,
    slices: dict[str, float] | None = None,
    baseline_slices: dict[str, float] | None = None,
    no_subgroup_regression: bool = True,
) -> tuple[bool, str]:
    """Evaluate acceptance criteria from SpecModel against computed scores.

    `delta_rule` is a comparison string like ">= 0.005" applied to
    `overall - baseline` (maximize) or `baseline - overall` (minimize).
    """
    import operator
    import re

    m = re.match(r"^\s*(>=|<=|>|<|==)\s*(-?\d+(?:\.\d+)?)\s*$", delta_rule)
    if not m:  # schema validated already, but double-check
        raise ValueError(f"malformed delta rule: {delta_rule!r}")
    op_map = {">=": operator.ge, "<=": operator.le, ">": operator.gt, "<": operator.lt, "==": operator.eq}
    op = op_map[m.group(1)]
    threshold = float(m.group(2))

    effective_delta = overall - baseline if direction == "maximize" else baseline - overall

    delta_ok = bool(op(effective_delta, threshold))
    if not delta_ok:
        return False, (
            f"delta_vs_baseline={effective_delta:.6f} fails {delta_rule} "
            f"(overall={overall:.6f}, baseline={baseline:.6f}, direction={direction})"
        )

    if no_subgroup_regression and slices and baseline_slices:
        regressions = [
            k for k in slices
            if k in baseline_slices and not direction_is_better(direction, slices[k], baseline_slices[k])
            and not _approx_equal(slices[k], baseline_slices[k])
        ]
        if regressions:
            return False, f"subgroup regression in {regressions}"

    return True, f"delta {effective_delta:.6f} satisfies {delta_rule}; no subgroup regressions"


def _approx_equal(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol
