"""Arbiter entry point — the deterministic D3 node of the orchestrator Graph.

Called as a plain Python function (and via @tool from the Strands graph). Loads the
grader for the spec's task_type, scores y_pred against y_true, evaluates acceptance
vs baseline, and emits an ArbiterReport.
"""

from __future__ import annotations

import math
import time
from typing import Any

from ..spec.schema import SpecModel
from .registry import get_grader
from .schema import ArbiterReport, InvalidSubmissionError, evaluate_acceptance


def _labelled_mask(y_true: Any) -> Any:
    """Boolean mask of the positions that HAVE a ground-truth label.

    ``pandas.isna`` rather than ``numpy.isnan`` so this works for object/string label arrays
    (classification) and datetime targets, not only floats — ``np.isnan`` raises on object dtype.
    """
    import numpy as np
    import pandas as pd

    return ~np.asarray(pd.isna(y_true))


def _drop_unlabelled(y_true: Any, y_pred: Any) -> tuple[Any, Any, int, int]:
    """Exclude positions with no ground truth, keeping ``y_true``/``y_pred`` index-aligned.

    A row whose LABEL is missing is ungradeable by definition — there is nothing to score a
    prediction against. Keeping such a row does not make grading stricter, it makes it impossible:
    a single ``NaN`` in ``y_true`` makes ``mean(...)`` ``NaN``, so the score is ``NaN`` for ANY
    model. Measured on the live GHCN run — the staged test split carries 61 unlabelled rows out of
    17,964, every iteration scored ``NaN``, and the same predictions score 0.0 RMSE over the 17,903
    rows that do have truth.

    Note the deliberate symmetry with training, which already drops rows with a missing target
    (``training_tools._preprocess_for_training``) because fitting on absent supervision is
    impossible and imputing a label would fabricate it. The grading side simply had no equivalent
    rule. Labels are NEVER imputed here either.

    Returns ``(y_true, y_pred, n_graded, n_excluded)``. When nothing is excluded the ORIGINAL
    objects are handed back untouched, so the overwhelmingly common fully-labelled path is
    byte-for-byte unchanged.
    """
    import numpy as np

    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)

    # Length agreement is checked HERE so the mask can be applied to both. The graders also
    # validate it; this keeps the same InvalidSubmissionError contract and message shape.
    if yt.shape[0] != yp.shape[0]:
        raise InvalidSubmissionError(
            f"shape mismatch: {yt.shape} vs {yp.shape}"
        )

    mask = _labelled_mask(yt)
    n_total = int(yt.shape[0])
    n_graded = int(mask.sum())

    if n_graded == 0:
        raise InvalidSubmissionError(
            f"no gradeable rows: all {n_total} y_true entries are unlabelled, so no metric can be "
            "computed. Stage a test split that has ground truth."
        )
    if n_graded == n_total:
        return y_true, y_pred, n_total, 0
    return yt[mask], yp[mask], n_graded, n_total - n_graded


def grade(
    *,
    spec: SpecModel,
    y_true: Any,
    y_pred: Any,
    baseline_score: float | None = None,
    baseline_slices: dict[str, float] | None = None,
    phase_log: list[dict[str, Any]] | None = None,
    llm_total_tokens: int = 0,
) -> ArbiterReport:
    """Score y_pred vs y_true per the spec's task_type and emit an ArbiterReport.

    Raises:
        InvalidSubmissionError: if y_pred is malformed (agent-fixable — reinject).
        KeyError: if spec.task_type has no registered grader (code bug).
    """
    t0 = time.monotonic()
    grader = get_grader(spec.task_type)

    # Grade only where there IS ground truth. The counts go into the report, because
    # silently changing which rows are scored changes what the score MEANS — ADR-0003 is about
    # honest grades, and "0.94 over 17,903 of 17,964 staged rows" is a different claim from
    # "0.94 over the test set".
    y_true, y_pred, n_graded, n_excluded = _drop_unlabelled(y_true, y_pred)

    overall = grader.grade_fn(y_true, y_pred)
    slices = grader.slice_fn(y_true, y_pred) if grader.slice_fn is not None else {}

    # A non-finite score must not travel. Before this, a NaN propagated all the way into
    # overall_score / delta_vs_baseline / every slice, and the run reported a populated
    # arbiter_report whose acceptance_reason was "delta_vs_baseline=nan fails >= 0.0" — a
    # number-shaped non-answer, four iterations in a row, with nothing naming the cause.
    if not math.isfinite(float(overall)):
        raise InvalidSubmissionError(
            f"{spec.primary_metric} computed as {overall!r} over {n_graded} labelled rows "
            f"({n_excluded} excluded as unlabelled). A non-finite score is not a grade — check "
            "y_pred for NaN/inf values, and for a degenerate target (e.g. zero variance) on this "
            "split."
        )

    # Baseline score: prefer explicit arg; else read from spec.baseline by primary_metric
    if baseline_score is None:
        baseline_score = _extract_baseline_score(spec)

    effective_delta = (
        overall - baseline_score if grader.direction == "maximize" else baseline_score - overall
    )

    passes, reason = evaluate_acceptance(
        overall=overall,
        baseline=baseline_score,
        direction=grader.direction,
        delta_rule=spec.acceptance.primary_metric_delta_vs_baseline,
        slices=slices,
        baseline_slices=baseline_slices,
        no_subgroup_regression=spec.acceptance.no_subgroup_regression,
    )

    return ArbiterReport(
        task_type=spec.task_type,
        primary_metric=spec.primary_metric,
        overall_score=float(overall),
        direction=grader.direction,
        baseline_score=float(baseline_score),
        delta_vs_baseline=float(effective_delta),
        slices={k: float(v) for k, v in slices.items()},
        passes_acceptance=passes,
        acceptance_reason=reason,
        wall_clock_seconds=time.monotonic() - t0,
        llm_total_tokens=llm_total_tokens,
        phase_log=list(phase_log or []),
        n_graded=n_graded,
        n_excluded_unlabelled=n_excluded,
    )


def _extract_baseline_score(spec: SpecModel) -> float:
    """Pull baseline score out of spec.baseline by the primary metric name."""
    raw = spec.baseline.model_dump()
    if spec.primary_metric in raw:
        return float(raw[spec.primary_metric])
    raise InvalidSubmissionError(
        f"baseline block has no value for primary_metric={spec.primary_metric!r}; "
        f"available keys: {sorted(k for k in raw if k != 'type')}"
    )
