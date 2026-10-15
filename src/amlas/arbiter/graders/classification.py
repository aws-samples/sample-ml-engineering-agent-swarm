"""Classification grader — F1-macro as overall, per-class F1 as slices."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import f1_score

from ..registry import register
from ..schema import Grader, InvalidSubmissionError


def _as_1d(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise InvalidSubmissionError(
            f"{name} must be 1-D, got shape {arr.shape}. For multi-class, pass class labels (not one-hot)."
        )
    return arr


def _grade(y_true: Any, y_pred: Any) -> float:
    yt = _as_1d(y_true, "y_true")
    yp = _as_1d(y_pred, "y_pred")
    if yt.shape != yp.shape:
        raise InvalidSubmissionError(
            f"shape mismatch: y_true={yt.shape} vs y_pred={yp.shape}"
        )
    if len(yt) == 0:
        raise InvalidSubmissionError("empty arrays — cannot grade")
    return float(f1_score(yt, yp, average="macro", zero_division=0))


def _slices(y_true: Any, y_pred: Any) -> dict[str, float]:
    yt = _as_1d(y_true, "y_true")
    yp = _as_1d(y_pred, "y_pred")
    classes = sorted(set(yt.tolist()) | set(yp.tolist()))
    per_class = f1_score(yt, yp, labels=classes, average=None, zero_division=0)
    return {f"f1_class_{c}": float(v) for c, v in zip(classes, per_class, strict=False)}


GRADER = Grader(
    task_type="classification",
    grade_fn=_grade,
    direction="maximize",
    slice_fn=_slices,
)
register(GRADER)
