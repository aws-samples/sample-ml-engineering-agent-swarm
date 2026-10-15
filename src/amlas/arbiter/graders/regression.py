"""Regression grader — RMSE as overall, MAE as secondary slice.

Adds the C-MAPSS asymmetric scoring function in slices (standard in RUL prediction —
over-prediction penalised more heavily than under-prediction).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..registry import register
from ..schema import Grader, InvalidSubmissionError


def _as_1d(x: Any, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise InvalidSubmissionError(f"{name} must be 1-D, got shape {arr.shape}")
    return arr


def _rmse(y_true: Any, y_pred: Any) -> float:
    yt = _as_1d(y_true, "y_true")
    yp = _as_1d(y_pred, "y_pred")
    if yt.shape != yp.shape:
        raise InvalidSubmissionError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    if len(yt) == 0:
        raise InvalidSubmissionError("empty arrays — cannot grade")
    if np.isnan(yp).any() or np.isinf(yp).any():
        raise InvalidSubmissionError("y_pred contains NaN/inf")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def _slices(y_true: Any, y_pred: Any) -> dict[str, float]:
    yt = _as_1d(y_true, "y_true")
    yp = _as_1d(y_pred, "y_pred")
    mae = float(np.mean(np.abs(yt - yp)))
    # C-MAPSS asymmetric: over-prediction penalised more
    d = yp - yt
    cmapss = float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))
    return {"mae": mae, "cmapss_score": cmapss}


GRADER = Grader(
    task_type="regression",
    grade_fn=_rmse,
    direction="minimize",
    slice_fn=_slices,
)
register(GRADER)
