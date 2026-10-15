"""Segmentation grader — mIoU as overall, per-class IoU as slices."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..registry import register
from ..schema import Grader, InvalidSubmissionError


def _check(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.shape != yp.shape:
        raise InvalidSubmissionError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    if yt.ndim not in (2, 3):
        raise InvalidSubmissionError(
            f"segmentation masks must be 2-D (H,W) or 3-D (N,H,W), got {yt.ndim}-D"
        )
    return yt, yp


def _per_class_iou(yt: np.ndarray, yp: np.ndarray, classes: list[int]) -> dict[int, float]:
    out: dict[int, float] = {}
    for c in classes:
        t = yt == c
        p = yp == c
        inter = np.logical_and(t, p).sum()
        union = np.logical_or(t, p).sum()
        out[c] = float(inter / union) if union > 0 else float("nan")
    return out


def _miou(y_true: Any, y_pred: Any) -> float:
    yt, yp = _check(y_true, y_pred)
    classes = sorted(set(np.unique(yt).tolist()) | set(np.unique(yp).tolist()))
    ious = _per_class_iou(yt, yp, classes)
    vals = [v for v in ious.values() if not np.isnan(v)]
    if not vals:
        raise InvalidSubmissionError("no overlapping classes between y_true and y_pred")
    return float(np.mean(vals))


def _slices(y_true: Any, y_pred: Any) -> dict[str, float]:
    yt, yp = _check(y_true, y_pred)
    classes = sorted(set(np.unique(yt).tolist()) | set(np.unique(yp).tolist()))
    ious = _per_class_iou(yt, yp, classes)
    return {f"iou_class_{c}": 0.0 if np.isnan(v) else v for c, v in ious.items()}


GRADER = Grader(
    task_type="segmentation",
    grade_fn=_miou,
    direction="maximize",
    slice_fn=_slices,
)
register(GRADER)
