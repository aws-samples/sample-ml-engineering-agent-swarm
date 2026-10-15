"""Detection grader — IoU-threshold mAP (simplified, single-threshold).

For v1 we implement mAP@0.5 only. Expected input:
    y_true: list[dict{boxes: [[x1,y1,x2,y2], ...], labels: [int, ...]}]  per image
    y_pred: list[dict{boxes, labels, scores}]  per image
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..registry import register
from ..schema import Grader, InvalidSubmissionError


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def _validate_image(d: dict[str, Any], idx: int, need_scores: bool) -> None:
    for key in ("boxes", "labels"):
        if key not in d:
            raise InvalidSubmissionError(f"image {idx} missing key {key!r}")
    if need_scores and "scores" not in d:
        raise InvalidSubmissionError(f"image {idx} missing key 'scores' in y_pred")
    boxes = np.asarray(d["boxes"], dtype=float)
    if (boxes.size and boxes.ndim != 2) or (boxes.size and boxes.shape[1] != 4):
        raise InvalidSubmissionError(f"image {idx}: boxes must be shape (N,4), got {boxes.shape}")


def _map_at_iou(y_true: list[Any], y_pred: list[Any], iou_thresh: float = 0.5) -> float:
    if len(y_true) != len(y_pred):
        raise InvalidSubmissionError(
            f"len(y_true)={len(y_true)} != len(y_pred)={len(y_pred)}"
        )
    if not y_true:
        raise InvalidSubmissionError("empty prediction list — cannot grade")

    # Collect per-class: {cls: {scores: [...], tp: [...], fp: [...], n_gt: int}}
    classes: dict[int, dict[str, list[float] | int]] = {}

    for i, (gt, pr) in enumerate(zip(y_true, y_pred, strict=False)):
        _validate_image(gt, i, need_scores=False)
        _validate_image(pr, i, need_scores=True)
        gt_boxes = np.asarray(gt["boxes"], dtype=float).reshape(-1, 4)
        gt_labels = list(gt["labels"])
        pr_boxes = np.asarray(pr["boxes"], dtype=float).reshape(-1, 4)
        pr_labels = list(pr["labels"])
        pr_scores = list(pr["scores"])

        for c in set(gt_labels) | set(pr_labels):
            classes.setdefault(c, {"scores": [], "tp": [], "fp": [], "n_gt": 0})
        for c in set(gt_labels):
            classes[c]["n_gt"] += gt_labels.count(c)  # type: ignore[operator]

        # Sort predictions by score desc
        order: Any = np.argsort(-np.asarray(pr_scores)) if pr_scores else []
        matched_gt: set[int] = set()
        for j in order:
            c = pr_labels[j]
            # Candidate gt indices of this class
            gt_cand = [k for k, gc in enumerate(gt_labels) if gc == c and k not in matched_gt]
            best_iou: float = 0.0
            best_k: int = -1
            for k in gt_cand:
                iou = _iou_xyxy(pr_boxes[j], gt_boxes[k])
                if iou > best_iou:
                    best_iou, best_k = iou, k
            tp = 1.0 if best_iou >= iou_thresh and best_k >= 0 else 0.0
            fp = 1.0 - tp
            if tp and best_k >= 0:
                matched_gt.add(best_k)
            classes[c]["scores"].append(pr_scores[j])  # type: ignore[union-attr]
            classes[c]["tp"].append(tp)  # type: ignore[union-attr]
            classes[c]["fp"].append(fp)  # type: ignore[union-attr]

    # Compute AP per class (11-point interpolation, VOC-style)
    aps: list[float] = []
    for _c, s in classes.items():
        n_gt = s["n_gt"]
        if n_gt == 0:
            continue
        scores = np.asarray(s["scores"], dtype=float)
        if scores.size == 0:
            aps.append(0.0)
            continue
        order = np.argsort(-scores)
        tp = np.asarray(s["tp"], dtype=float)[order]
        fp = np.asarray(s["fp"], dtype=float)[order]
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
        recall = cum_tp / float(n_gt)  # type: ignore[arg-type]
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            p_at_t = precision[recall >= t].max() if (recall >= t).any() else 0.0
            ap += p_at_t / 11.0
        aps.append(float(ap))

    return float(np.mean(aps)) if aps else 0.0


def _grade(y_true: Any, y_pred: Any) -> float:
    return _map_at_iou(y_true, y_pred, iou_thresh=0.5)


def _slices(y_true: Any, y_pred: Any) -> dict[str, float]:
    return {
        "map_0.5": _map_at_iou(y_true, y_pred, 0.5),
        "map_0.75": _map_at_iou(y_true, y_pred, 0.75),
    }


GRADER = Grader(
    task_type="detection",
    grade_fn=_grade,
    direction="maximize",
    slice_fn=_slices,
)
register(GRADER)
