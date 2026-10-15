"""6-dimension Data-Quality scorecard — master plan §2 (A5), extends the legacy
5-dimension :func:`score_data_quality` in ``tools/data_quality_tools.py``.

Dimensions (the existing 5 + **accuracy**):
  completeness, validity, consistency, accuracy, uniqueness, timeliness.

The legacy 5-dim tool is preserved untouched (its offline validation asserts an exact
5-key set); this module adds a NEW ``score_data_quality_6dim`` tool plus a pure
helper :func:`compute_six_dimensions` reused by ``scan_dq_patterns`` so the scan
report and the standalone scorecard agree.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from strands import tool

from ...errors import tool_error_boundary

DIMENSION_NAMES = (
    "completeness",
    "validity",
    "consistency",
    "accuracy",
    "uniqueness",
    "timeliness",
)

# Weights sum to 1.0 (accuracy carved out of the legacy completeness/validity mass).
_WEIGHTS = {
    "completeness": 0.25,
    "validity": 0.20,
    "consistency": 0.15,
    "accuracy": 0.15,
    "uniqueness": 0.15,
    "timeliness": 0.10,
}


def _is_numeric_str(val: Any) -> bool:
    if not isinstance(val, str):
        return False
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


def compute_six_dimensions(
    df: pd.DataFrame, target_column: str | None = None
) -> dict[str, dict[str, Any]]:
    """Return the 6 dimension scores (0-100) + human-readable details.

    Pure function (no I/O) so both the scorecard tool and ``scan_dq_patterns``
    can share one implementation.
    """
    n_rows, n_cols = df.shape
    dims: dict[str, dict[str, Any]] = {}

    # ── Completeness ─────────────────────────────────────────────────────────
    total_cells = n_rows * n_cols
    null_cells = int(df.isnull().sum().sum()) if total_cells else 0
    completeness_ratio = 1.0 - (null_cells / total_cells) if total_cells > 0 else 1.0
    dims["completeness"] = {
        "score": round(completeness_ratio * 100.0, 1),
        "details": f"{null_cells} null cells of {total_cells} ({completeness_ratio*100:.1f}% complete)",
    }

    # ── Uniqueness (duplicate rows) ──────────────────────────────────────────
    n_dup = int(df.duplicated().sum()) if n_rows else 0
    dup_ratio = n_dup / n_rows if n_rows > 0 else 0.0
    dims["uniqueness"] = {
        "score": round((1.0 - dup_ratio) * 100.0, 1),
        "details": f"{n_dup} duplicate rows of {n_rows} ({(1.0-dup_ratio)*100:.1f}% unique)",
    }

    # ── Consistency (mixed-type object columns) ──────────────────────────────
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    consistency_penalties = 0
    for col in obj_cols:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        numeric_ratio = non_null.apply(_is_numeric_str).mean()
        if 0.1 < numeric_ratio < 0.9:
            consistency_penalties += 1
    dims["consistency"] = {
        "score": round(max(0.0, 100.0 - consistency_penalties * 15.0), 1),
        "details": (
            f"{consistency_penalties} mixed-type columns"
            if consistency_penalties
            else "All columns have consistent types"
        ),
    }

    # ── Validity (inf + extreme >6σ) ─────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    validity_penalties = 0
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        if int(np.isinf(series).sum()) > 0:
            validity_penalties += 1
        if series.std() > 0:
            z = np.abs((series - series.mean()) / series.std())
            if int((z > 6).sum()) > 0:
                validity_penalties += 1
    dims["validity"] = {
        "score": round(max(0.0, 100.0 - validity_penalties * 10.0), 1),
        "details": (
            f"{validity_penalties} validity issues in numeric columns"
            if validity_penalties
            else "All numeric values within reasonable ranges"
        ),
    }

    # ── Accuracy (NEW: outlier rate + cross-field sanity proxy) ──────────────
    # Accuracy = how close values are to plausible reality. We proxy it with the
    # IQR-outlier rate across numeric columns (lower outlier rate → higher
    # accuracy). This is distinct from validity (hard impossibilities) — it
    # captures "suspicious but not impossible" values.
    outlier_cells = 0
    total_numeric_cells = 0
    for col in numeric_cols:
        series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            total_numeric_cells += len(series)
            continue
        lower, upper = q1 - 3.0 * iqr, q3 + 3.0 * iqr  # 3×IQR = "far" outliers
        outlier_cells += int(((series < lower) | (series > upper)).sum())
        total_numeric_cells += len(series)
    outlier_rate = outlier_cells / total_numeric_cells if total_numeric_cells else 0.0
    dims["accuracy"] = {
        "score": round(max(0.0, 100.0 - outlier_rate * 100.0 * 5.0), 1),
        "details": (
            f"{outlier_cells} far-outlier cells of {total_numeric_cells} numeric "
            f"({outlier_rate*100:.2f}% outlier rate)"
            if total_numeric_cells
            else "No numeric columns to assess accuracy"
        ),
    }

    # ── Timeliness ───────────────────────────────────────────────────────────
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    for col in obj_cols:
        sample = df[col].dropna().head(5)
        if len(sample) > 0:
            try:
                # Non-date object columns make pandas emit a noisy
                # "Could not infer format" UserWarning and fall back to slow
                # per-element parsing. We only care whether it parses at all,
                # so suppress the expected warning here.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    pd.to_datetime(sample)
                datetime_cols.append(col)
            except (ValueError, TypeError):
                pass
    dims["timeliness"] = {
        "score": 90.0 if datetime_cols else 80.0,
        "details": (
            f"{len(datetime_cols)} datetime columns detected"
            if datetime_cols
            else "No datetime columns detected — timeliness not assessable"
        ),
    }

    return dims


def overall_from_dimensions(dims: dict[str, dict[str, Any]]) -> float:
    """Weighted overall score (0-100) from the 6 dimension scores."""
    total = sum(float(dims[name]["score"]) * _WEIGHTS[name] for name in DIMENSION_NAMES)
    return round(total, 1)


@tool
@tool_error_boundary(source="tools.data_quality.score")
def score_data_quality_6dim(data_path: str, target_column: str | None = None) -> dict[str, Any]:
    """6-dimension data-quality scorecard (completeness, validity, consistency,
    accuracy, uniqueness, timeliness).

    Extends the legacy 5-dimension ``score_data_quality`` with an **accuracy**
    dimension. Returns overall score (0-100), per-dimension scores+details, and
    a natural-language summary.

    Args:
        data_path: Filesystem path to a CSV file.
        target_column: Optional name of the target column (reserved for future
            target-aware accuracy checks; currently informational).
    """
    p = Path(data_path)
    if not p.exists():
        return {"success": False, "error": f"File not found: {data_path}", "overall_score": 0}
    try:
        df = pd.read_csv(p)
    except Exception as e:
        return {"success": False, "error": f"Failed to read CSV: {e}", "overall_score": 0}

    if df.shape[0] == 0:
        dims: dict[str, dict[str, Any]] = {
            name: {"score": 100.0, "details": "empty dataset"} for name in DIMENSION_NAMES
        }
        return {
            "overall_score": 100.0,
            "dimensions": dims,
            "summary": "Empty dataset — no rows to assess; trivially scores 100/100.",
        }

    dims = compute_six_dimensions(df, target_column)
    overall = overall_from_dimensions(dims)

    weak = [name for name in DIMENSION_NAMES if float(dims[name]["score"]) < 80.0]
    if weak:
        summary = (
            f"6-dimension data quality score: {overall:.1f}/100. "
            f"Weak dimensions needing attention: {', '.join(weak)}."
        )
    else:
        summary = (
            f"6-dimension data quality score: {overall:.1f}/100. "
            "All dimensions healthy; data appears ready for modeling."
        )

    return {
        "overall_score": overall,
        "dimensions": dims,
        "summary": summary,
    }
