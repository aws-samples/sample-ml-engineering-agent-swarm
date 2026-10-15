"""Shared helpers for the imputation tools (path handling, frame I/O, metrics).

Kept small and dependency-light (pandas/numpy/sklearn only) so the whole
``tools.imputation`` package imports without the optional ``data`` extra; the
single statsmodels dependency is imported lazily inside ``timeseries``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def imputed_output_path(data_path: str, method: str, suffix: str = "imputed") -> Path:
    """Return a deterministic write-back path next to the input CSV.

    ``/x/data.csv`` + method ``knn`` → ``/x/data.imputed.knn.csv``. Writing next
    to the input keeps the medallion silver→silver story local for this phase
    (cloud staging is a later phase).
    """
    p = Path(data_path)
    stem = p.stem
    return p.with_name(f"{stem}.{suffix}.{method}.csv")


def read_csv_or_error(data_path: str) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    """Load a CSV, returning ``(df, None)`` on success or ``(None, error_dict)``."""
    p = Path(data_path)
    if not p.exists():
        return None, {"success": False, "error": f"File not found: {data_path}"}
    try:
        return pd.read_csv(p), None
    except Exception as e:
        return None, {"success": False, "error": f"Failed to read CSV: {e}"}


def numeric_columns_with_missing(df: pd.DataFrame) -> list[str]:
    """Numeric columns that contain at least one NaN (default impute target set)."""
    num = df.select_dtypes(include=["number"])
    return [c for c in num.columns if df[c].isnull().any()]


def rmse(true: np.ndarray, pred: np.ndarray) -> float:
    """Root-mean-square error over aligned arrays."""
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def mae(true: np.ndarray, pred: np.ndarray) -> float:
    """Mean absolute error over aligned arrays."""
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.mean(np.abs(true - pred)))


def ks_pvalue(observed: np.ndarray, imputed_all: np.ndarray) -> float:
    """Two-sample KS p-value comparing observed vs full (post-impute) distribution.

    A small p-value means the imputed column distribution diverged from the
    originally-observed values — the classic mean-imputation variance-collapse
    failure mode the research KB warns about.
    """
    from scipy import stats

    a = np.asarray(observed, dtype=float)
    b = np.asarray(imputed_all, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return 1.0
    try:
        return float(stats.ks_2samp(a, b).pvalue)
    except Exception:
        return 1.0


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two numeric samples (lower = more stable)."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) < 2 or len(actual) < 2:
        return 0.0
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if len(edges) < 2:
        return 0.0
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_frac = np.clip(e_counts / max(e_counts.sum(), 1), 1e-6, None)
    a_frac = np.clip(a_counts / max(a_counts.sum(), 1), 1e-6, None)
    return float(np.sum((a_frac - e_frac) * np.log(a_frac / e_frac)))
