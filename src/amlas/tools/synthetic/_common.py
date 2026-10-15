"""Shared helpers for the synthetic-data tools (paths, SDV metadata, fidelity math).

Kept dependency-light at import time: pandas/numpy/scipy only. The heavy optional
deps (``sdv``/``sdmetrics``/``imblearn``) are imported lazily inside the tool
functions so ``tools.synthetic`` imports cleanly without the ``data`` extra
(matching the ``tools.imputation`` convention). Reuses the Phase-7
``tools.imputation._common`` I/O + KS helpers so the synthetic engine and the
imputation engine share one statistical core.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Reuse the Phase-7 I/O + distribution helpers (single statistical core). The
# redundant ``X as X`` aliases are the PEP 484 explicit-re-export form so mypy
# strict (``no_implicit_reexport``) lets the synthetic tools import these shared
# helpers from this module.
from ..imputation._common import ks_pvalue as ks_pvalue
from ..imputation._common import psi as psi
from ..imputation._common import read_csv_or_error as read_csv_or_error

# SDV single-table synthesizers the agent may pick among (research KB §1).
SDV_GENERATORS: tuple[str, ...] = ("gaussian_copula", "ctgan", "tvae")
# imbalanced-learn augmentation strategies (research KB §1, SMOTE family).
AUGMENT_STRATEGIES: tuple[str, ...] = ("smote", "smotenc", "adasyn", "random")

_GENERATOR_ALIASES = {
    "gaussiancopula": "gaussian_copula",
    "copula": "gaussian_copula",
    "gc": "gaussian_copula",
    "ctgan": "ctgan",
    "gan": "ctgan",
    "tvae": "tvae",
    "vae": "tvae",
}


def canonical_generator(name: str) -> str:
    """Map a generator alias to a canonical SDV generator name (identity if canonical)."""
    n = str(name).strip().lower()
    return _GENERATOR_ALIASES.get(n, n)


def synthetic_output_path(data_path: str, generator: str, suffix: str = "synthetic") -> Path:
    """Deterministic write-back path next to the input CSV.

    ``/x/data.csv`` + generator ``gaussian_copula`` →
    ``/x/data.synthetic.gaussian_copula.csv``.
    """
    p = Path(data_path)
    return p.with_name(f"{p.stem}.{suffix}.{generator}.csv")


def build_sdv_metadata(df: pd.DataFrame, table_name: str = "table") -> Any:
    """Detect SDV ``Metadata`` from a frame (lazy import of ``sdv``)."""
    from sdv.metadata import Metadata

    return Metadata.detect_from_dataframe(df, table_name=table_name)


def single_table_metadata_dict(metadata: Any) -> dict[str, Any]:
    """Extract a single-table metadata dict (``{"columns": {...}}``) for sdmetrics.

    ``sdv.metadata.Metadata`` is multi-table; ``sdmetrics.reports.single_table``
    wants the inner per-table dict. Handles both the multi-table
    (``{"tables": {name: {...}}}``) and already-single-table shapes.
    """
    d = metadata.to_dict() if hasattr(metadata, "to_dict") else dict(metadata)
    if d.get("tables"):
        return next(iter(d["tables"].values()))
    return d


# ─── fidelity statistics (scipy-only, no sdmetrics needed) ───────────────────


def ks_complement(real: np.ndarray, synth: np.ndarray) -> float:
    """1 − KS statistic between two numeric samples (1.0 = identical shape, 0 = disjoint)."""
    from scipy import stats

    a = np.asarray(real, dtype=float)
    b = np.asarray(synth, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return 1.0
    try:
        return float(max(0.0, 1.0 - stats.ks_2samp(a, b).statistic))
    except Exception:
        return 1.0


def tv_complement(real: pd.Series, synth: pd.Series) -> float:
    """1 − total-variation distance between two categorical frequency vectors."""
    r = real.astype(str).value_counts(normalize=True)
    s = synth.astype(str).value_counts(normalize=True)
    cats = set(r.index) | set(s.index)
    tv = 0.5 * sum(abs(float(r.get(c, 0.0)) - float(s.get(c, 0.0))) for c in cats)
    return float(max(0.0, 1.0 - tv))


def column_shape_score(real: pd.DataFrame, synth: pd.DataFrame) -> tuple[float, float]:
    """Mean per-column shape fidelity (KSComplement numeric / TVComplement categorical).

    Returns ``(overall_score, numeric_ks_mean)`` where ``overall_score`` is the
    mean column-shape fidelity across shared columns (the scipy fidelity proxy
    for SDV's Column-Shapes quality) and ``numeric_ks_mean`` is the mean KS
    statistic over numeric columns (lower = closer).
    """
    shared = [c for c in real.columns if c in synth.columns]
    scores: list[float] = []
    ks_stats: list[float] = []
    for c in shared:
        if pd.api.types.is_numeric_dtype(real[c]) and pd.api.types.is_numeric_dtype(synth[c]):
            comp = ks_complement(real[c].to_numpy(), synth[c].to_numpy())
            scores.append(comp)
            ks_stats.append(1.0 - comp)
        else:
            scores.append(tv_complement(real[c], synth[c]))
    overall = float(np.mean(scores)) if scores else 1.0
    ks_mean = float(np.mean(ks_stats)) if ks_stats else 0.0
    return overall, ks_mean


def correlation_drift_frobenius(real: pd.DataFrame, synth: pd.DataFrame) -> float:
    """Frobenius norm of the numeric correlation-matrix difference (research KB single metric).

    ``np.linalg.norm(real.corr - synth.corr, 'fro')`` over the shared numeric
    columns. 0 = correlation structure perfectly preserved; the research KB
    flags drift > ~0.5 as a fidelity failure.
    """
    num = [
        c
        for c in real.columns
        if c in synth.columns
        and pd.api.types.is_numeric_dtype(real[c])
        and pd.api.types.is_numeric_dtype(synth[c])
    ]
    if len(num) < 2:
        return 0.0
    rc = real[num].corr().fillna(0.0).to_numpy()
    sc = synth[num].corr().fillna(0.0).to_numpy()
    return float(np.linalg.norm(rc - sc, "fro"))
