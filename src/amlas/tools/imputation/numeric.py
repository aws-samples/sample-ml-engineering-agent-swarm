"""Multi-strategy NUMERIC imputation (master plan §6, spec sheet 2.3).

``impute_numeric`` selects among:

  * ``mean`` / ``median`` / ``mode`` — sklearn ``SimpleImputer`` (univariate,
    MCAR-only-unbiased fallback; ``mean`` distorts variance — research KB),
  * ``knn``                          — ``KNNImputer`` (multivariate, MAR-aware),
  * ``iterative``                    — ``IterativeImputer`` (MICE, BayesianRidge),
  * ``missforest``                   — ``IterativeImputer(RandomForestRegressor)``
    (handles non-linearities + mixed/correlated columns),
  * ``matrix_factorization``         — in-house iterative low-rank SVD fill for
    wide, highly-correlated sensor panels.

The numeric core (:func:`impute_numeric_frame`) is a plain function reused by the
evaluation harness and the engine; the ``@tool`` wrapper handles path I/O,
warnings, and the JSON-dict / fail-soft contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from strands import tool

from ...errors import tool_error_boundary
from ._common import imputed_output_path, read_csv_or_error

NUMERIC_METHODS: tuple[str, ...] = (
    "mean",
    "median",
    "mode",
    "knn",
    "iterative",
    "missforest",
    "matrix_factorization",
)

# aliases the agent / catalog may emit → canonical method name
_METHOD_ALIASES = {
    "most_frequent": "mode",
    "mice": "iterative",
    "iterativeimputer": "iterative",
    "rf": "missforest",
    "random_forest": "missforest",
    "matrix_fact": "matrix_factorization",
    "matrixfactorization": "matrix_factorization",
    "softimpute": "matrix_factorization",
    "iterative_svd": "matrix_factorization",
}


def canonical_numeric_method(method: str) -> str:
    """Map an alias to a canonical numeric method name (identity if already canonical)."""
    m = str(method).strip().lower()
    return _METHOD_ALIASES.get(m, m)


def _simple_impute(values: np.ndarray, strategy: str) -> np.ndarray:
    from sklearn.impute import SimpleImputer

    imp = SimpleImputer(strategy=strategy)
    return np.asarray(imp.fit_transform(values.reshape(-1, 1)).ravel())


def _matrix_factorization_fill(
    matrix: np.ndarray, rank: int = 2, n_iters: int = 50, tol: float = 1e-4
) -> np.ndarray:
    """In-house IterativeSVD / SoftImpute-style low-rank fill.

    Initialise NaNs with column means, then repeatedly take a rank-``r``
    truncated-SVD reconstruction and overwrite the missing cells with the
    reconstruction until convergence. Captures latent structure across
    correlated columns without a new dependency.
    """
    from sklearn.decomposition import TruncatedSVD

    X = matrix.astype(float).copy()
    mask = np.isnan(X)
    if not mask.any():
        return X

    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
    inds = np.where(mask)
    X[inds] = np.take(col_means, inds[1])

    eff_rank = max(1, min(rank, min(X.shape) - 1)) if min(X.shape) > 1 else 1
    prev = X.copy()
    for _ in range(n_iters):
        try:
            svd = TruncatedSVD(n_components=eff_rank, random_state=0)
            reduced = svd.fit_transform(X)
            recon = reduced @ svd.components_
        except Exception:
            break
        X[mask] = recon[mask]
        if np.linalg.norm(X - prev) / (np.linalg.norm(prev) + 1e-12) < tol:
            break
        prev = X.copy()
    return X


def impute_numeric_frame(
    df: pd.DataFrame,
    columns: list[str],
    method: str,
    params: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[str], int]:
    """Impute ``columns`` of ``df`` in place (on a copy); return (frame, warnings, n_filled).

    Multivariate methods (``knn``/``iterative``/``missforest``/
    ``matrix_factorization``) use ALL numeric columns as context but only write
    back the requested target ``columns`` so untouched columns are byte-identical.
    """
    params = params or {}
    out = df.copy()
    warnings: list[str] = []

    numeric_cols = out.select_dtypes(include=["number"]).columns.tolist()
    targets: list[str] = []
    for col in columns:
        if col not in out.columns:
            warnings.append(f"column '{col}' not found — skipped")
            continue
        if col not in numeric_cols:
            warnings.append(f"column '{col}' is not numeric — skipped")
            continue
        if out[col].isnull().all():
            warnings.append(f"column '{col}' is entirely null — cannot impute, flagged")
            continue
        targets.append(col)

    if not targets:
        return out, warnings, 0

    n_before = int(out[targets].isnull().sum().sum())
    method = canonical_numeric_method(method)

    if method in ("mean", "median", "mode"):
        strategy = "most_frequent" if method == "mode" else method
        for col in targets:
            out[col] = _simple_impute(out[col].to_numpy(dtype=float), strategy)

    elif method in ("knn", "iterative", "missforest", "matrix_factorization"):
        context = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
        filled = _multivariate_fill(context.to_numpy(), method, params)
        filled_df = pd.DataFrame(filled, columns=numeric_cols, index=out.index)
        for col in targets:
            # only overwrite the originally-missing cells of the target column
            miss = out[col].isnull()
            out.loc[miss, col] = filled_df.loc[miss, col]
        # any residual NaN (e.g. all-context-missing rows) → median safety net
        for col in targets:
            if out[col].isnull().any():
                out[col] = _simple_impute(out[col].to_numpy(dtype=float), "median")
    else:  # pragma: no cover — guarded by the @tool wrapper
        raise ValueError(f"unknown numeric method: {method}")

    n_after = int(out[targets].isnull().sum().sum())
    return out, warnings, n_before - n_after


def _multivariate_fill(matrix: np.ndarray, method: str, params: dict[str, Any]) -> np.ndarray:
    if method == "knn":
        from sklearn.impute import KNNImputer

        n_neighbors = int(params.get("n_neighbors", 5))
        return np.asarray(KNNImputer(n_neighbors=n_neighbors).fit_transform(matrix))
    if method == "iterative":
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer

        return np.asarray(
            IterativeImputer(random_state=0, max_iter=int(params.get("max_iter", 10))).fit_transform(matrix)
        )
    if method == "missforest":
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        est = RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 50)),
            random_state=0,
            n_jobs=1,
        )
        return np.asarray(
            IterativeImputer(estimator=est, random_state=0, max_iter=int(params.get("max_iter", 5))).fit_transform(
                matrix
            )
        )
    if method == "matrix_factorization":
        return _matrix_factorization_fill(matrix, rank=int(params.get("rank", 2)))
    raise ValueError(f"unknown multivariate method: {method}")  # pragma: no cover


@tool
@tool_error_boundary(source="tools.imputation.numeric")
def impute_numeric(
    data_path: str,
    columns: list[str] | None = None,
    method: str = "median",
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Impute missing numeric values with a selectable strategy (master plan §6).

    Args:
        data_path: Filesystem path to a CSV file.
        columns: Numeric columns to impute (default: all numeric columns with NaN).
        method: One of ``mean``/``median``/``mode``/``knn``/``iterative``/
            ``missforest``/``matrix_factorization`` (aliases accepted, e.g.
            ``mice``→iterative, ``rf``→missforest).
        params: Optional per-method kwargs (``n_neighbors``, ``max_iter``,
            ``n_estimators``, ``rank``).

    Returns:
        ``{"imputed_path","method","n_filled","columns","warnings"}`` on success,
        or ``{"success": False, "error": ...}``.
    """
    df, err = read_csv_or_error(data_path)
    if err is not None:
        return err
    assert df is not None

    canon = canonical_numeric_method(method)
    if canon not in NUMERIC_METHODS:
        return {
            "success": False,
            "error": f"unknown numeric method '{method}'; choose from {list(NUMERIC_METHODS)}",
        }

    from ._common import numeric_columns_with_missing

    target_cols = columns if columns is not None else numeric_columns_with_missing(df)

    try:
        out, warnings, n_filled = impute_numeric_frame(df, target_cols, canon, params)
    except Exception as e:
        return {"success": False, "error": f"imputation failed: {e}"}

    out_path = imputed_output_path(data_path, canon)
    try:
        out.to_csv(out_path, index=False)
    except Exception as e:
        return {"success": False, "error": f"failed to write imputed CSV: {e}"}

    imputed_targets = [c for c in target_cols if c in out.columns and c in df.select_dtypes(include=["number"]).columns]
    return {
        "imputed_path": str(out_path),
        "method": canon,
        "n_filled": int(n_filled),
        "columns": imputed_targets,
        "warnings": warnings,
    }
