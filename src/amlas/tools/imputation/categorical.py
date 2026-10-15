"""CATEGORICAL / TEXT ("verbiage") imputation (master plan §6, spec sheet 2.3).

``impute_categorical`` selects between:

  * ``mode``       — most-frequent-value fill (baseline; MCAR-safe),
  * ``classifier`` — a sklearn classifier (RandomForest) predicting the missing
    category from the other columns (MAR-aware; beats mode when the category is
    predictable from row context).

The classical paths are fully implemented here; the *LLM-extrapolation* path for
semantically-rich text gaps lives in :mod:`llm_extrapolate` (injectable, mockable).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from strands import tool

from ...errors import tool_error_boundary
from ._common import imputed_output_path, read_csv_or_error

CATEGORICAL_METHODS: tuple[str, ...] = ("mode", "classifier")


def impute_categorical_frame(
    df: pd.DataFrame,
    columns: list[str],
    method: str,
) -> tuple[pd.DataFrame, list[str], int]:
    """Impute categorical ``columns``; return (frame, warnings, n_filled)."""
    out = df.copy()
    warnings: list[str] = []
    method = str(method).strip().lower()

    targets: list[str] = []
    for col in columns:
        if col not in out.columns:
            warnings.append(f"column '{col}' not found — skipped")
            continue
        if out[col].isnull().all():
            warnings.append(f"column '{col}' is entirely null — cannot impute, flagged")
            continue
        if not out[col].isnull().any():
            continue
        targets.append(col)

    if not targets:
        return out, warnings, 0

    n_before = int(out[targets].isnull().sum().sum())

    if method == "mode":
        for col in targets:
            mode_vals = out[col].mode(dropna=True)
            fill = mode_vals.iloc[0] if len(mode_vals) else ""
            out[col] = out[col].fillna(fill)
    elif method == "classifier":
        for col in targets:
            out[col] = _classifier_fill(out, col, warnings)
    else:
        raise ValueError(f"unknown categorical method: {method}")

    n_after = int(out[targets].isnull().sum().sum())
    return out, warnings, n_before - n_after


def _classifier_fill(df: pd.DataFrame, col: str, warnings: list[str]) -> pd.Series:
    """Predict the missing categories of ``col`` from the other columns (RF)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder

    series = df[col]
    miss = series.isnull()
    known = ~miss

    # build a simple numeric feature matrix from the OTHER columns
    feature_cols = [c for c in df.columns if c != col]
    feats = df[feature_cols].copy()
    encoded = pd.DataFrame(index=df.index)
    for c in feature_cols:
        s = feats[c]
        if pd.api.types.is_numeric_dtype(s):
            encoded[c] = s.fillna(s.median())
        else:
            codes = s.astype("object").fillna("__nan__").astype(str)
            encoded[c] = LabelEncoder().fit_transform(codes)

    if encoded.shape[1] == 0 or known.sum() < 3 or series[known].nunique() < 2:
        # not enough signal → fall back to mode
        warnings.append(f"column '{col}' classifier fell back to mode (insufficient signal)")
        mode_vals = series.mode(dropna=True)
        fill = mode_vals.iloc[0] if len(mode_vals) else ""
        return series.fillna(fill)

    y_enc = LabelEncoder()
    y_known = y_enc.fit_transform(series[known].astype(str))
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1)
    clf.fit(encoded[known.to_numpy()], y_known)
    preds = clf.predict(encoded[miss.to_numpy()])
    pred_labels = y_enc.inverse_transform(preds)

    result = series.copy().astype("object")
    result.loc[miss] = pred_labels
    return result


@tool
@tool_error_boundary(source="tools.imputation.categorical")
def impute_categorical(
    data_path: str,
    columns: list[str] | None = None,
    method: str = "mode",
) -> dict[str, Any]:
    """Impute missing categorical / discrete-text values (master plan §6).

    Args:
        data_path: Filesystem path to a CSV file.
        columns: Categorical columns to impute (default: all non-numeric columns
            that contain at least one missing value).
        method: ``mode`` (most-frequent) or ``classifier`` (RF predicting the
            category from other columns; MAR-aware).

    Returns:
        ``{"imputed_path","method","n_filled","columns","warnings"}`` or
        ``{"success": False, "error": ...}``.
    """
    df, err = read_csv_or_error(data_path)
    if err is not None:
        return err
    assert df is not None

    m = str(method).strip().lower()
    if m not in CATEGORICAL_METHODS:
        return {
            "success": False,
            "error": f"unknown categorical method '{method}'; choose from {list(CATEGORICAL_METHODS)}",
        }

    if columns is None:
        non_numeric = df.select_dtypes(exclude=["number"]).columns.tolist()
        columns = [c for c in non_numeric if df[c].isnull().any()]

    try:
        out, warnings, n_filled = impute_categorical_frame(df, columns, m)
    except Exception as e:
        return {"success": False, "error": f"categorical imputation failed: {e}"}

    out_path = imputed_output_path(data_path, f"cat_{m}")
    try:
        out.to_csv(out_path, index=False)
    except Exception as e:
        return {"success": False, "error": f"failed to write imputed CSV: {e}"}

    return {
        "imputed_path": str(out_path),
        "method": m,
        "n_filled": int(n_filled),
        "columns": [c for c in columns if c in out.columns],
        "warnings": warnings,
    }
