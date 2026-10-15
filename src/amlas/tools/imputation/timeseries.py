"""TIME-SERIES / long-lineage imputation (master plan §6, spec sheet 2.3).

``impute_timeseries`` handles Amlas's 68-year sensor/maintenance gaps via:

  * ``linear`` / ``spline`` / ``time`` — pandas ``interpolate`` (time-weighted
    when ``time`` and a datetime index),
  * ``ffill`` / ``bfill``             — forward / backward fill for
    slowly-changing attributes,
  * ``seasonal``                      — statsmodels ``seasonal_decompose`` to
    remove a periodic component, interpolate the remainder, then recompose
    (falls back to linear with a warning when <2 full periods are available).

statsmodels is imported lazily so the package imports without the ``data`` extra.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from strands import tool

from ...errors import tool_error_boundary
from ._common import imputed_output_path, read_csv_or_error

TIMESERIES_METHODS: tuple[str, ...] = ("linear", "spline", "time", "ffill", "bfill", "seasonal")


def _prepare_time_index(df: pd.DataFrame, time_col: str) -> tuple[pd.DataFrame | None, str | None]:
    """Sort by + set a parsed datetime index; return (frame, error_message)."""
    if time_col not in df.columns:
        return None, f"time_col '{time_col}' not found"
    try:
        parsed = pd.to_datetime(df[time_col], errors="raise")
    except Exception as e:
        return None, f"time_col '{time_col}' is not datetime-parseable: {e}"
    out = df.copy()
    out[time_col] = parsed
    out = out.sort_values(time_col).reset_index(drop=True)
    return out, None


def impute_timeseries_frame(
    df: pd.DataFrame,
    columns: list[str],
    method: str,
    time_col: str,
) -> tuple[pd.DataFrame, list[str], int]:
    """Time-series impute ``columns``; return (frame, warnings, n_filled)."""
    method = str(method).strip().lower()
    prepared, err = _prepare_time_index(df, time_col)
    if err is not None:
        raise ValueError(err)
    assert prepared is not None
    out = prepared
    warnings: list[str] = []

    targets = [c for c in columns if c in out.columns and pd.api.types.is_numeric_dtype(out[c])]
    for c in columns:
        if c not in out.columns:
            warnings.append(f"column '{c}' not found — skipped")
        elif not pd.api.types.is_numeric_dtype(out[c]):
            warnings.append(f"column '{c}' is not numeric — skipped")
    if not targets:
        return out, warnings, 0

    n_before = int(out[targets].isnull().sum().sum())

    indexed = out.set_index(time_col)
    for col in targets:
        indexed[col] = _interpolate_column(indexed[col], method, warnings, col)
    out = indexed.reset_index()

    # residual edge NaNs (e.g. leading gap for ffill) → both-direction fill
    for col in targets:
        if out[col].isnull().any():
            out[col] = out[col].ffill().bfill()

    n_after = int(out[targets].isnull().sum().sum())
    return out, warnings, n_before - n_after


def _interpolate_column(series: pd.Series, method: str, warnings: list[str], col: str) -> pd.Series:
    if method in ("ffill", "bfill"):
        return series.ffill() if method == "ffill" else series.bfill()
    if method == "seasonal":
        return _seasonal_fill(series, warnings, col)
    if method == "time":
        return series.interpolate(method="time", limit_direction="both")
    if method == "spline":
        non_null = series.notna().sum()
        order = 3 if non_null > 4 else 1
        try:
            return series.interpolate(method="spline", order=order, limit_direction="both")
        except Exception:
            warnings.append(f"column '{col}' spline failed — fell back to linear")
            return series.interpolate(method="linear", limit_direction="both")
    # default: linear
    return series.interpolate(method="linear", limit_direction="both")


def _seasonal_fill(series: pd.Series, warnings: list[str], col: str) -> pd.Series:
    """Seasonal-decompose → interpolate remainder → recompose (fallback: linear)."""
    n = len(series)
    # infer a period: try the index frequency, else a small default
    period = 0
    pd.infer_freq(series.index) if isinstance(series.index, pd.DatetimeIndex) else None
    if n >= 24:
        period = 12
    elif n >= 8:
        period = 4
    if period < 2 or n < 2 * period:
        warnings.append(f"column '{col}' seasonal needs >=2 periods — fell back to linear")
        return series.interpolate(method="linear", limit_direction="both")

    try:
        from statsmodels.tsa.seasonal import seasonal_decompose

        # decompose on a linearly-interpolated copy to obtain a seasonal template
        base = series.interpolate(method="linear", limit_direction="both")
        result = seasonal_decompose(base, period=period, model="additive", extrapolate_trend="freq")
        seasonal = result.seasonal
        deseasonalized = series - seasonal
        filled = deseasonalized.interpolate(method="linear", limit_direction="both") + seasonal
        return filled
    except Exception as e:
        warnings.append(f"column '{col}' seasonal_decompose failed ({e}) — fell back to linear")
        return series.interpolate(method="linear", limit_direction="both")


@tool
@tool_error_boundary(source="tools.imputation.timeseries")
def impute_timeseries(
    data_path: str,
    columns: list[str],
    method: str,
    time_col: str,
) -> dict[str, Any]:
    """Impute a time-ordered numeric column via interpolation / fill / seasonal.

    Args:
        data_path: Filesystem path to a CSV file.
        columns: Numeric time-series columns to impute.
        method: One of ``linear``/``spline``/``time``/``ffill``/``bfill``/``seasonal``.
        time_col: Name of the datetime column used to order the series.

    Returns:
        ``{"imputed_path","method","n_filled","columns","warnings","time_col"}``
        or ``{"success": False, "error": ...}``.
    """
    df, err = read_csv_or_error(data_path)
    if err is not None:
        return err
    assert df is not None

    m = str(method).strip().lower()
    if m not in TIMESERIES_METHODS:
        return {
            "success": False,
            "error": f"unknown time-series method '{method}'; choose from {list(TIMESERIES_METHODS)}",
        }

    try:
        out, warnings, n_filled = impute_timeseries_frame(df, columns, m, time_col)
    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"time-series imputation failed: {e}"}

    out_path = imputed_output_path(data_path, f"ts_{m}")
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
        "time_col": time_col,
    }
