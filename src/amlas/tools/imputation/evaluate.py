"""Imputation self-evaluation + steered write-back (master plan §6, spec sheet 2.4).

``evaluate_imputation`` is the **"prove it works" mask-and-score harness**: it
takes the *original* (complete-enough) data, masks a fraction of KNOWN cells
under a chosen mechanism (MCAR/MAR/MNAR), re-imputes with each candidate method,
and scores recovery against the held-out truth (RMSE/MAE/R² numeric;
accuracy/macro-F1 categorical) plus distribution preservation (KS / PSI) so the
mean-imputation variance-collapse failure mode is caught.

``apply_imputation_plan`` executes a per-column ``{method,params}`` plan, but
FIRST enforces the HARD steering constraints (``forbid_methods`` / ``pin_methods``)
deterministically — the agent can propose a plan but cannot violate a pin/forbid.
It writes the imputed dataset back to a local artifact and emits an
``imputation_report`` manifest (per-column method, n_filled, recovery scores).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from strands import tool

from ...errors import tool_error_boundary
from ..data_quality import diagnose_missingness  # noqa: F401  (reused by engine; kept importable)
from ._common import (
    imputed_output_path,
    ks_pvalue,
    mae,
    psi,
    read_csv_or_error,
    rmse,
)
from .categorical import CATEGORICAL_METHODS, impute_categorical_frame
from .numeric import NUMERIC_METHODS, canonical_numeric_method, impute_numeric_frame

# ─── mask-and-score self-evaluation ──────────────────────────────────────────


def _make_mask(
    df: pd.DataFrame, col: str, mechanism: str, frac: float, rng: np.random.Generator
) -> np.ndarray:
    """Boolean mask of cells to artificially blank under ``mechanism`` (on known cells)."""
    known = df[col].notna().to_numpy()
    known_idx = np.where(known)[0]
    n_mask = max(1, round(len(known_idx) * frac))
    n_mask = min(n_mask, len(known_idx))

    mech = mechanism.upper()
    if mech == "MNAR":
        # value-dependent: preferentially mask the largest known values
        vals = df[col].to_numpy(dtype=float)
        order = known_idx[np.argsort(-vals[known_idx])]
        chosen = order[:n_mask]
    elif mech == "MAR":
        # depends on another observed column (proxy: rank by a second numeric col)
        others = [c for c in df.select_dtypes(include=["number"]).columns if c != col]
        if others:
            proxy = df[others[0]].to_numpy(dtype=float)
            order = known_idx[np.argsort(-proxy[known_idx])]
            chosen = order[:n_mask]
        else:
            chosen = rng.choice(known_idx, size=n_mask, replace=False)
    else:  # MCAR
        chosen = rng.choice(known_idx, size=n_mask, replace=False)

    mask = np.zeros(len(df), dtype=bool)
    mask[chosen] = True
    return mask


def evaluate_methods(
    df: pd.DataFrame,
    column: str,
    methods: list[str],
    mechanism: str = "MCAR",
    frac: float = 0.1,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Score each candidate method's recovery of masked-known cells for ``column``.

    Returns ``{method: {metric, value, ks_pvalue, psi, distribution_ok}}`` where
    ``value`` is RMSE (numeric) or 1-accuracy error (categorical, lower better).
    """
    rng = np.random.default_rng(seed)
    is_numeric = pd.api.types.is_numeric_dtype(df[column])
    mask = _make_mask(df, column, mechanism, frac, rng)
    truth = df.loc[mask, column].copy()

    results: dict[str, dict[str, Any]] = {}
    observed = df.loc[~mask & df[column].notna(), column]

    for method in methods:
        trial = df.copy()
        trial.loc[mask, column] = np.nan
        if is_numeric:
            canon = canonical_numeric_method(method)
            filled, _w, _n = impute_numeric_frame(trial, [column], canon, None)
            pred = filled.loc[mask, column].to_numpy(dtype=float)
            true = truth.to_numpy(dtype=float)
            value = rmse(true, pred)
            ks_p = ks_pvalue(observed.to_numpy(dtype=float), filled[column].to_numpy(dtype=float))
            psi_v = psi(observed.to_numpy(dtype=float), filled[column].to_numpy(dtype=float))
            results[method] = {
                "metric": "rmse",
                "value": round(value, 6),
                "mae": round(mae(true, pred), 6),
                "ks_pvalue": round(ks_p, 6),
                "psi": round(psi_v, 6),
                "distribution_ok": bool(ks_p > 0.05),
            }
        else:
            filled, _w, _n = impute_categorical_frame(trial, [column], method)
            pred = filled.loc[mask, column].astype(str).to_numpy()
            true = truth.astype(str).to_numpy()
            acc = float(np.mean(pred == true)) if len(true) else 1.0
            results[method] = {
                "metric": "accuracy",
                "value": round(1.0 - acc, 6),  # error: lower is better (uniform with rmse)
                "accuracy": round(acc, 6),
                "macro_f1": round(_macro_f1(true, pred), 6),
                "ks_pvalue": None,
                "psi": None,
                "distribution_ok": True,
            }
    return results


def _macro_f1(true: np.ndarray, pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    try:
        return float(f1_score(true, pred, average="macro", zero_division=0))
    except Exception:
        return 0.0


@tool
@tool_error_boundary(source="tools.imputation.evaluate")
def evaluate_imputation(
    original_path: str,
    columns: list[str],
    methods: list[str] | None = None,
    mechanism: str = "MCAR",
    frac: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    """Mask-and-score self-evaluation: rank imputation methods by recovery.

    Masks ``frac`` of the KNOWN cells of each column under ``mechanism``,
    re-imputes with each candidate ``method``, and scores recovery (RMSE numeric /
    accuracy+macro-F1 categorical) plus distribution preservation (KS/PSI).

    Args:
        original_path: CSV with enough complete cells to mask.
        columns: Columns to evaluate.
        methods: Candidate methods (default: a sensible set per dtype).
        mechanism: ``MCAR``/``MAR``/``MNAR`` masking pattern.
        frac: Fraction of known cells to blank (0..1).
        seed: RNG seed (reproducible masking).

    Returns:
        ``{"per_column":{col:{"methods":{...},"best_method_hint":str}}, "overall":{...}}``
        or ``{"success": False, "error": ...}``.
    """
    df, err = read_csv_or_error(original_path)
    if err is not None:
        return err
    assert df is not None

    per_column: dict[str, Any] = {}
    best_overall: list[str] = []
    for col in columns:
        if col not in df.columns:
            per_column[col] = {"skipped": f"column '{col}' not found"}
            continue
        non_null = int(df[col].notna().sum())
        if non_null < 20:
            per_column[col] = {"skipped": f"only {non_null} non-null cells (<20) — scoring skipped"}
            continue

        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        cand = methods or (
            ["median", "knn", "iterative", "missforest"] if is_numeric else list(CATEGORICAL_METHODS)
        )
        cand = [m for m in cand if (canonical_numeric_method(m) in NUMERIC_METHODS) or (m in CATEGORICAL_METHODS)]
        scored = evaluate_methods(df, col, cand, mechanism, frac, seed)
        if not scored:
            per_column[col] = {"skipped": "no valid candidate methods"}
            continue
        best = min(scored.items(), key=lambda kv: kv[1]["value"])[0]
        per_column[col] = {"methods": scored, "best_method_hint": best, "mechanism": mechanism}
        best_overall.append(best)

    return {
        "per_column": per_column,
        "overall": {"mechanism": mechanism, "frac": frac, "evaluated_columns": len(per_column)},
    }


# ─── steered write-back (apply_imputation_plan) ──────────────────────────────


@tool
@tool_error_boundary(source="tools.imputation.evaluate")
def apply_imputation_plan(
    data_path: str,
    plan: dict[str, Any],
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a per-column imputation plan, enforcing HARD steering, and write back.

    ``plan`` = ``{column: {"method": str, "params": dict|None, "time_col": str|None}}``.
    Before executing, HARD steering from ``profile`` is enforced deterministically:
    a method on ``forbid_methods`` is rejected; a column/dtype ``pin_methods`` entry
    OVERRIDES the proposed method. Violations are reported in ``steering_violations``
    and rejected (not silently run).

    Args:
        data_path: CSV to impute.
        plan: Per-column method plan (see above).
        profile: Optional ``SteeringProfile.as_profile_dict()`` (forbid/pin/etc.).

    Returns:
        ``{"imputed_path","plan_applied","imputation_report","steering_violations"}``
        or ``{"success": False, "error": ...}``.
    """
    df, err = read_csv_or_error(data_path)
    if err is not None:
        return err
    assert df is not None

    forbid = {str(m).strip().lower() for m in (profile or {}).get("forbid_methods", [])}
    pins = (profile or {}).get("pin_methods", {}) or {}

    steering_violations: list[dict[str, Any]] = []
    plan_applied: dict[str, Any] = {}
    report_columns: list[dict[str, Any]] = []
    out = df.copy()

    for col, spec in plan.items():
        if col not in df.columns:
            steering_violations.append({"column": col, "reason": "column not found"})
            continue
        proposed = str(spec.get("method", "median"))
        params = spec.get("params")
        time_col = spec.get("time_col")

        dtype_str = str(df[col].dtype)
        # HARD pin override: a column pin beats a dtype pin beats the proposal.
        pinned = pins.get(col) or pins.get(dtype_str) or pins.get(_coarse_dtype(dtype_str))
        effective = pinned if pinned is not None else proposed

        # HARD forbid: reject (do not run) if the EFFECTIVE method is forbidden.
        if str(effective).strip().lower() in forbid:
            steering_violations.append(
                {"column": col, "reason": f"method '{effective}' is forbidden by steering"}
            )
            continue

        if pinned is not None and str(pinned).strip().lower() != proposed.strip().lower():
            steering_violations.append(
                {"column": col, "reason": f"pin override: '{proposed}' -> '{pinned}'", "enforced": True}
            )

        out, n_filled, method_kind = _apply_one(out, col, effective, params, time_col)
        plan_applied[col] = {"method": effective, "kind": method_kind, "n_filled": n_filled}
        report_columns.append(
            {"column": col, "method": effective, "kind": method_kind, "n_filled": n_filled}
        )

    # if every requested column violated steering, fail soft so the caller retries
    if not plan_applied and steering_violations:
        return {
            "success": False,
            "error": "all planned columns rejected by hard steering",
            "steering_violations": steering_violations,
        }

    out_path = imputed_output_path(data_path, "plan")
    try:
        out.to_csv(out_path, index=False)
    except Exception as e:
        return {"success": False, "error": f"failed to write imputed CSV: {e}"}

    report = {
        "source_path": data_path,
        "imputed_path": str(out_path),
        "columns": report_columns,
        "total_filled": int(sum(c["n_filled"] for c in report_columns)),
        "steering_applied": {"forbid_methods": sorted(forbid), "pin_methods": pins},
    }
    return {
        "imputed_path": str(out_path),
        "plan_applied": plan_applied,
        "imputation_report": report,
        "steering_violations": steering_violations,
    }


def _coarse_dtype(dtype: str) -> str:
    d = dtype.lower()
    if d.startswith("float"):
        return "float"
    if d.startswith("int") or d.startswith("uint"):
        return "int"
    if d.startswith("datetime") or d.startswith("date"):
        return "datetime"
    if d.startswith("bool"):
        return "bool"
    return d


def _apply_one(
    df: pd.DataFrame,
    col: str,
    method: str,
    params: dict[str, Any] | None,
    time_col: str | None,
) -> tuple[pd.DataFrame, int, str]:
    """Dispatch a single column to numeric / categorical / timeseries impute."""
    from .timeseries import TIMESERIES_METHODS, impute_timeseries_frame

    m = str(method).strip().lower()
    if m in TIMESERIES_METHODS and time_col is not None:
        out, _w, n = impute_timeseries_frame(df, [col], m, time_col)
        return out, n, "timeseries"
    if pd.api.types.is_numeric_dtype(df[col]):
        canon = canonical_numeric_method(m)
        out, _w, n = impute_numeric_frame(df, [col], canon, params)
        return out, n, "numeric"
    out, _w, n = impute_categorical_frame(df, [col], m if m in CATEGORICAL_METHODS else "mode")
    return out, n, "categorical"
