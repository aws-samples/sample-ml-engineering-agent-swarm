"""Multi-strategy imputation decision engine (master plan §6 decision logic).

``recommend_imputation_plan`` is the deterministic heart of the §6 engine: it
consumes the Phase-6 ``diagnose_missingness`` output (MCAR/MAR/MNAR per column)
+ a ``SteeringProfile`` to build a per-column candidate set by dtype, runs the
``evaluate_imputation`` mask-and-score harness to rank candidates, and selects the
best recovery method per column **honouring hard steering** (pins/forbids). The
agent (Imputation Strategist) may then add novel strategies up to
``novelty_budget``, but cannot override the hard constraints enforced here and in
``apply_imputation_plan``.

Decision logic (plan §6):
  1. diagnose_missingness → mechanism per column (MCAR/MAR/MNAR)
  2. per-column candidate set by dtype + mechanism:
       MCAR numeric → {median, knn}            (any imputer safe)
       MAR  numeric → {knn, iterative, missforest}  (multivariate)
       MNAR numeric → {missforest, iterative}  (+ caller adds indicator)
       datetime     → {linear, time, seasonal}
       categorical  → {mode, classifier}
  3. hard pin/forbid filtering (SteeringProfile)
  4. evaluate_imputation(mask-and-score) ranks candidates
  5. pick best recovery per column (pins win outright)
"""

from __future__ import annotations

from typing import Any

from strands import tool

from ...errors import tool_error_boundary
from ..data_quality.missingness import diagnose_missingness
from ._common import read_csv_or_error
from .categorical import CATEGORICAL_METHODS
from .evaluate import evaluate_methods

# candidate sets by (dtype-family, mechanism) — research KB §2 + plan §6
_NUMERIC_BY_MECHANISM = {
    "MCAR": ["median", "knn"],
    "MAR": ["knn", "iterative", "missforest"],
    "MNAR": ["missforest", "iterative"],
    "complete": [],
    "unusable": [],
}
_DATETIME_CANDIDATES = ["linear", "time", "seasonal"]


def _candidate_methods(dtype: str, mechanism: str) -> list[str]:
    """Return the dtype/mechanism-appropriate candidate method names."""
    d = dtype.lower()
    if d.startswith(("datetime", "date")):
        return list(_DATETIME_CANDIDATES)
    if d.startswith(("float", "int", "uint")):
        return list(_NUMERIC_BY_MECHANISM.get(mechanism.upper(), ["median", "knn", "missforest"]))
    return list(CATEGORICAL_METHODS)  # object/categorical


def _filter_by_steering(methods: list[str], column: str, dtype: str, profile: dict[str, Any] | None) -> list[str]:
    """Drop forbidden methods; a pin collapses the candidate set to the pinned method."""
    prof = profile or {}
    forbid = {str(m).strip().lower() for m in prof.get("forbid_methods", [])}
    pins = prof.get("pin_methods", {}) or {}

    pinned = pins.get(column) or pins.get(dtype) or pins.get(_coarse(dtype))
    if pinned is not None:
        # hard pin wins outright (still respect forbid: a forbidden pin is a config error)
        return [] if str(pinned).strip().lower() in forbid else [pinned]

    return [m for m in methods if str(m).strip().lower() not in forbid]


def _coarse(dtype: str) -> str:
    d = dtype.lower()
    if d.startswith("float"):
        return "float"
    if d.startswith("int") or d.startswith("uint"):
        return "int"
    if d.startswith("datetime") or d.startswith("date"):
        return "datetime"
    return d


@tool
@tool_error_boundary(source="tools.imputation.engine")
def recommend_imputation_plan(
    data_path: str,
    target: str | None = None,
    profile: dict[str, Any] | None = None,
    frac: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Recommend a per-column imputation method using mechanism diagnosis + steering.

    Consumes ``diagnose_missingness`` (MCAR/MAR/MNAR), builds a dtype/mechanism
    candidate set, filters it through the hard ``SteeringProfile`` constraints,
    runs the mask-and-score harness, and selects the best-recovery method per
    column. The output ``plan`` is directly consumable by ``apply_imputation_plan``.

    Args:
        data_path: CSV to plan for.
        target: Optional target column (excluded from imputation candidates).
        profile: Optional ``SteeringProfile.as_profile_dict()``.
        frac: Mask fraction for the self-evaluation ranking.
        seed: RNG seed.

    Returns:
        ``{"plan":{col:{method,params,recovery,mechanism}}, "diagnosis":{...},
           "steering_honored":bool}`` or ``{"success": False, "error": ...}``.
    """
    df, err = read_csv_or_error(data_path)
    if err is not None:
        return err
    assert df is not None

    diag = diagnose_missingness(data_path=data_path, profile=profile)
    if diag.get("success") is False:
        return diag
    per_col_mech = diag.get("per_column", {})

    plan: dict[str, Any] = {}
    cols_with_missing = [c for c in df.columns if c != target and df[c].isnull().any()]

    for col in cols_with_missing:
        dtype = str(df[col].dtype)
        mech = per_col_mech.get(col, {}).get("mechanism", "MCAR")
        if mech in ("complete", "unusable"):
            continue

        candidates = _candidate_methods(dtype, mech)
        candidates = _filter_by_steering(candidates, col, dtype, profile)
        if not candidates:
            continue

        # datetime/timeseries candidates can't be scored by evaluate_methods (needs
        # numeric recovery); pick the first steering-approved one deterministically.
        if dtype.lower().startswith(("datetime", "date")):
            plan[col] = {"method": candidates[0], "params": None, "mechanism": mech, "recovery": None}
            continue

        non_null = int(df[col].notna().sum())
        if non_null < 20 or len(candidates) == 1:
            plan[col] = {"method": candidates[0], "params": None, "mechanism": mech, "recovery": None}
            continue

        scored = evaluate_methods(df, col, candidates, mechanism=mech, frac=frac, seed=seed)
        if not scored:
            plan[col] = {"method": candidates[0], "params": None, "mechanism": mech, "recovery": None}
            continue
        best, best_metrics = min(scored.items(), key=lambda kv: kv[1]["value"])
        plan[col] = {
            "method": best,
            "params": None,
            "mechanism": mech,
            "recovery": best_metrics,
            "candidates_scored": {m: s["value"] for m, s in scored.items()},
        }

    return {
        "plan": plan,
        "diagnosis": {c: per_col_mech.get(c, {}).get("mechanism") for c in cols_with_missing},
        "steering_honored": True,
        "overall_mechanism": diag.get("overall_mechanism"),
    }
