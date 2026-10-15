"""Missing-Data Imputation Engine (Capability Group B — Phase 7).

Implements the master-plan §6 multi-strategy imputation engine:

  * :mod:`numeric`     — mean/median/mode, KNN, IterativeImputer (MICE),
    RandomForest IterativeImputer (MissForest), and an in-house low-rank
    ``matrix_factorization`` fill (``impute_numeric``).
  * :mod:`categorical` — mode + classifier-based categorical fill
    (``impute_categorical``).
  * :mod:`timeseries`  — pandas interpolation / ffill / bfill + statsmodels
    seasonal decomposition (``impute_timeseries``).
  * :mod:`llm_extrapolate` — injectable LLM-strategy gap-fill for numeric AND
    text/"verbiage" columns (``llm_extrapolate_values``), testable without live
    Bedrock via a mock callable.
  * :mod:`evaluate`    — the mask-and-score self-evaluation harness
    (``evaluate_imputation``) and the steering-enforcing write-back
    (``apply_imputation_plan``).
  * :mod:`engine`      — the multi-strategy decision engine that consumes the
    Phase-6 MCAR/MAR/MNAR diagnosis + steering to rank & select a per-column
    method (``recommend_imputation_plan``).

Conventions (match ``tools/data_quality/``):
  - Tools are ``@tool``-decorated, path-based, return a JSON-serializable dict,
    and fail soft with ``{"success": False, "error": "..."}``.
  - Optional heavy deps (statsmodels) are imported lazily inside the function so
    the package still imports when the ``data`` extra is absent.
"""

from __future__ import annotations

from .categorical import impute_categorical
from .engine import recommend_imputation_plan
from .evaluate import apply_imputation_plan, evaluate_imputation
from .llm_extrapolate import llm_extrapolate_values
from .numeric import NUMERIC_METHODS, impute_numeric
from .timeseries import TIMESERIES_METHODS, impute_timeseries

__all__ = [
    "NUMERIC_METHODS",
    "TIMESERIES_METHODS",
    "apply_imputation_plan",
    "evaluate_imputation",
    "impute_categorical",
    "impute_numeric",
    "impute_timeseries",
    "llm_extrapolate_values",
    "recommend_imputation_plan",
]
