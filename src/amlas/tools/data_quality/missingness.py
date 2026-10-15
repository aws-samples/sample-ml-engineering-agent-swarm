"""Missingness-mechanism diagnosis + pandera expectations gate
(master plan §2 A2/A3, spec sheet 2.2).

``diagnose_missingness`` implements an **in-house Little's MCAR chi-square test**
(no new dependency, per plan §13) plus a per-column missing-mask classifier
(sklearn LogisticRegression, CV AUC) to separate MCAR / MAR / MNAR.

``check_expectations`` is the declarative silver gate; it lazily imports
**pandera** and degrades with a clear error dict if the ``data`` extra is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from strands import tool

from ...errors import tool_error_boundary

# ─── Little's MCAR chi-square (in-house) ─────────────────────────────────────


def little_mcar_test(df: pd.DataFrame) -> dict[str, Any]:
    """In-house Little's MCAR test over the numeric columns of ``df``.

    Little's test groups rows by their missing-data *pattern* and asks whether
    the observed means within each pattern differ from the grand (ML) means more
    than chance. Under MCAR the weighted squared deviations follow a chi-square
    distribution with ``sum_p (k_p) - k`` degrees of freedom, where ``k_p`` is the
    number of observed variables in pattern ``p`` and ``k`` the number of variables.

    Returns ``{"statistic","df","p_value","n_patterns"}``. ``p_value`` is None when
    the test is undefined (e.g. <2 patterns or insufficient data).
    """
    from scipy import stats

    num = df.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan)
    num = num.dropna(axis=1, how="all")
    cols = list(num.columns)
    k = len(cols)
    if k == 0 or len(num) < 3:
        return {"statistic": 0.0, "df": 0, "p_value": None, "n_patterns": 0}

    # grand means + covariance from complete-case (fallback to pairwise) data.
    global_means = num.mean()
    cov = num.cov()
    # regularize covariance for invertibility
    cov_values = cov.to_numpy()
    cov_values = cov_values + np.eye(k) * (1e-6 + 1e-6 * np.nanmean(np.diag(cov_values)))

    # group by missingness pattern (tuple of observed-column booleans)
    mask = num.notna()
    pattern_keys = mask.apply(lambda r: tuple(r.to_numpy()), axis=1)
    groups = num.groupby(pattern_keys)

    d2 = 0.0
    total_observed = 0
    n_patterns = 0
    for key, sub in groups:
        observed_idx = [i for i, present in enumerate(key) if present]
        if not observed_idx:
            continue
        n_p = len(sub)
        n_patterns += 1
        total_observed += len(observed_idx)
        obs_cols = [cols[i] for i in observed_idx]
        mean_diff = (sub[obs_cols].mean() - global_means[obs_cols]).to_numpy()
        sub_cov = cov_values[np.ix_(observed_idx, observed_idx)]
        try:
            inv = np.linalg.pinv(sub_cov)
        except np.linalg.LinAlgError:
            continue
        d2 += n_p * float(mean_diff @ inv @ mean_diff)

    dof = total_observed - k
    if n_patterns < 2 or dof <= 0:
        return {"statistic": round(d2, 4), "df": int(max(dof, 0)), "p_value": None, "n_patterns": n_patterns}

    p_value = float(1.0 - stats.chi2.cdf(d2, dof))
    return {"statistic": round(d2, 4), "df": int(dof), "p_value": round(p_value, 4), "n_patterns": n_patterns}


def _mask_auc(df: pd.DataFrame, col: str) -> float | None:
    """CV AUC predicting the null-mask of ``col`` from the other numeric columns."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    y = df[col].isnull().astype(int).to_numpy()
    n_missing = int(y.sum())
    if n_missing < 5 or (len(y) - n_missing) < 5:
        return None
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    feats = [c for c in numeric if c != col]
    if not feats:
        return None
    X = df[feats].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    if len(np.unique(y)) < 2:
        return None
    folds = min(5, n_missing, len(y) - n_missing)
    if folds < 2:
        return None
    try:
        Xs = StandardScaler().fit_transform(X)
        scores = cross_val_score(LogisticRegression(max_iter=200), Xs, y, cv=folds, scoring="roc_auc")
        return float(np.mean(scores))
    except Exception:
        return None


def _looks_mnar(series: pd.Series, miss_frac: float) -> bool:
    present = series.dropna()
    if len(present) < 10 or present.std() == 0:
        return False
    skew = float(present.skew())
    return miss_frac > 0.15 and abs(skew) > 1.0


@tool
@tool_error_boundary(source="tools.data_quality.missingness")
def diagnose_missingness(
    data_path: str,
    columns: list[str] | None = None,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Diagnose the missingness mechanism (MCAR / MAR / MNAR) per column.

    Combines an in-house **Little's MCAR chi-square** test (dataset-level) with a
    per-column **missing-mask classifier** (sklearn LogisticRegression, CV AUC):

      * Little p>0.05 AND mask-AUC≈0.5  → ``MCAR`` (any imputer safe)
      * mask-AUC>0.6                    → ``MAR``  (MICE / MissForest / KNN)
      * else, value-dependent heuristic → ``MNAR`` candidate (not statistically
        confirmable — defer to ``mechanism_hints``).

    Args:
        data_path: Filesystem path to a CSV file.
        columns: Optional subset of columns to diagnose (default: all).
        profile: Optional steering dict; ``mechanism_hints`` override the diagnosis.

    Returns:
        ``{"per_column":{col:{mechanism,evidence,mask_auc,confidence}},
           "overall_mechanism":str, "little_p_value":float|None}`` or an error dict.
    """
    # Path-traversal containment (Security review, CWE-22): agent-supplied
    # data_path must stay inside the sandbox roots / CWD / tempdir.
    from ..pathref import resolve_dataset_path

    try:
        p = resolve_dataset_path(data_path)
    except PermissionError as e:
        return {"success": False, "error": str(e)}
    if not p.exists():
        return {"success": False, "error": f"File not found: {data_path}"}
    try:
        df = pd.read_csv(p)
    except Exception as e:
        return {"success": False, "error": f"Failed to read CSV: {e}"}

    n = len(df)
    hints = (profile or {}).get("mechanism_hints", {}) if isinstance(profile, dict) else {}
    target_cols = [c for c in (columns or df.columns) if c in df.columns]

    little = little_mcar_test(df)
    little_p = little["p_value"]

    per_column: dict[str, dict[str, Any]] = {}
    mechanisms: list[str] = []

    for col in target_cols:
        n_missing = int(df[col].isnull().sum())
        low_conf_warn = "" if n >= 10 else " (warning: <10 rows, low confidence)"

        # explicit steering hint wins
        if col in hints:
            mech = str(hints[col]).upper()
            per_column[col] = {
                "mechanism": mech,
                "evidence": f"pinned via mechanism_hints -> {mech}",
                "mask_auc": None,
                "confidence": 1.0,
            }
            mechanisms.append(mech)
            continue

        if n_missing == 0:
            per_column[col] = {
                "mechanism": "complete",
                "evidence": "no missing values",
                "mask_auc": None,
                "confidence": 1.0,
            }
            continue
        if n_missing == n:
            per_column[col] = {
                "mechanism": "unusable",
                "evidence": "column is entirely missing",
                "mask_auc": None,
                "confidence": 1.0,
            }
            continue

        miss_frac = n_missing / n
        auc = _mask_auc(df, col)

        if auc is not None and auc > 0.6:
            mech = "MAR"
            evidence = f"null-mask predictable from other columns (AUC={auc:.2f}){low_conf_warn}"
            confidence = min(1.0, (auc - 0.5) * 2)
        elif _looks_mnar(df[col], miss_frac):
            mech = "MNAR"
            evidence = (
                f"value-dependent missingness suspected (skewed survivors, {miss_frac*100:.0f}% missing); "
                f"not statistically confirmable — confirm via domain hint{low_conf_warn}"
            )
            confidence = 0.4
        else:
            mech = "MCAR"
            auc_txt = f"AUC={auc:.2f}" if auc is not None else "AUC~0.5 (insufficient signal)"
            little_txt = f"Little p={little_p}" if little_p is not None else "Little p undefined"
            evidence = f"null-mask independent of other columns ({auc_txt}; {little_txt}){low_conf_warn}"
            confidence = 0.6 if (little_p is None or little_p > 0.05) else 0.4

        per_column[col] = {
            "mechanism": mech,
            "evidence": evidence,
            "mask_auc": round(auc, 3) if auc is not None else None,
            "confidence": round(confidence, 3),
        }
        mechanisms.append(mech)

    # overall = worst (most concerning) mechanism present
    priority = {"MNAR": 3, "MAR": 2, "MCAR": 1}
    overall = "MCAR"
    if mechanisms:
        overall = max(mechanisms, key=lambda m: priority.get(m, 0))

    return {
        "per_column": per_column,
        "overall_mechanism": overall,
        "little_p_value": little_p,
        "little_test": little,
    }


# ─── pandera expectations gate ───────────────────────────────────────────────


@tool
@tool_error_boundary(source="tools.data_quality.missingness")
def check_expectations(data_path: str, suite: str | None = None) -> dict[str, Any]:
    """Declarative schema / expectations gate over a CSV using **pandera**.

    Auto-infers a ``DataFrameSchema`` (types, non-null on dense columns, numeric
    range bounds, uniqueness on identifier-like columns) when ``suite`` is None,
    otherwise loads a YAML suite. Lazily imports pandera so the package still
    imports without the ``data`` extra.

    Args:
        data_path: Filesystem path to a CSV file.
        suite: Optional path to a pandera YAML schema file.

    Returns:
        ``{"passed":bool,"violations":[{"column","check","n_failed"}],"suite_used":str}``
        or ``{"success": False, "error": ...}``.
    """
    # Path-traversal containment (Security review, CWE-22): agent-supplied
    # data_path must stay inside the sandbox roots / CWD / tempdir.
    from ..pathref import resolve_dataset_path

    try:
        p = resolve_dataset_path(data_path)
    except PermissionError as e:
        return {"success": False, "error": str(e)}
    if not p.exists():
        return {"success": False, "error": f"File not found: {data_path}"}

    try:
        import pandera.pandas as pa  # pandera >=0.20 splits the pandas accessor
    except ImportError:
        try:
            import pandera as pa  # type: ignore[no-redef]  # older pandera (<0.20) layout fallback
        except ImportError:
            return {"success": False, "error": "pandera not installed — install extra: data"}

    try:
        df = pd.read_csv(p)
    except Exception as e:
        return {"success": False, "error": f"Failed to read CSV: {e}"}

    # ── load or infer schema ─────────────────────────────────────────────────
    if suite is not None:
        sp = Path(suite)
        if not sp.exists():
            return {"success": False, "error": f"Suite file not found: {suite}"}
        try:
            schema = pa.DataFrameSchema.from_yaml(sp.read_text())
            suite_used = suite
        except Exception as e:
            return {"success": False, "error": f"Failed to load suite: {e}"}
    else:
        schema = _infer_schema(df, pa)
        suite_used = "auto-inferred"

    # ── validate (lazy=True collects ALL failures) ───────────────────────────
    try:
        schema.validate(df, lazy=True)
        return {"passed": True, "violations": [], "suite_used": suite_used}
    except pa.errors.SchemaErrors as err:
        violations = _summarize_failures(err)
        return {"passed": False, "violations": violations, "suite_used": suite_used}
    except Exception as e:
        return {"success": False, "error": f"Validation error: {e}"}


def _infer_schema(df: pd.DataFrame, pa: Any) -> Any:
    """Auto-infer a reasonable pandera schema (types + range + uniqueness)."""
    from ..leakage_columns import detect_id_like_columns

    columns: dict[str, Any] = {}
    id_cols = set(detect_id_like_columns(df))

    for col in df.columns:
        series = df[col]
        nullable = bool(series.isnull().any())
        checks = []
        dtype: Any
        if pd.api.types.is_integer_dtype(series):
            dtype = int
        elif pd.api.types.is_float_dtype(series):
            dtype = float
        elif pd.api.types.is_bool_dtype(series):
            dtype = bool
        else:
            dtype = object

        # numeric range bounds inferred from observed min/max (validity gate)
        if dtype in (int, float):
            finite = series.replace([np.inf, -np.inf], np.nan).dropna()
            if len(finite) > 0:
                lo, hi = float(finite.min()), float(finite.max())
                if np.isfinite(lo) and np.isfinite(hi):
                    checks.append(pa.Check.in_range(lo, hi, include_min=True, include_max=True))

        unique = col in id_cols and not series.dropna().duplicated().any()

        columns[col] = pa.Column(
            dtype,
            checks=checks or None,
            nullable=nullable,
            unique=unique,
            coerce=True,
            required=True,
        )

    return pa.DataFrameSchema(columns, strict=False)


def _summarize_failures(err: Any) -> list[dict[str, Any]]:
    """Compress a pandera SchemaErrors failure_cases frame into our contract."""
    violations: list[dict[str, Any]] = []
    try:
        fc = err.failure_cases  # DataFrame with column / check / failure_case
        if fc is None or len(fc) == 0:
            return violations
        grouped = fc.groupby(["column", "check"], dropna=False).size().reset_index(name="n_failed")
        for _, row in grouped.iterrows():
            col = row.get("column")
            violations.append({
                "column": None if (col is None or (isinstance(col, float) and pd.isna(col))) else str(col),
                "check": str(row.get("check")),
                "n_failed": int(row["n_failed"]),
            })
    except Exception:
        violations.append({"column": None, "check": "schema", "n_failed": -1})
    return violations
