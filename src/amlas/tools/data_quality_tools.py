"""Data quality scoring, anomaly detection, and automated EDA tools.

Tools:
  - score_data_quality: Comprehensive data quality assessment across 5 dimensions
  - detect_anomalies: Outlier detection via IQR, Z-score, or Isolation Forest
  - generate_eda_report: Automated exploratory data analysis with recommendations
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from strands import tool

from ..errors import tool_error_boundary

# ─── Data Quality Score ──────────────────────────────────────────────────────


@tool
@tool_error_boundary(source="tools.data_quality_tools")
def score_data_quality(data_path: str, target_column: str) -> dict[str, Any]:
    """Comprehensive data quality assessment across 5 dimensions.

    Returns overall score (0-100), per-dimension scores, issues list, and summary.

    Args:
        data_path: Filesystem path to CSV file.
        target_column: Name of the target column.
    """
    p = Path(data_path)
    if not p.exists():
        return {"success": False, "error": f"File not found: {data_path}", "overall_score": 0}

    try:
        df = pd.read_csv(p)
    except Exception as e:
        return {"success": False, "error": f"Failed to read CSV: {e}", "overall_score": 0}

    issues: list[dict[str, str]] = []
    n_rows, n_cols = df.shape

    # ─── Dimension 1: Completeness (% non-null) ──────────────────────────
    total_cells = n_rows * n_cols
    null_cells = int(df.isnull().sum().sum())
    completeness_ratio = 1.0 - (null_cells / total_cells) if total_cells > 0 else 1.0
    completeness_score = completeness_ratio * 100.0

    # Track columns with significant missing values
    for col in df.columns:
        missing_pct = df[col].isnull().mean() * 100
        if missing_pct > 5:
            severity = "high" if missing_pct > 30 else ("medium" if missing_pct > 15 else "low")
            issues.append({
                "column": col,
                "issue": f"{missing_pct:.1f}% missing values",
                "severity": severity,
                "recommendation": (
                    "Drop column" if missing_pct > 60
                    else "Impute with median/mode" if missing_pct > 15
                    else "Impute or investigate"
                ),
            })

    completeness_details = (
        f"{null_cells} null values across {n_cols} columns "
        f"({completeness_ratio*100:.1f}% complete)"
    )

    # ─── Dimension 2: Uniqueness (duplicate rows) ────────────────────────
    n_duplicates = int(df.duplicated().sum())
    duplicate_ratio = n_duplicates / n_rows if n_rows > 0 else 0.0
    uniqueness_score = (1.0 - duplicate_ratio) * 100.0

    if n_duplicates > 0:
        severity = "high" if duplicate_ratio > 0.10 else ("medium" if duplicate_ratio > 0.03 else "low")
        issues.append({
            "column": "_all_",
            "issue": f"{n_duplicates} duplicate rows ({duplicate_ratio*100:.1f}%)",
            "severity": severity,
            "recommendation": "Remove duplicate rows before training",
        })

    uniqueness_details = (
        f"{n_duplicates} duplicate rows out of {n_rows} "
        f"({uniqueness_score:.1f}% unique)"
    )

    # ─── Dimension 3: Consistency (type consistency) ─────────────────────
    # Check for mixed types in object columns (e.g., "123" mixed with 123)
    consistency_penalties = 0
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in object_cols:
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        # Check if numeric-like strings are mixed with text
        numeric_like = non_null.apply(lambda x: _is_numeric_str(x))
        numeric_ratio = numeric_like.mean()
        if 0.1 < numeric_ratio < 0.9:
            consistency_penalties += 1
            issues.append({
                "column": col,
                "issue": f"Mixed types: {numeric_ratio*100:.0f}% numeric-like values in object column",
                "severity": "medium",
                "recommendation": "Convert to appropriate dtype or investigate data entry errors",
            })

    # Penalize up to 30 points for consistency issues
    consistency_score = max(0.0, 100.0 - (consistency_penalties * 15.0))
    consistency_details = (
        f"{consistency_penalties} columns with mixed-type issues"
        if consistency_penalties > 0
        else "All columns have consistent types"
    )

    # ─── Dimension 4: Validity (within expected ranges) ──────────────────
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    validity_penalties = 0

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
        # Check for infinite values
        n_inf = int(np.isinf(series).sum())
        if n_inf > 0:
            validity_penalties += 1
            issues.append({
                "column": col,
                "issue": f"{n_inf} infinite values",
                "severity": "high",
                "recommendation": "Replace infinite values with NaN or appropriate bounds",
            })
        # Check for extreme outliers (>6 std deviations). Skip binary/flag
        # columns: a Gaussian sigma rule on a 0/1 label column
        # flags the rare positive rows — the signal you want to keep — as
        # "outliers to remove", which is misleading and, if followed, destructive.
        if series.nunique() > 2 and series.std() > 0:
            z_scores = np.abs((series - series.mean()) / series.std())
            extreme = int((z_scores > 6).sum())
            if extreme > 0:
                validity_penalties += 1
                issues.append({
                    "column": col,
                    "issue": f"{extreme} extreme outliers (>6σ from mean)",
                    "severity": "medium",
                    "recommendation": "Investigate outliers — may need capping or removal",
                })

    validity_score = max(0.0, 100.0 - (validity_penalties * 10.0))
    validity_details = (
        f"{validity_penalties} validity issues found in numeric columns"
        if validity_penalties > 0
        else "All numeric values within reasonable ranges"
    )

    # ─── Dimension 5: Timeliness (staleness indicators) ──────────────────
    # Check for datetime columns and their recency
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    # Also try to parse object columns that look like dates
    for col in object_cols:
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

    # No datetime = neutral score (not penalized)
    timeliness_score = 80.0 if not datetime_cols else 90.0
    timeliness_details = (
        f"{len(datetime_cols)} datetime columns detected"
        if datetime_cols
        else "No datetime columns detected — timeliness not assessable"
    )

    # ─── Overall Score (weighted average) ────────────────────────────────
    weights = {
        "completeness": 0.30,
        "uniqueness": 0.20,
        "consistency": 0.20,
        "validity": 0.20,
        "timeliness": 0.10,
    }
    overall_score = (
        completeness_score * weights["completeness"]
        + uniqueness_score * weights["uniqueness"]
        + consistency_score * weights["consistency"]
        + validity_score * weights["validity"]
        + timeliness_score * weights["timeliness"]
    )

    # Build summary
    summary_parts = [f"Data quality score: {overall_score:.1f}/100."]
    if issues:
        high_issues = [i for i in issues if i["severity"] == "high"]
        if high_issues:
            summary_parts.append(
                f" Found {len(high_issues)} high-severity issues that should be addressed before modeling."
            )
        summary_parts.append(f" Total issues: {len(issues)}.")
    else:
        summary_parts.append(" Data appears clean and ready for modeling.")

    return {
        "overall_score": round(overall_score, 1),
        "dimensions": {
            "completeness": {"score": round(completeness_score, 1), "details": completeness_details},
            "uniqueness": {"score": round(uniqueness_score, 1), "details": uniqueness_details},
            "consistency": {"score": round(consistency_score, 1), "details": consistency_details},
            "validity": {"score": round(validity_score, 1), "details": validity_details},
            "timeliness": {"score": round(timeliness_score, 1), "details": timeliness_details},
        },
        "issues": issues,
        "summary": "".join(summary_parts),
    }


def _is_numeric_str(val: Any) -> bool:
    """Check if a value looks like a number stored as string."""
    if not isinstance(val, str):
        return False
    try:
        float(val)
        return True
    except (ValueError, TypeError):
        return False


# ─── Anomaly Detection ───────────────────────────────────────────────────────


@tool
@tool_error_boundary(source="tools.data_quality_tools")
def detect_anomalies(
    data_path: str,
    columns: list[str] | None = None,
    method: str = "iqr",
) -> dict[str, Any]:
    """Detect anomalies/outliers in numeric columns.

    Methods:
      - "iqr": Interquartile range (1.5 * IQR)
      - "zscore": Z-score > 3
      - "isolation_forest": Isolation Forest (sklearn)

    Args:
        data_path: Filesystem path to CSV file.
        columns: Specific columns to analyze (default: all numeric).
        method: Detection method — "iqr", "zscore", or "isolation_forest".
    """
    p = Path(data_path)
    if not p.exists():
        return {"success": False, "error": f"File not found: {data_path}"}

    try:
        df = pd.read_csv(p)
    except Exception as e:
        return {"success": False, "error": f"Failed to read CSV: {e}"}

    # Select numeric columns
    all_numeric = df.select_dtypes(include=["number"]).columns.tolist()
    target_cols = [c for c in (columns or all_numeric) if c in all_numeric]

    if not target_cols:
        return {
            "total_anomalies": 0,
            "anomaly_rate": 0.0,
            "per_column": {},
            "recommendation": "keep",
        }

    per_column: dict[str, dict[str, Any]] = {}
    total_anomalies = 0

    if method == "iqr":
        for col in target_cols:
            series = df[col].dropna()
            if len(series) < 4:
                per_column[col] = {"count": 0, "rate": 0.0, "bounds": [float(series.min()), float(series.max())]}
                continue
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = (series < lower) | (series > upper)
            count = int(outlier_mask.sum())
            total_anomalies += count
            per_column[col] = {
                "count": count,
                "rate": round(count / len(series), 4),
                "bounds": [round(lower, 4), round(upper, 4)],
            }

    elif method == "zscore":
        for col in target_cols:
            series = df[col].dropna()
            if len(series) < 4 or series.std() == 0:
                per_column[col] = {"count": 0, "rate": 0.0, "bounds": [float(series.min()), float(series.max())]}
                continue
            mean = float(series.mean())
            std = float(series.std())
            z_scores = np.abs((series - mean) / std)
            outlier_mask = z_scores > 3
            count = int(outlier_mask.sum())
            total_anomalies += count
            lower = mean - 3 * std
            upper = mean + 3 * std
            per_column[col] = {
                "count": count,
                "rate": round(count / len(series), 4),
                "bounds": [round(lower, 4), round(upper, 4)],
            }

    elif method == "isolation_forest":
        from sklearn.ensemble import IsolationForest

        # Use all target columns for multivariate detection
        subset = df[target_cols].dropna()
        if len(subset) < 10:
            return {
                "total_anomalies": 0,
                "anomaly_rate": 0.0,
                "per_column": {},
                "recommendation": "keep",
            }

        iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
        predictions = iso.fit_predict(subset)
        anomaly_mask = predictions == -1
        total_anomalies = int(anomaly_mask.sum())

        # Per-column contribution (which columns have extreme values in anomalous rows)
        for col in target_cols:
            col_data = subset[col]
            anomalous_values = col_data[anomaly_mask]
            # Count per-column outliers using IQR for interpretation
            q1 = float(col_data.quantile(0.25))
            q3 = float(col_data.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            col_outlier_count = int(((anomalous_values < lower) | (anomalous_values > upper)).sum())
            per_column[col] = {
                "count": col_outlier_count,
                "rate": round(col_outlier_count / len(col_data), 4),
                "bounds": [round(lower, 4), round(upper, 4)],
            }
    else:
        return {"success": False, "error": f"Unknown method: {method}. Use 'iqr', 'zscore', or 'isolation_forest'."}

    # Calculate overall anomaly rate
    total_observations = len(df) * len(target_cols)
    anomaly_rate = total_anomalies / total_observations if total_observations > 0 else 0.0

    # Determine recommendation
    if anomaly_rate > 0.10:
        recommendation = "investigate"
    elif anomaly_rate > 0.05:
        recommendation = "cap"
    elif anomaly_rate > 0.01:
        recommendation = "remove"
    else:
        recommendation = "keep"

    return {
        "total_anomalies": total_anomalies,
        "anomaly_rate": round(anomaly_rate, 4),
        "per_column": per_column,
        "recommendation": recommendation,
    }


# ─── Automated EDA Report ────────────────────────────────────────────────────


@tool
@tool_error_boundary(source="tools.data_quality_tools")
def generate_eda_report(
    data_path: str,
    target_column: str,
    task_type: str = "classification",
) -> dict[str, Any]:
    """Generate comprehensive EDA report with feature analysis and recommendations.

    Args:
        data_path: Filesystem path to CSV file.
        target_column: Name of the target column.
        task_type: "classification" or "regression".
    """
    p = Path(data_path)
    if not p.exists():
        return {"success": False, "error": f"File not found: {data_path}"}

    try:
        df = pd.read_csv(p)
    except Exception as e:
        return {"success": False, "error": f"Failed to read CSV: {e}"}

    n_rows, n_cols = df.shape

    # ─── Feature Type Classification ────────────────────────────────────
    numeric_features: list[str] = []
    categorical_features: list[str] = []
    datetime_features: list[str] = []
    text_features: list[str] = []

    for col in df.columns:
        if col == target_column:
            continue
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            numeric_features.append(col)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            datetime_features.append(col)
        else:
            # Check if object column is text (high cardinality + long strings)
            nunique = df[col].nunique()
            avg_len = df[col].dropna().astype(str).str.len().mean() if len(df[col].dropna()) > 0 else 0
            if nunique > 50 and avg_len > 30:
                text_features.append(col)
            else:
                categorical_features.append(col)

    # ─── Target Distribution ────────────────────────────────────────────
    target_dist: dict[str, int] = {}
    class_imbalance = 0.0

    if target_column in df.columns:
        value_counts = df[target_column].value_counts()
        target_dist = {str(k): int(v) for k, v in value_counts.items()}
        if task_type == "classification" and len(value_counts) >= 2:
            minority = int(value_counts.min())
            majority = int(value_counts.max())
            class_imbalance = minority / majority if majority > 0 else 0.0

    # ─── Correlations ──────────────────────────────────────────────────
    correlations: dict[str, list[Any]] = {"top_positive": [], "top_negative": []}

    if target_column in df.columns and numeric_features:
        # Compute correlation with target
        corr_cols = [c for c in numeric_features if c in df.columns]
        if pd.api.types.is_numeric_dtype(df[target_column].dtype) and corr_cols:
            corr_series = df[[*corr_cols, target_column]].corr()[target_column].drop(target_column, errors="ignore")
            sorted_corr = corr_series.sort_values(ascending=False)

            # Top positive correlations
            positives = sorted_corr[sorted_corr > 0].head(5)
            for col_name, corr_val in positives.items():
                correlations["top_positive"].append({"feature": str(col_name), "correlation": round(float(corr_val), 4)})

            # Top negative correlations
            negatives = sorted_corr[sorted_corr < 0].tail(5)
            for col_name, corr_val in negatives.items():
                correlations["top_negative"].append({"feature": str(col_name), "correlation": round(float(corr_val), 4)})

    # ─── Missing Pattern ───────────────────────────────────────────────
    missing_pattern: dict[str, int] = {}
    for col in df.columns:
        n_missing = int(df[col].isnull().sum())
        if n_missing > 0:
            missing_pattern[col] = n_missing

    # ─── Recommendations ───────────────────────────────────────────────
    recommended_actions: list[str] = []

    # Class imbalance
    if task_type == "classification" and class_imbalance < 0.20 and class_imbalance > 0:
        recommended_actions.append(
            f"Severe class imbalance detected (ratio {class_imbalance:.3f}). "
            "Consider SMOTE, class weights, or oversampling the minority class."
        )

    # Missing values
    total_missing = sum(missing_pattern.values())
    if total_missing > 0:
        pct_missing = total_missing / (n_rows * n_cols) * 100
        if pct_missing > 10:
            recommended_actions.append(
                f"Significant missing data ({pct_missing:.1f}%). "
                "Consider imputation strategies (KNN, iterative, or median)."
            )
        else:
            recommended_actions.append(
                f"Minor missing data ({pct_missing:.1f}%). Simple imputation should suffice."
            )

    # High-cardinality categoricals
    for col in categorical_features:
        nunique = df[col].nunique()
        if nunique > 20:
            recommended_actions.append(
                f"High-cardinality categorical '{col}' ({nunique} unique values). "
                "Consider target encoding or embedding."
            )

    # Feature count
    if n_cols > 50:
        recommended_actions.append(
            f"High dimensionality ({n_cols} features). Consider feature selection or PCA."
        )

    # Small dataset
    if n_rows < 1000:
        recommended_actions.append(
            f"Small dataset ({n_rows} rows). Prefer simpler models to avoid overfitting. "
            "Use cross-validation with 5-10 folds."
        )

    # Default recommendation if clean
    if not recommended_actions:
        recommended_actions.append("Data appears suitable for modeling. Start with baseline experiments.")

    # ─── Markdown Report ───────────────────────────────────────────────
    report_lines = [
        "# Exploratory Data Analysis Report",
        "",
        "## Dataset Overview",
        f"- **Rows**: {n_rows}",
        f"- **Columns**: {n_cols}",
        f"- **Target**: `{target_column}` ({task_type})",
        "",
        "## Feature Types",
        f"- Numeric: {len(numeric_features)} features",
        f"- Categorical: {len(categorical_features)} features",
        f"- Datetime: {len(datetime_features)} features",
        f"- Text: {len(text_features)} features",
        "",
    ]

    if target_dist:
        report_lines.extend([
            "## Target Distribution",
            "",
        ])
        for label, count in target_dist.items():
            report_lines.append(f"- `{label}`: {count} ({count/n_rows*100:.1f}%)")
        if class_imbalance > 0:
            report_lines.append(f"\n**Imbalance ratio**: {class_imbalance:.3f}")
        report_lines.append("")

    if missing_pattern:
        report_lines.extend([
            "## Missing Values",
            "",
            "| Column | Missing | % |",
            "|--------|---------|---|",
        ])
        for col, n_miss in sorted(missing_pattern.items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"| {col} | {n_miss} | {n_miss/n_rows*100:.1f}% |")
        report_lines.append("")

    if correlations["top_positive"] or correlations["top_negative"]:
        report_lines.extend([
            "## Top Correlations with Target",
            "",
        ])
        if correlations["top_positive"]:
            report_lines.append("**Positive:**")
            for item in correlations["top_positive"]:
                report_lines.append(f"- {item['feature']}: {item['correlation']:.4f}")
        if correlations["top_negative"]:
            report_lines.append("\n**Negative:**")
            for item in correlations["top_negative"]:
                report_lines.append(f"- {item['feature']}: {item['correlation']:.4f}")
        report_lines.append("")

    report_lines.extend([
        "## Recommendations",
        "",
    ])
    for i, rec in enumerate(recommended_actions, 1):
        report_lines.append(f"{i}. {rec}")

    report_markdown = "\n".join(report_lines)

    return {
        "shape": {"rows": n_rows, "cols": n_cols},
        "target_distribution": target_dist,
        "feature_types": {
            "numeric": numeric_features,
            "categorical": categorical_features,
            "datetime": datetime_features,
            "text": text_features,
        },
        "correlations": correlations,
        "missing_pattern": missing_pattern,
        "class_imbalance": round(class_imbalance, 4),
        "recommended_actions": recommended_actions,
        "report_markdown": report_markdown,
    }
