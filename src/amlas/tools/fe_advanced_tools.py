"""Advanced feature engineering tools — interaction terms, polynomials, target encoding.

Functions (not @tool decorated — used as library utilities):
  - create_interaction_features: Pairwise product and ratio interactions
  - create_polynomial_features: Polynomial expansion (x^2, x^3)
  - target_encode: Target-mean encoding with smoothing
  - advanced_preprocess: Full pipeline combining all techniques
  - apply_advanced_preprocess: Apply saved pipeline config to test set
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

# Columns that are identifiers / not useful for ML
_ID_COLUMNS = {"PassengerId", "Name", "name", "id", "ID", "Id"}


def create_interaction_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    max_pairs: int = 10,
) -> pd.DataFrame:
    """Create pairwise interaction terms between numeric columns.

    Generates:
      - Product interactions (A * B) named "A_x_B"
      - Ratio interactions (A / B) named "A_div_B" where B != 0

    Args:
        df: Input DataFrame.
        columns: Specific numeric columns to use. If None, uses all numeric columns.
        max_pairs: Maximum number of column pairs to generate interactions for.

    Returns:
        New DataFrame with original columns + interaction columns added.
    """
    result = df.copy()

    # Select numeric columns only
    if columns is not None:
        numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    else:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if len(numeric_cols) < 2:
        return result

    # Generate pairs (limited by max_pairs)
    all_pairs = list(combinations(numeric_cols, 2))
    pairs = all_pairs[:max_pairs]

    for col_a, col_b in pairs:
        # Product interaction
        product_name = f"{col_a}_x_{col_b}"
        result[product_name] = df[col_a] * df[col_b]

        # Ratio interaction (A / B) — safe division
        ratio_name = f"{col_a}_div_{col_b}"
        b_values = df[col_b].replace(0, np.nan)
        result[ratio_name] = df[col_a] / b_values

    return result


def create_polynomial_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    degree: int = 2,
    include_bias: bool = False,
) -> pd.DataFrame:
    """Create polynomial features (x^2, x^3) for numeric columns.

    Args:
        df: Input DataFrame.
        columns: Specific columns to expand. If None, uses all numeric columns.
        degree: Polynomial degree (default 2).
        include_bias: Whether to include a bias (all-ones) column.

    Returns:
        DataFrame with original columns + polynomial columns added.
    """
    result = df.copy()

    if columns is not None:
        numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df[c])]
    else:
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_cols:
        return result

    for col in numeric_cols:
        for d in range(2, degree + 1):
            poly_name = f"{col}_pow{d}"
            result[poly_name] = df[col] ** d

    return result


def target_encode(
    df: pd.DataFrame,
    target_col: str,
    categorical_cols: list[str] | None = None,
    smoothing: int = 10,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Encode categorical features using target mean with smoothing.

    Uses Bayesian smoothing formula:
        encoding = (count * category_mean + smoothing * global_mean) / (count + smoothing)

    This prevents overfitting for rare categories by pulling their encoding
    toward the global mean.

    Args:
        df: Input DataFrame (must contain target_col).
        target_col: Name of the target column.
        categorical_cols: Columns to encode. If None, auto-detects object/category columns.
        smoothing: Smoothing factor (higher = more regularization toward global mean).

    Returns:
        Tuple of (encoded DataFrame, mapping dict for test set application).
    """
    result = df.copy()
    mapping: dict[str, dict[str, float]] = {}

    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        # Exclude target if it happens to be categorical
        categorical_cols = [c for c in categorical_cols if c != target_col]

    global_mean = df[target_col].mean()

    for col in categorical_cols:
        # Calculate per-category statistics
        stats = df.groupby(col)[target_col].agg(["mean", "count"])
        category_mean = stats["mean"]
        category_count = stats["count"]

        # Bayesian smoothed encoding
        smooth_encoding = (
            (category_count * category_mean + smoothing * global_mean)
            / (category_count + smoothing)
        )

        col_mapping = smooth_encoding.to_dict()
        mapping[col] = col_mapping

        # Apply encoding
        result[col] = df[col].map(col_mapping).astype(float)

    return result, mapping


def advanced_preprocess(
    df: pd.DataFrame,
    target_col: str,
    task_type: str = "classification",
    enable_interactions: bool = True,
    enable_polynomials: bool = False,
    enable_target_encoding: bool = True,
    correlation_threshold: float = 0.95,
    max_interaction_pairs: int = 10,
    polynomial_degree: int = 2,
    target_encoding_smoothing: int = 10,
    high_cardinality_threshold: int = 10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extended preprocessing pipeline combining all advanced techniques.

    Applies:
      1. Remove ID columns
      2. Handle missing values (numeric: median, categorical: mode)
      3. Target encoding for high-cardinality categoricals (>threshold unique values)
      4. One-hot encoding for low-cardinality categoricals
      5. Interaction features (top correlated pairs)
      6. Polynomial features (optional, degree 2)
      7. Feature selection via correlation threshold (drop r > threshold)
      8. Standard scaling of numeric features

    Args:
        df: Input DataFrame with target column.
        target_col: Name of the target column.
        task_type: "classification" or "regression".
        enable_interactions: Whether to add interaction features.
        enable_polynomials: Whether to add polynomial features.
        enable_target_encoding: Whether to use target encoding for high-card cats.
        correlation_threshold: Drop features with |r| above this (default 0.95).
        max_interaction_pairs: Max interaction pairs to generate.
        polynomial_degree: Degree for polynomial features.
        target_encoding_smoothing: Smoothing for target encoding.
        high_cardinality_threshold: Unique value count above which to use target encoding.

    Returns:
        Tuple of (processed DataFrame without target, pipeline config dict).
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    # Copy to avoid mutation
    work_df = df.copy()

    # Track pipeline configuration for reproducibility
    config: dict[str, Any] = {
        "enable_interactions": enable_interactions,
        "enable_polynomials": enable_polynomials,
        "enable_target_encoding": enable_target_encoding,
        "correlation_threshold": correlation_threshold,
        "max_interaction_pairs": max_interaction_pairs,
        "polynomial_degree": polynomial_degree,
        "target_encoding_smoothing": target_encoding_smoothing,
        "high_cardinality_threshold": high_cardinality_threshold,
        "target_col": target_col,
    }

    # Step 1: Remove ID columns and target
    exclude = _ID_COLUMNS | {target_col}
    feature_cols = [c for c in work_df.columns if c not in exclude]
    work_df = work_df[feature_cols]

    # Step 2: Identify column types
    numeric_cols = work_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = work_df.select_dtypes(include=["object", "bool", "category"]).columns.tolist()

    config["numeric_columns"] = numeric_cols
    config["categorical_columns"] = categorical_cols

    # Step 3: Handle missing values for numeric
    if numeric_cols:
        imputer = SimpleImputer(strategy="median")
        work_df[numeric_cols] = imputer.fit_transform(work_df[numeric_cols])
        config["numeric_imputer_values"] = dict(zip(numeric_cols, imputer.statistics_.tolist(), strict=False))

    # Step 4: Handle categorical columns
    # Separate high-cardinality from low-cardinality
    high_card_cols = []
    low_card_cols = []
    for col in categorical_cols:
        # Impute missing with mode first
        mode_val = work_df[col].mode()
        if len(mode_val) > 0:
            work_df[col] = work_df[col].fillna(mode_val.iloc[0])

        n_unique = work_df[col].nunique()
        if n_unique > high_cardinality_threshold:
            high_card_cols.append(col)
        else:
            low_card_cols.append(col)

    config["high_cardinality_cols"] = high_card_cols
    config["low_cardinality_cols"] = low_card_cols

    # Step 5: Target encoding for high-cardinality categoricals
    target_encoding_mapping: dict[str, dict[str, float]] = {}
    if enable_target_encoding and high_card_cols:
        # Need target for encoding
        target_series = df[target_col]
        if target_series.dtype == object or target_series.dtype == bool:
            target_series = target_series.map({True: 1, False: 0, "True": 1, "False": 0})

        temp_df = work_df.copy()
        temp_df[target_col] = target_series.values

        encoded_df, mapping = target_encode(
            temp_df, target_col=target_col,
            categorical_cols=high_card_cols,
            smoothing=target_encoding_smoothing,
        )
        # Update work_df with encoded values
        for col in high_card_cols:
            work_df[col] = encoded_df[col]

        target_encoding_mapping = mapping
        # Now these columns are numeric
        numeric_cols = numeric_cols + high_card_cols

    config["target_encoding_mapping"] = target_encoding_mapping

    # Step 6: One-hot encode low-cardinality categoricals
    if low_card_cols:
        work_df = pd.get_dummies(work_df, columns=low_card_cols, drop_first=False)

    # Update numeric columns list after one-hot encoding
    all_numeric_cols = work_df.select_dtypes(include=["number"]).columns.tolist()

    # Step 7: Interaction features
    if enable_interactions and len(all_numeric_cols) >= 2:
        # Select top correlated columns for interactions (avoid explosion)
        interaction_cols = all_numeric_cols[:min(len(all_numeric_cols), 10)]
        work_df = create_interaction_features(
            work_df, columns=interaction_cols, max_pairs=max_interaction_pairs
        )

    # Step 8: Polynomial features
    if enable_polynomials and all_numeric_cols:
        # Only apply to original numeric columns (not derived ones)
        poly_cols = numeric_cols[:5]  # Limit to avoid feature explosion
        work_df = create_polynomial_features(
            work_df, columns=poly_cols, degree=polynomial_degree
        )

    # Step 9: Drop highly correlated features
    dropped_columns: list[str] = []
    numeric_work_cols = work_df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_work_cols) > 1:
        # Handle NaN for correlation computation
        corr_df = work_df[numeric_work_cols].fillna(0)
        corr_matrix = corr_df.corr().abs()

        # Find pairs above threshold
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [
            col for col in upper_triangle.columns
            if any(upper_triangle[col] > correlation_threshold)
        ]
        dropped_columns = to_drop
        work_df = work_df.drop(columns=to_drop)

    config["dropped_columns"] = dropped_columns

    # Step 10: Standard scaling
    final_numeric_cols = work_df.select_dtypes(include=["number"]).columns.tolist()
    if final_numeric_cols:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(work_df[final_numeric_cols].fillna(0))
        work_df[final_numeric_cols] = scaled_values
        config["scaler_mean"] = dict(zip(final_numeric_cols, scaler.mean_.tolist(), strict=False))
        config["scaler_scale"] = dict(zip(final_numeric_cols, scaler.scale_.tolist(), strict=False))

    config["final_columns"] = work_df.columns.tolist()

    # Ensure no NaN remains
    work_df = work_df.fillna(0)

    return work_df, config


def apply_advanced_preprocess(
    df: pd.DataFrame,
    config: dict[str, Any],
    target_col: str | None = None,
) -> pd.DataFrame:
    """Apply a saved pipeline config to a new (test) DataFrame.

    Ensures the test set gets identical transformations as the training set.

    Args:
        df: New DataFrame to transform.
        config: Pipeline config dict from advanced_preprocess.
        target_col: Target column name (excluded from features).

    Returns:
        Transformed DataFrame with same columns as training output.
    """

    if target_col is None:
        target_col = config.get("target_col", "target")

    work_df = df.copy()

    # Step 1: Remove ID columns and target
    exclude = _ID_COLUMNS | {target_col}
    feature_cols = [c for c in work_df.columns if c not in exclude]
    work_df = work_df[feature_cols]

    # Step 2: Impute numeric with saved values
    numeric_cols = config.get("numeric_columns", [])
    imputer_values = config.get("numeric_imputer_values", {})
    for col in numeric_cols:
        if col in work_df.columns and col in imputer_values:
            work_df[col] = work_df[col].fillna(imputer_values[col])

    # Step 3: Impute categorical with mode
    categorical_cols = config.get("categorical_columns", [])
    for col in categorical_cols:
        if col in work_df.columns:
            mode_val = work_df[col].mode()
            if len(mode_val) > 0:
                work_df[col] = work_df[col].fillna(mode_val.iloc[0])

    # Step 4: Target encoding for high-cardinality
    high_card_cols = config.get("high_cardinality_cols", [])
    target_encoding_mapping = config.get("target_encoding_mapping", {})
    for col in high_card_cols:
        if col in work_df.columns and col in target_encoding_mapping:
            col_mapping = target_encoding_mapping[col]
            # Map known categories, fill unknowns with global mean (mean of mapping values)
            global_mean = np.mean(list(col_mapping.values()))
            work_df[col] = work_df[col].map(col_mapping).fillna(global_mean).astype(float)

    # Step 5: One-hot encode low-cardinality categoricals
    low_card_cols = config.get("low_cardinality_cols", [])
    existing_low_card = [c for c in low_card_cols if c in work_df.columns]
    if existing_low_card:
        work_df = pd.get_dummies(work_df, columns=existing_low_card, drop_first=False)

    # Step 6: Interaction features
    if config.get("enable_interactions", False):
        all_numeric = work_df.select_dtypes(include=["number"]).columns.tolist()
        if len(all_numeric) >= 2:
            interaction_cols = all_numeric[:min(len(all_numeric), 10)]
            work_df = create_interaction_features(
                work_df, columns=interaction_cols,
                max_pairs=config.get("max_interaction_pairs", 10),
            )

    # Step 7: Polynomial features
    if config.get("enable_polynomials", False):
        poly_cols = [c for c in numeric_cols if c in work_df.columns][:5]
        if poly_cols:
            work_df = create_polynomial_features(
                work_df, columns=poly_cols,
                degree=config.get("polynomial_degree", 2),
            )

    # Step 8: Drop correlated columns (same ones as training)
    dropped_columns = config.get("dropped_columns", [])
    cols_to_drop = [c for c in dropped_columns if c in work_df.columns]
    if cols_to_drop:
        work_df = work_df.drop(columns=cols_to_drop)

    # Step 9: Standard scaling with saved params
    scaler_mean = config.get("scaler_mean", {})
    scaler_scale = config.get("scaler_scale", {})
    final_columns = config.get("final_columns", [])

    # Ensure all expected columns exist (add missing as zeros)
    for col in final_columns:
        if col not in work_df.columns:
            work_df[col] = 0.0

    # Keep only final columns in correct order
    work_df = work_df[[c for c in final_columns if c in work_df.columns]]

    # Apply scaling
    for col in work_df.columns:
        if col in scaler_mean and col in scaler_scale:
            scale = scaler_scale[col]
            if scale != 0:
                work_df[col] = (work_df[col].fillna(0) - scaler_mean[col]) / scale

    # Final NaN cleanup
    work_df = work_df.fillna(0)

    return work_df
