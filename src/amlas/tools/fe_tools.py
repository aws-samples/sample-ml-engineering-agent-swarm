"""Strands @tool wrappers for automatic feature engineering / preprocessing.

Tools:
  - auto_preprocess: Auto-detect and apply preprocessing for numeric and categorical features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from strands import tool

from amlas.security.model_artifacts import dump_model_artifact

from ..errors import tool_error_boundary

# Columns that are identifiers / not useful for ML
_ID_COLUMNS = {"PassengerId", "Name", "name", "id", "ID", "Id"}


def _identify_column_types(df: pd.DataFrame, target_column: str) -> tuple[list[str], list[str]]:
    """Separate numeric and categorical columns (excluding target and IDs)."""
    exclude = _ID_COLUMNS | {target_column}
    numeric_cols = []
    categorical_cols = []

    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


@tool
@tool_error_boundary(source="tools.fe_tools")
def auto_preprocess(
    path: str,
    target_column: str,
    output_dir: str | None = None,
    drop_threshold: float = 0.5,
) -> dict[str, Any]:
    """Automatically preprocess a CSV dataset for ML training.

    Applies:
      - Drop columns with >drop_threshold fraction missing
      - Numeric: impute missing with median, standard scale
      - Categorical: impute missing with mode, one-hot encode
      - Excludes ID/Name columns automatically

    Args:
        path: Filesystem path to CSV file.
        target_column: Name of the target column (excluded from features).
        output_dir: Optional directory to save processed CSV and Skops pipeline artifact.
        drop_threshold: Fraction of missing values above which a column is dropped (default 0.5).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    p = Path(path)
    if not p.exists():
        return {"success": False, "error": f"File not found: {path}"}

    try:
        df = pd.read_csv(p)
    except Exception as e:
        return {"success": False, "error": f"Failed to read CSV: {e}"}

    if target_column not in df.columns:
        return {"success": False, "error": f"Target column '{target_column}' not found"}

    # Identify columns to drop (>threshold missing)
    missing_frac = df.isnull().mean()
    dropped_columns = [col for col in df.columns if missing_frac[col] > drop_threshold and col != target_column]

    # Remove ID columns and high-missing columns
    exclude = _ID_COLUMNS | {target_column} | set(dropped_columns)
    feature_df = df[[c for c in df.columns if c not in exclude]]

    # Identify column types
    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = feature_df.select_dtypes(include=["object", "bool"]).columns.tolist()

    # Build preprocessing pipeline
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # PERF-14 / high-cardinality one-hot guard (mirrors training_tools.py:289): cap the
    # encoded feature space so a pathological near-unique categorical column (e.g. ADSA
    # ``Cabin`` = 6560 uniques in 8693 rows) cannot explode the dense matrix to thousands
    # of columns. Left unbounded, that ~6.6k-wide dense matrix allocated ~436 MiB
    # (shape (8693, 6570)) and OOM'd the ~4GB Linux CI runner (numpy ArrayMemoryError);
    # macOS' larger RAM absorbed it, hence the local/CI divergence. ``max_categories=50``
    # keeps the 50 most-frequent levels and folds the long rare tail into a single
    # "infrequent" bucket; ``handle_unknown="infrequent_if_exist"`` routes unseen
    # categories there too (falling back to the all-zeros row when no infrequent bucket
    # exists — the same effect the prior ``"ignore"`` gave). Columns with ≤50 distinct
    # values are encoded IDENTICALLY to before (the cap never binds), so the small
    # in-repo fixtures are numerically unchanged.
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                sparse_output=False,
                max_categories=50,
            ),
        ),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    # Fit and transform
    X = feature_df
    X_processed = preprocessor.fit_transform(X)

    # Get feature names after transformation
    feature_names = list(numeric_cols)
    if categorical_cols:
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
        feature_names.extend(cat_feature_names)

    # Extract target
    y = df[target_column]
    if y.dtype == object or y.dtype == bool:
        y = y.map({"True": 1, "False": 0, True: 1, False: 0})

    result: dict[str, Any] = {
        "success": True,
        "n_rows": len(df),
        "n_numeric_features": len(numeric_cols),
        "n_categorical_original": len(categorical_cols),
        "n_features_after": X_processed.shape[1],
        "n_missing_after": int(np.isnan(X_processed).sum()) if isinstance(X_processed, np.ndarray) else 0,
        "dropped_columns": dropped_columns,
        "feature_names": feature_names,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
    }

    # Save artifacts if output_dir specified
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save processed features as CSV
        processed_df = pd.DataFrame(X_processed, columns=feature_names)
        processed_df[target_column] = y.values
        processed_path = out / "processed.csv"
        processed_df.to_csv(processed_path, index=False)
        result["processed_path"] = str(processed_path)

        # Persist the fitted preprocessing graph in inspect-before-load Skops format.
        pipeline_path = out / "preprocessing_pipeline.skops"
        dump_model_artifact(preprocessor, pipeline_path)
        result["pipeline_path"] = str(pipeline_path)

    return result
