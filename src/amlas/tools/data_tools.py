"""Strands @tool wrappers for dataset loading and splitting.

Tools:
  - load_dataset: Load CSV from local path / ``s3://`` URI / ``http(s)://`` URL
  - split_dataset: Train/test split with optional stratification
  - describe_dataset: Summary statistics, dtypes, missing values

Feature 1 (remote reach): the READ side of each tool funnels through the shared
``amlas.tools.pathref`` seam, so a local path keeps the existing sandbox behavior
while ``s3://`` (via the resolved AWS profile) and ``http(s)://`` (anonymous,
SSRF-guarded) references load transparently. ``split_dataset``'s ``output_dir``
WRITE stays local-only (writing datasets to S3 is out of scope for this iteration).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from strands import tool

from ..errors import tool_error_boundary
from .pathref import PathRefError, read_dataframe, resolve_ref


@tool
@tool_error_boundary(source="tools.data_tools")
def load_dataset(path: str) -> dict[str, Any]:
    """Load a CSV dataset and return a summary of its shape and columns.

    Accepts local paths, ``s3://`` URIs, and ``http(s)://`` URLs (remote reads ride
    the resolved AWS profile for S3 and an anonymous, SSRF-guarded GET for HTTP).

    Args:
        path: Path or URI to a CSV file (local / ``s3://`` / ``http(s)://``).
    """
    try:
        # Explicit user-provided dataset path (tier-1 ``/data`` action): opt out of
        # the agent fs sandbox so a legit local dataset OUTSIDE the workspace roots
        # (byte-identical to the pre-Feature-1 ``pd.read_csv(Path(path))`` contract)
        # still loads. Remote (S3/HTTP) gating + SSRF guard are UNAFFECTED.
        ref = resolve_ref(path, sandbox_local=False)
        df = read_dataframe(ref)
    except PathRefError as e:
        return {"success": False, "error": str(e)}
    except FileNotFoundError:
        return {"success": False, "error": f"File not found: {path}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to read CSV: {e}"}

    return {
        "success": True,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        # intentional schema preview (3 rows) — do not pass PII-bearing files
        # without redaction first (see pii_tools.redact_pii).
        "sample_rows": df.head(3).to_dict(orient="records"),
    }


@tool
def split_dataset(
    path: str,
    target_column: str,
    test_size: float = 0.2,
    stratify: bool = True,
    seed: int = 42,
    output_dir: str | None = None,
) -> dict[str, Any]:
    """Split a dataset into train/test sets with optional stratification.

    The source ``path`` accepts local paths, ``s3://`` URIs, and ``http(s)://`` URLs
    (the read rides ``pathref``); ``output_dir`` WRITES remain local-only.

    Args:
        path: Path or URI to a CSV file (local / ``s3://`` / ``http(s)://``).
        target_column: Name of the target column for stratification.
        test_size: Fraction of data to use as test set (0.0-1.0).
        stratify: Whether to stratify by target column.
        seed: Random seed for reproducibility.
        output_dir: Optional LOCAL directory to save train/test CSVs.
    """
    from sklearn.model_selection import train_test_split

    try:
        # Explicit user-provided dataset path: opt out of the agent fs sandbox
        # (see ``load_dataset``). ``output_dir`` writes below stay local-only.
        ref = resolve_ref(path, sandbox_local=False)
        df = read_dataframe(ref)
    except PathRefError as e:
        return {"success": False, "error": str(e)}
    except FileNotFoundError:
        return {"success": False, "error": f"File not found: {path}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to read CSV: {e}"}

    if target_column not in df.columns:
        return {"success": False, "error": f"Target column '{target_column}' not in dataset. Available: {list(df.columns)}"}

    stratify_col = df[target_column] if stratify else None
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=stratify_col
    )

    # Compute target ratios
    train_ratio = train_df[target_column].mean() if train_df[target_column].dtype != object else (train_df[target_column] == train_df[target_column].mode().iloc[0]).mean()
    test_ratio = test_df[target_column].mean() if test_df[target_column].dtype != object else (test_df[target_column] == test_df[target_column].mode().iloc[0]).mean()

    # Handle boolean target
    if train_df[target_column].dtype == bool or set(train_df[target_column].dropna().unique()) <= {True, False, "True", "False"}:
        train_ratio = (train_df[target_column].astype(str) == "True").mean()
        test_ratio = (test_df[target_column].astype(str) == "True").mean()

    result: dict[str, Any] = {
        "success": True,
        "train_size": len(train_df),
        "test_size": len(test_df),
        "train_target_ratio": float(train_ratio),
        "test_target_ratio": float(test_ratio),
    }

    # Save to disk if output_dir specified
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        train_path = out / "train_split.csv"
        test_path = out / "test_split.csv"
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        result["train_path"] = str(train_path)
        result["test_path"] = str(test_path)

    return result


@tool
def describe_dataset(path: str, target_column: str | None = None) -> dict[str, Any]:
    """Return summary statistics for a dataset: shape, dtypes, missing values, distributions.

    Accepts local paths, ``s3://`` URIs, and ``http(s)://`` URLs (remote reads ride
    the resolved AWS profile for S3 and an anonymous, SSRF-guarded GET for HTTP).

    Args:
        path: Path or URI to a CSV file (local / ``s3://`` / ``http(s)://``).
        target_column: Optional target column for class distribution reporting.
    """
    try:
        # Explicit user-provided dataset path: opt out of the agent fs sandbox
        # (see ``load_dataset``). Remote (S3/HTTP) gating + SSRF guard unaffected.
        ref = resolve_ref(path, sandbox_local=False)
        df = read_dataframe(ref)
    except PathRefError as e:
        return {"success": False, "error": str(e)}
    except FileNotFoundError:
        return {"success": False, "error": f"File not found: {path}"}
    except Exception as e:
        return {"success": False, "error": f"Failed to read CSV: {e}"}

    # Missing value counts
    missing_counts = df.isnull().sum().to_dict()

    # Numeric statistics
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_stats = {}
    for col in numeric_cols:
        numeric_stats[col] = {
            "mean": float(df[col].mean()) if not df[col].isna().all() else None,
            "std": float(df[col].std()) if not df[col].isna().all() else None,
            "min": float(df[col].min()) if not df[col].isna().all() else None,
            "max": float(df[col].max()) if not df[col].isna().all() else None,
            "missing": int(df[col].isna().sum()),
        }

    # Categorical statistics
    cat_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()
    categorical_stats = {}
    for col in cat_cols:
        value_counts = df[col].value_counts().head(10).to_dict()
        categorical_stats[col] = {
            "n_unique": int(df[col].nunique()),
            "top_values": {str(k): int(v) for k, v in value_counts.items()},
            "missing": int(df[col].isna().sum()),
        }

    result: dict[str, Any] = {
        "success": True,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": {k: int(v) for k, v in missing_counts.items()},
        "numeric_stats": numeric_stats,
        "categorical_stats": categorical_stats,
    }

    # Target distribution
    if target_column and target_column in df.columns:
        target_dist = df[target_column].value_counts().to_dict()
        result["target_distribution"] = {str(k): int(v) for k, v in target_dist.items()}

    return result
