"""Data-driven identifier / leakage column detection (LEAK-01).

The deterministic training path historically excluded only a hardcoded Titanic
set (``PassengerId``/``Name``/``id``…), which missed the AI4I ``Product ID``/``UDI``
leakage columns called out in the spec. This module replaces that static set with
a heuristic, data-driven detector so identifier and perfect-predictor columns are
dropped regardless of dataset.

Detection heuristics (any match flags a column, excluding the target):
  1. Name looks like an identifier — token in {id, uid, uuid, guid, key, index,
     udi} or the name ends with "id"/"_id"/" id" (e.g. ``Product ID``, ``UDI``,
     ``PassengerId``), or is a known name field (``name``).
  2. High-cardinality near-unique column — number of distinct values is >= 99%
     of the row count (a row identifier perfectly predicts the target on train).
"""

from __future__ import annotations

import re

import pandas as pd

# Explicit identifier tokens (matched case-insensitively against normalized names).
_ID_TOKENS = {"id", "uid", "uuid", "guid", "key", "index", "udi", "name"}

# Backwards-compatible static hints (kept so callers importing this keep working).
STATIC_ID_COLUMNS = {"PassengerId", "Name", "name", "id", "ID", "Id", "Product ID", "UDI"}


def _normalize(name: str) -> str:
    """Lowercase and collapse non-alphanumerics to single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def _looks_like_id_name(column: str) -> bool:
    """True if the column name suggests an identifier (LEAK-01 heuristic 1)."""
    norm = _normalize(column)
    if not norm:
        return False
    tokens = norm.split(" ")
    # Whole-name token match (e.g. "id", "udi", "name", "product id" -> tokens).
    if any(tok in _ID_TOKENS for tok in tokens):
        return True
    # Trailing "...id" with no separator (e.g. "passengerid", "productid").
    if norm.replace(" ", "").endswith("id"):
        return True
    return False


def detect_id_like_columns(
    df: pd.DataFrame,
    target_column: str | None = None,
    uniqueness_threshold: float = 0.99,
) -> list[str]:
    """Return columns that look like identifiers / leakage columns (LEAK-01).

    Args:
        df: Input dataframe.
        target_column: Column to never flag (the prediction target).
        uniqueness_threshold: Fraction of distinct values (relative to row count)
            at/above which a column is treated as a near-unique row identifier.

    Returns:
        Sorted list of column names to exclude from features.
    """
    flagged: set[str] = set()
    n_rows = len(df)

    for col in df.columns:
        if target_column is not None and col == target_column:
            continue

        # Heuristic 1 — name looks like an identifier.
        if _looks_like_id_name(col):
            flagged.add(col)
            continue

        # Heuristic 2 — high-cardinality near-unique column (perfect predictor).
        # Only applies to string/object and INTEGER columns: continuous float
        # features are legitimately near-unique and must NOT be flagged.
        if n_rows > 0:
            series = df[col]
            is_float = pd.api.types.is_float_dtype(series)
            if not is_float:
                nunique = series.nunique(dropna=True)
                if nunique >= uniqueness_threshold * n_rows and nunique > 1:
                    flagged.add(col)

    return sorted(flagged)
