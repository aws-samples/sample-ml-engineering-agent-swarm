"""Hive-partition helper + partition-strategy advisor (master-plan gap **T-2**, Phase 7).

Closes the Cat-8 lakehouse partitioning gap with two ``@tool``s that complement
the existing ``write_lake_dataset`` (which had no ``by=<key>`` Hive-partition
helper):

  * :func:`partition_dataset`            (DQ-LAKE-009) — write a dataset as a
    Hive-style ``key=value/`` partition tree so downstream engines (Athena/Glue)
    can PRUNE partitions. **Offline-safe**: writes a local directory tree (no live
    AWS), so it is fully directly verifiable with a temp dir.
  * :func:`recommend_partition_strategy` (DQ-LAKE-010) — profile a dataset's
    per-column cardinality/size and RECOMMEND partition keys that balance file
    size against query pruning (deterministic heuristic).

Conventions (match ``tools/lake/__init__.py``): ``@tool``-decorated, path-based,
JSON-serializable dict return, fail-soft ``{"success": False, "error": "..."}``.
Reuses the lake module's format-dispatching ``_read_frame``/``_write_frame``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from strands import tool

from ...errors import tool_error_boundary

# Reuse the lake module's format-dispatching frame I/O (CSV/Parquet) — no dup.
from . import _read_frame, _write_frame

__all__ = ["partition_dataset", "recommend_partition_strategy"]

# Heuristic bounds (research-KB partitioning guidance): a good partition key has a
# SMALL number of distinct values (few, well-sized partitions) — not near-unique
# (one file per row = tiny-file explosion) and not constant (a single partition).
_MAX_PARTITION_CARDINALITY = 50  # absolute cap on distinct values for a key
_MAX_CARDINALITY_RATIO = 0.5  # distinct/rows must be below this (not id-like)


def _hive_segment(column: str, value: Any) -> str:
    """Render one Hive partition path segment ``column=value`` (filesystem-safe)."""
    safe = str(value).replace("/", "_").replace(" ", "_")
    return f"{column}={safe}"


@tool
@tool_error_boundary(source="tools.lake.partition")
def partition_dataset(
    data_path: str,
    by: list[str],
    dest_dir: str | None = None,
    fmt: str = "csv",
) -> dict[str, Any]:
    """Write a dataset as a Hive-style ``key=value/`` partition tree (DQ-LAKE-009).

    Groups the rows by the ``by`` partition key(s) and writes one data file per
    partition under ``dest_dir/col=val[/col2=val2]/part.{fmt}`` — the layout Athena
    / Glue prune on. **Offline-safe**: writes to a local directory (a per-run
    ``tempfile.mkdtemp`` when ``dest_dir`` is omitted), never live S3, so it is
    deterministic + directly verifiable.

    Args:
        data_path: Filesystem path to the source CSV/Parquet.
        by: One or more column names to partition by (Hive order).
        dest_dir: Output root dir for the partition tree (default: a temp dir).
        fmt: Partition file format — ``csv`` (default) or ``parquet``.

    Returns:
        ``{"partition_root","partition_keys","num_partitions","partition_paths",
        "rows","fmt"}`` on success, or ``{"success": False, "error": ...}``
        (missing file, empty ``by``, or an unknown partition column).
    """
    if not by:
        return {"success": False, "error": "no partition keys given (by=[] is empty)"}

    p = Path(data_path)
    if not p.exists():
        return {"success": False, "error": f"File not found: {data_path}"}
    try:
        df = _read_frame(p)
    except Exception as e:
        return {"success": False, "error": f"Failed to read dataset: {e}"}

    missing = [c for c in by if c not in df.columns]
    if missing:
        return {
            "success": False,
            "error": f"partition column(s) not found: {missing}; columns are {list(df.columns)}",
        }

    suffix = ".parquet" if str(fmt).lower() == "parquet" else ".csv"
    root = Path(dest_dir) if dest_dir else Path(tempfile.mkdtemp(prefix="amlas_part_"))
    root.mkdir(parents=True, exist_ok=True)

    partition_paths: list[str] = []
    # ``sort=True`` for a deterministic partition order across runs; ``dropna=False``
    # so a NaN-keyed row is partitioned into a ``col=nan`` partition rather than
    # SILENTLY DROPPED — a partition helper must never lose rows (no-data-loss).
    for keys, group in df.groupby(by, sort=True, dropna=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        segments = [_hive_segment(col, val) for col, val in zip(by, key_tuple, strict=True)]
        part_dir = root.joinpath(*segments)
        part_dir.mkdir(parents=True, exist_ok=True)
        part_file = part_dir / f"part{suffix}"
        try:
            _write_frame(group.reset_index(drop=True), part_file)
        except Exception as e:
            return {"success": False, "error": f"failed to write partition {part_dir}: {e}"}
        partition_paths.append(str(part_dir))

    return {
        "partition_root": str(root),
        "partition_keys": list(by),
        "num_partitions": len(partition_paths),
        "partition_paths": partition_paths,
        "rows": len(df),
        "fmt": "parquet" if suffix == ".parquet" else "csv",
    }


@tool
@tool_error_boundary(source="tools.lake.partition")
def recommend_partition_strategy(data_path: str) -> dict[str, Any]:
    """Recommend partition key(s) from a dataset's cardinality profile (DQ-LAKE-010).

    Profiles per-column distinct-value counts and recommends columns whose
    cardinality is LOW enough to make well-sized, prunable partitions (distinct
    count ≤ :data:`_MAX_PARTITION_CARDINALITY` AND distinct/rows ratio below
    :data:`_MAX_CARDINALITY_RATIO`, so id-like near-unique columns are rejected).
    Deterministic (pure pandas profile), no live AWS.

    Args:
        data_path: Filesystem path to the source CSV/Parquet to profile.

    Returns:
        ``{"recommended_keys","column_cardinality","candidates","rows","rationale"}``
        on success, or ``{"success": False, "error": ...}``.
    """
    p = Path(data_path)
    if not p.exists():
        return {"success": False, "error": f"File not found: {data_path}"}
    try:
        df = _read_frame(p)
    except Exception as e:
        return {"success": False, "error": f"Failed to read dataset: {e}"}

    rows = len(df)
    cardinality = {c: int(df[c].nunique(dropna=False)) for c in df.columns}

    candidates: list[dict[str, Any]] = []
    for col, distinct in cardinality.items():
        if distinct < 2:
            continue  # constant column → a single partition (useless)
        ratio = distinct / rows if rows else 1.0
        # Prefer non-numeric (categorical/era/region) keys; a numeric metric column
        # is a poor partition key even when its cardinality happens to be low.
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if distinct <= _MAX_PARTITION_CARDINALITY and ratio <= _MAX_CARDINALITY_RATIO and not is_numeric:
            candidates.append({"column": col, "cardinality": distinct, "ratio": round(ratio, 4)})

    # Rank by ascending cardinality (fewer, larger partitions first) then name for
    # a stable, deterministic recommendation; cap at the two best keys.
    candidates.sort(key=lambda c: (c["cardinality"], c["column"]))
    recommended_keys = [c["column"] for c in candidates[:2]]

    if recommended_keys:
        rationale = (
            f"Recommend partitioning by {recommended_keys}: low-cardinality "
            f"categorical key(s) (≤{_MAX_PARTITION_CARDINALITY} distinct values, "
            f"below {int(_MAX_CARDINALITY_RATIO * 100)}% distinct/row ratio) that "
            "yield well-sized, prunable partitions without tiny-file explosion."
        )
    else:
        rationale = (
            "No suitable partition key found — every column is either near-unique "
            "(id-like), numeric metric, or constant. Consider a derived low-"
            "cardinality bucket (e.g. a date/era band) before partitioning."
        )

    return {
        "recommended_keys": recommended_keys,
        "column_cardinality": cardinality,
        "candidates": candidates,
        "rows": rows,
        "rationale": rationale,
    }
