"""Strands @tool wrappers over our pure-Python tool library.

The pure implementations in `leakage_check.py` / `data_verify.py` stay agent-agnostic —
they take DataFrames and return dataclasses. This module wraps them into file-path-based
tools suitable for agent use (agents pass paths + scalars, never DataFrames).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from strands import tool

from .data_verify import data_verify
from .leakage_check import leakage_check


def _read_any(path: str) -> pd.DataFrame:
    """Load parquet, csv, or json based on extension."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"no such file: {path}")
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(p, lines=(suffix == ".jsonl"))
    raise ValueError(f"unsupported extension: {suffix}")


@tool
def data_verify_tool(path: str) -> dict[str, Any]:
    """Produce a compact data profile (shape, dtypes, NaN%, sample) for agent consumption.

    Args:
        path: Filesystem path to a parquet / csv / json(l) frame.
    """
    df = _read_any(path)
    profile = data_verify(df)
    return {
        "n_rows": profile.n_rows,
        "n_cols": profile.n_cols,
        "columns": [
            {
                "name": c.name,
                "dtype": c.dtype,
                "nan_pct": round(c.nan_pct, 3),
                "nunique": c.nunique,
                "sample": c.sample[:3],
                "numeric_stats": c.numeric_stats,
            }
            for c in profile.columns
        ],
    }


@tool
def leakage_check_tool(
    train_path: str,
    test_path: str,
    target: str,
    split_key: str | None = None,
    time_col: str | None = None,
    cutoff: str | None = None,
) -> dict[str, Any]:
    """Run leakage checks against train/test frames.

    Returns a dict with `blocking: bool` and a list of `findings`. The calling agent
    MUST treat `blocking=True` as a pipeline halt — no modeling proceeds.

    Args:
        train_path: path to the training frame (parquet/csv/json).
        test_path: path to the held-out test frame.
        target: name of the target column.
        split_key: optional primary key column for split-dup detection.
        time_col: optional timestamp column for time-order checks.
        cutoff: ISO timestamp required when time_col is provided.
    """
    train = _read_any(train_path)
    test = _read_any(test_path)
    report = leakage_check(
        train=train,
        test=test,
        target=target,
        split_key=split_key,
        time_col=time_col,
        cutoff=cutoff,
    )
    return {
        "blocking": report.blocking,
        "summary": report.summary(),
        "findings": [
            {"kind": f.kind, "severity": f.severity, "detail": f.detail}
            for f in report.findings
        ],
    }
