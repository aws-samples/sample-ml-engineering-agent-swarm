"""Data verification — MetaGPT DI CheckData port.

Ritual applied before any data-touching task: print shape, dtypes, NaN%, head,
basic stats. Returns a DataProfile that agents can reason about without loading
the whole frame into their context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    nan_pct: float
    nunique: int
    sample: list[Any] = field(default_factory=list)
    numeric_stats: dict[str, float] | None = None


@dataclass
class DataProfile:
    n_rows: int
    n_cols: int
    columns: list[ColumnProfile] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"Shape: ({self.n_rows}, {self.n_cols})", "", "| col | dtype | nan% | nunique | sample |", "|---|---|---|---|---|"]
        for c in self.columns:
            sample_s = ", ".join(str(s) for s in c.sample[:3])
            lines.append(f"| {c.name} | {c.dtype} | {c.nan_pct:.2f}% | {c.nunique} | {sample_s} |")
        return "\n".join(lines)


def data_verify(df: pd.DataFrame, sample_size: int = 3) -> DataProfile:
    """Produce a compact profile of a DataFrame for agent consumption."""
    profile = DataProfile(n_rows=len(df), n_cols=df.shape[1])
    for col in df.columns:
        s = df[col]
        nan_pct = float(s.isna().mean() * 100)
        sample = s.dropna().head(sample_size).tolist()
        numeric_stats: dict[str, float] | None = None
        if pd.api.types.is_numeric_dtype(s) and len(s.dropna()) > 0:
            numeric_stats = {
                "min": float(s.min()),
                "max": float(s.max()),
                "mean": float(s.mean()),
                "std": float(s.std()) if len(s.dropna()) > 1 else 0.0,
            }
        profile.columns.append(
            ColumnProfile(
                name=str(col),
                dtype=str(s.dtype),
                nan_pct=nan_pct,
                nunique=int(s.nunique(dropna=True)),
                sample=sample,
                numeric_stats=numeric_stats,
            )
        )
    return profile
