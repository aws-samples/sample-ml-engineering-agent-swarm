"""AI4I 2020 Predictive Maintenance — tabular TDD-fast golden dataset.

UCI ML Repository, CC-BY-4.0. ~10k rows, 14 features, 5 failure modes (TWF/HDF/PWF/OSF/RNF).
Baseline: XGBoost default ≈ F1-macro 0.96.

We ship tiny_ai4i.csv (500 rows) as a sample fixture for sub-5-second unit-test cycles.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

AI4I_TARGET = "failure_mode"  # synthetic collapsed target from TWF/HDF/PWF/OSF/RNF columns
AI4I_NUMERIC_FEATURES = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
AI4I_CATEGORICAL_FEATURES = ["type"]  # L/M/H quality grade


@dataclass(frozen=True)
class SplitManifest:
    """Frozen split description — Arbiter verifies agent didn't fit on held-out data."""

    n_train: int
    n_test: int
    train_hash: str
    test_hash: str
    target: str


def collapse_failure_modes(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse 5 binary failure-mode columns into one multi-class target.

    If multiple modes active, picks the first in priority order. If none, target = 0 (no failure).
    """
    priority = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    out = df.copy()
    target_col = pd.Series(0, index=df.index, name=AI4I_TARGET)
    for i, col in enumerate(priority, start=1):
        if col in out.columns:
            mask = (out[col].astype(int) == 1) & (target_col == 0)
            target_col[mask] = i
    out[AI4I_TARGET] = target_col
    for col in priority:
        if col in out.columns:
            out = out.drop(columns=[col])
    return out


def load_ai4i(path: str | Path) -> pd.DataFrame:
    """Load a local AI4I csv (tiny fixture or full set) and collapse failure modes."""
    df = pd.read_csv(path)
    # normalise headers — tolerate UCI casing
    df.columns = [c.strip().lower().replace(" ", "_").replace("[", "").replace("]", "") for c in df.columns]
    # common renames
    renames = {
        "air_temperature_k": "air_temperature_k",
        "process_temperature_k": "process_temperature_k",
        "rotational_speed_rpm": "rotational_speed_rpm",
        "torque_nm": "torque_nm",
        "tool_wear_min": "tool_wear_min",
        "machine_failure": "machine_failure",
        "twf": "TWF",
        "hdf": "HDF",
        "pwf": "PWF",
        "osf": "OSF",
        "rnf": "RNF",
    }
    df = df.rename(columns=renames)
    df = collapse_failure_modes(df)
    return df


def split_ai4i(
    df: pd.DataFrame, *, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, SplitManifest]:
    """Stratified train/test split with a deterministic manifest hash."""
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[AI4I_TARGET]
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    return train, test, SplitManifest(
        n_train=len(train),
        n_test=len(test),
        train_hash=_hash_df(train),
        test_hash=_hash_df(test),
        target=AI4I_TARGET,
    )


def _hash_df(df: pd.DataFrame) -> str:
    """Return a deterministic SHA-256 digest over row-stable DataFrame bytes."""
    b: Any = pd.util.hash_pandas_object(df, index=False).values.tobytes()
    return hashlib.sha256(b).hexdigest()
