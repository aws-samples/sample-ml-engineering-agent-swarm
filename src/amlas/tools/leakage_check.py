"""Leakage sentinel — ml-ralph Data Verification Checklist + MLE-STAR leakage-check.

Blocks the pipeline before training if any of the following are detected:
    1. Rows from the same primary key appear in BOTH train and test (split key dup).
    2. Target column leaks into features (target-bleed — feature == target, or r > 0.99).
    3. For time_based splits: any train timestamp >= cutoff (time-order violation).
    4. Duplicate row-hashes across train and test (near-identical instances).

Returns a LeakageReport with `blocking=True` if any violation found. FE Lead treats
blocking report as a hard stop — no modeling proceeds until fixed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


@dataclass
class LeakageFinding:
    kind: Literal["split_key_dup", "target_bleed", "time_order", "row_hash_dup"]
    detail: str
    severity: Literal["block", "warn"] = "block"


@dataclass
class LeakageReport:
    findings: list[LeakageFinding] = field(default_factory=list)

    @property
    def blocking(self) -> bool:
        return any(f.severity == "block" for f in self.findings)

    def summary(self) -> str:
        if not self.findings:
            return "no leakage detected"
        return "; ".join(f"[{f.severity}/{f.kind}] {f.detail}" for f in self.findings)


def _row_hash(df: pd.DataFrame) -> pd.Series:
    """Return stable SHA-256 row fingerprints for duplicate detection."""
    serialized = df.astype(str).agg("|".join, axis=1)
    return serialized.apply(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())


def leakage_check(
    *,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    split_key: str | None = None,
    time_col: str | None = None,
    cutoff: str | None = None,
    target_corr_threshold: float = 0.99,
) -> LeakageReport:
    """Run all leakage checks against train/test frames.

    Args:
        train: training frame, must include `target`.
        test: held-out frame, must include `target` (label may be masked in production
            but we need it to detect split-key dupes).
        target: name of the target column.
        split_key: column name identifying primary key for split-dup check. Optional.
        time_col: column name for time-order check when using time_based splits.
        cutoff: ISO timestamp (for time_based splits); train `time_col` must be < cutoff.
    """
    report = LeakageReport()

    for df, name in ((train, "train"), (test, "test")):
        if target not in df.columns:
            report.findings.append(
                LeakageFinding("target_bleed", f"target {target!r} missing from {name}")
            )
            return report

    # (1) split key dup
    if split_key and split_key in train.columns and split_key in test.columns:
        dup = set(train[split_key]) & set(test[split_key])
        if dup:
            report.findings.append(
                LeakageFinding(
                    "split_key_dup",
                    f"{len(dup)} keys appear in both train and test (examples: {list(dup)[:3]})",
                )
            )

    # (2) target bleed — identical column, or near-perfect correlation
    for col in train.columns:
        if col == target:
            continue
        if train[col].equals(train[target]):
            report.findings.append(
                LeakageFinding("target_bleed", f"feature {col!r} == target column")
            )
        elif pd.api.types.is_numeric_dtype(train[col]) and pd.api.types.is_numeric_dtype(
            train[target]
        ):
            corr = train[col].corr(train[target])
            if pd.notna(corr) and abs(corr) >= target_corr_threshold:
                report.findings.append(
                    LeakageFinding(
                        "target_bleed",
                        f"feature {col!r} has |corr|={abs(corr):.3f} with target (>= {target_corr_threshold})",
                    )
                )

    # (3) time order — train timestamps must all be < cutoff
    if time_col and cutoff and time_col in train.columns:
        tr_times = pd.to_datetime(train[time_col], errors="coerce")
        co = pd.to_datetime(cutoff)
        violations = (tr_times >= co).sum()
        if violations > 0:
            report.findings.append(
                LeakageFinding(
                    "time_order",
                    f"{violations} train rows have {time_col} >= cutoff {cutoff}",
                )
            )

    # (4) row hash dup — identical rows (excluding target) across splits
    feature_cols = [c for c in train.columns if c != target and c in test.columns]
    if feature_cols:
        tr_hash = _row_hash(train[feature_cols])
        te_hash = _row_hash(test[feature_cols])
        dup_count = tr_hash.isin(te_hash).sum()
        if dup_count > 0:
            report.findings.append(
                LeakageFinding(
                    "row_hash_dup",
                    f"{dup_count} feature-identical rows appear in both splits",
                )
            )

    return report
