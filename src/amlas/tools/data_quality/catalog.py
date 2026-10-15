"""Machine-readable Data-Quality pattern catalog — master plan §5.

This is the *validation contract* referenced throughout the initiative
("does it check these patterns as it goes along"). It declares the 25 patterns
across 5 families as a structured, immutable registry so that:

  * ``patterns.py`` detectors map 1:1 to a ``pattern_id`` here,
  * downstream phases (imputation / synthetic) read ``suggested_strategy``,
  * the validation phase (§12.1) can iterate the registry to assert ≥0.95
    precision/recall over a fixture corpus.

Each :class:`PatternSpec` is metadata only — the actual detection logic lives in
``patterns.py`` (deterministic, CPU/pandas). ``severity_default`` is the severity
a detector emits unless evidence escalates it.

Families (counts must stay 6/5/5/4/5 = 25):
  * Missingness (6)            — MISS-*
  * Validity (5)               — VAL-*
  * Uniqueness/consistency (5) — UNQ-* / CON-*
  * Distribution/leakage/drift (4) — DST-* / LEAK-*
  * Long-lineage/OCR (5)       — OCR-* / LIN-*
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Severity = Literal["low", "medium", "high"]

# ─── Family identifiers ──────────────────────────────────────────────────────
FAMILY_MISSINGNESS = "missingness"
FAMILY_VALIDITY = "validity"
FAMILY_UNIQUENESS = "uniqueness_consistency"
FAMILY_DISTRIBUTION = "distribution_leakage_drift"
FAMILY_OCR_LINEAGE = "long_lineage_ocr"
FAMILY_CONTENT_SAFETY = "content_safety"

FAMILIES: tuple[str, ...] = (
    FAMILY_MISSINGNESS,
    FAMILY_VALIDITY,
    FAMILY_UNIQUENESS,
    FAMILY_DISTRIBUTION,
    FAMILY_OCR_LINEAGE,
    FAMILY_CONTENT_SAFETY,
)


@dataclass(frozen=True)
class PatternSpec:
    """A single catalog entry (metadata only — detection lives in patterns.py)."""

    pattern_id: str
    family: str
    title: str
    description: str
    severity_default: Severity
    suggested_strategy: str  # a method name the imputation / synthetic agents understand

    def to_dict(self) -> dict[str, str]:
        return {
            "pattern_id": self.pattern_id,
            "family": self.family,
            "title": self.title,
            "description": self.description,
            "severity_default": self.severity_default,
            "suggested_strategy": self.suggested_strategy,
        }


# ─── The 25-pattern catalog (ordered by family) ──────────────────────────────
_CATALOG: tuple[PatternSpec, ...] = (
    # ── Missingness (6) ──────────────────────────────────────────────────────
    PatternSpec(
        "MISS-MCAR", FAMILY_MISSINGNESS, "Missing Completely At Random",
        "Null mask is independent of all columns (Little's test p>0.05, mask-AUC≈0.5).",
        "low", "any_imputer",
    ),
    PatternSpec(
        "MISS-MAR", FAMILY_MISSINGNESS, "Missing At Random",
        "Null mask is predictable from other observed columns (mask-AUC>0.6).",
        "medium", "mice_missforest_knn",
    ),
    PatternSpec(
        "MISS-MNAR", FAMILY_MISSINGNESS, "Missing Not At Random",
        "Missingness depends on the unobserved value itself (value-dependent gaps).",
        "high", "pattern_mixture_plus_indicator",
    ),
    PatternSpec(
        "MISS-RUN", FAMILY_MISSINGNESS, "Missing Runs Over Time",
        "Long contiguous runs of nulls in a time-ordered column (sensor dropout).",
        "medium", "time_interp_plus_flag",
    ),
    PatternSpec(
        "MISS-BLOCK", FAMILY_MISSINGNESS, "Block / Era Missingness",
        "Null density concentrated in a contiguous block of rows (an era never recorded it).",
        "medium", "per_era_impute",
    ),
    PatternSpec(
        "MISS-SENTINEL", FAMILY_MISSINGNESS, "Disguised / Sentinel Missing",
        "Missing encoded as a sentinel value (-999, 'N/A', 'unknown') rather than NaN.",
        "high", "normalize_then_impute",
    ),
    # ── Validity (5) ──────────────────────────────────────────────────────────
    PatternSpec(
        "VAL-TYPE", FAMILY_VALIDITY, "Mixed-Type Column",
        "An object column mixes numeric-like and text values (type inconsistency).",
        "medium", "coerce_or_split",
    ),
    PatternSpec(
        "VAL-RANGE", FAMILY_VALIDITY, "Out-of-Range Values",
        "Numeric values fall outside a physically plausible range (e.g. negative age).",
        "high", "cap_or_null_then_impute",
    ),
    PatternSpec(
        "VAL-INF", FAMILY_VALIDITY, "Infinite Values",
        "Column contains +/-inf, typically from division-by-zero or overflow.",
        "high", "replace_with_bounds",
    ),
    PatternSpec(
        "VAL-FORMAT", FAMILY_VALIDITY, "Format / Enum Violation",
        "String values violate an expected date/regex/enum format.",
        "medium", "normalize_format",
    ),
    PatternSpec(
        "VAL-UNIT", FAMILY_VALIDITY, "Unit Drift",
        "A numeric column shows multi-modal scale consistent with imperial<->metric mixing.",
        "high", "unit_normalize",
    ),
    # ── Uniqueness / consistency (5) ─────────────────────────────────────────
    PatternSpec(
        "UNQ-DUP", FAMILY_UNIQUENESS, "Exact Duplicate Rows",
        "Fully duplicated rows that bias training and inflate counts.",
        "medium", "drop_duplicates",
    ),
    PatternSpec(
        "UNQ-KEY", FAMILY_UNIQUENESS, "Duplicate Primary Key",
        "A declared/identifier key column has repeated values (broken uniqueness).",
        "high", "resolve_key_collision",
    ),
    PatternSpec(
        "UNQ-FUZZY", FAMILY_UNIQUENESS, "Near-Duplicate Strings",
        "High string-similarity categories that are the same entity ('ACME' vs 'ACME.').",
        "low", "fuzzy_merge",
    ),
    PatternSpec(
        "CON-REF", FAMILY_UNIQUENESS, "Cross-Field Rule Break",
        "A deterministic cross-column relationship is violated (a<=b broken).",
        "medium", "reconcile_fields",
    ),
    PatternSpec(
        "CON-TEMP", FAMILY_UNIQUENESS, "Temporal Inconsistency",
        "Future-dated timestamps or non-monotonic time series where monotonic expected.",
        "medium", "repair_or_flag_temporal",
    ),
    # ── Distribution / leakage / drift (4) ───────────────────────────────────
    PatternSpec(
        "DST-OUT", FAMILY_DISTRIBUTION, "Distributional Outliers",
        "Extreme values far from the bulk distribution (reuse detect_anomalies).",
        "medium", "cap_or_remove",
    ),
    PatternSpec(
        "DST-IMB", FAMILY_DISTRIBUTION, "Class Imbalance",
        "Severe target class imbalance harming minority recall.",
        "medium", "smote_or_class_weight",
    ),
    PatternSpec(
        "DST-DRIFT", FAMILY_DISTRIBUTION, "Era Distribution Drift",
        "A feature's distribution shifts across era bins (KS/PSI across time).",
        "high", "era_aware_modeling",
    ),
    PatternSpec(
        "LEAK-CARD", FAMILY_DISTRIBUTION, "Leakage / Identifier Column",
        "Identifier-like or near-unique column that leaks the target (reuse leakage_columns).",
        "high", "drop_or_quarantine",
    ),
    # ── Long-lineage / OCR — Amlas specific (5) ─────────────────────────────
    PatternSpec(
        "OCR-CONF", FAMILY_OCR_LINEAGE, "OCR Character Confusion",
        "Classic OCR confusions in text/IDs: O<->0, l<->1, S<->5, B<->8.",
        "high", "correct_or_flag_ocr",
    ),
    PatternSpec(
        "OCR-NOISE", FAMILY_OCR_LINEAGE, "OCR Stray Noise",
        "Stray punctuation / broken decimals from scanning ('1.2.3', '4,5,6', 'a|b').",
        "medium", "clean_ocr_noise",
    ),
    PatternSpec(
        "OCR-CASE", FAMILY_OCR_LINEAGE, "Casing / Whitespace Variants",
        "Same category appears under mixed casing / stray whitespace (' Cat', 'CAT ').",
        "low", "canonicalize_case",
    ),
    PatternSpec(
        "LIN-SCHEMA", FAMILY_OCR_LINEAGE, "Schema Drift Across Eras",
        "Column present only in a contiguous era (added/removed over a 60+ year lineage).",
        "high", "era_schema_mapping",
    ),
    PatternSpec(
        "LIN-ENCODE", FAMILY_OCR_LINEAGE, "Encoding Inconsistency",
        "Non-ASCII / mojibake / mixed encodings introduced across decades of digitization.",
        "medium", "normalize_encoding",
    ),
    # ── Content Safety (2) — SAFE-* (cloud-only, uses Amazon Comprehend) ──────
    PatternSpec(
        "SAFE-TOX", FAMILY_CONTENT_SAFETY, "Toxic / Harmful Content",
        "Text column contains toxic, threatening, insulting, or hateful language "
        "that a model could learn and reproduce.",
        "high", "redact_or_remove_toxic",
    ),
    PatternSpec(
        "SAFE-PII", FAMILY_CONTENT_SAFETY, "Personally Identifiable Information (PII)",
        "Text column contains PII (names, emails, phone numbers, SSNs, credit cards) "
        "that should be redacted before training.",
        "high", "redact_pii",
    ),
)

# Public registry keyed by id (insertion-ordered, immutable view via tuple above).
PATTERN_CATALOG: dict[str, PatternSpec] = {p.pattern_id: p for p in _CATALOG}


# ─── Convenience accessors (used by tools, agents, validation phase) ─────────
def all_pattern_ids() -> list[str]:
    """Return all pattern ids in catalog order."""
    return [p.pattern_id for p in _CATALOG]


def get_pattern(pattern_id: str) -> PatternSpec:
    """Return the :class:`PatternSpec` for ``pattern_id`` (KeyError if unknown)."""
    return PATTERN_CATALOG[pattern_id]


def patterns_in_family(family: str) -> list[PatternSpec]:
    """Return all catalog entries belonging to ``family`` (catalog order)."""
    return [p for p in _CATALOG if p.family == family]
