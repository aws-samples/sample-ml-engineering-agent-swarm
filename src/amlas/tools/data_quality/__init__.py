"""Data Quality Assessment subpackage (Capability Group A — Phase 6).

Implements the master-plan §5 validation contract: a machine-readable
25-pattern catalog (5 families), deterministic per-pattern detectors, the
``scan_dq_patterns`` aggregation tool, MCAR/MAR/MNAR missingness diagnosis
(in-house Little's MCAR chi-square + sklearn mask classifier), a pandera
expectations gate, OCR / long-lineage artifact detection, and a 6-dimension
data-quality scorecard.

Conventions (match ``tools/data_quality_tools.py``):
  - Tools are ``@tool``-decorated, path-based, return a JSON-serializable dict,
    and fail soft with ``{"success": False, "error": "..."}``.
  - Optional heavy deps (pandera) are imported lazily inside the function so the
    package still imports when the ``data`` extra is absent.

The catalog is exported as a structured registry so later phases (imputation,
synthetic) and the validation phase can iterate over every pattern.
"""

from __future__ import annotations

from .catalog import (
    FAMILIES,
    FAMILY_DISTRIBUTION,
    FAMILY_MISSINGNESS,
    FAMILY_OCR_LINEAGE,
    FAMILY_UNIQUENESS,
    FAMILY_VALIDITY,
    PATTERN_CATALOG,
    PatternSpec,
    all_pattern_ids,
    get_pattern,
    patterns_in_family,
)
from .missingness import check_expectations, diagnose_missingness
from .patterns import detect_ocr_artifacts, scan_dq_patterns
from .score import score_data_quality_6dim

__all__ = [
    "FAMILIES",
    "FAMILY_DISTRIBUTION",
    "FAMILY_MISSINGNESS",
    "FAMILY_OCR_LINEAGE",
    "FAMILY_UNIQUENESS",
    "FAMILY_VALIDITY",
    "PATTERN_CATALOG",
    # catalog
    "PatternSpec",
    "all_pattern_ids",
    "check_expectations",
    "detect_ocr_artifacts",
    "diagnose_missingness",
    "get_pattern",
    "patterns_in_family",
    # tools
    "scan_dq_patterns",
    "score_data_quality_6dim",
]
