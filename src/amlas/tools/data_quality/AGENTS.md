# src/amlas/tools/data_quality

## Purpose

Profile data quality through the catalog, patterns, missingness analysis and scorecard.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `catalog.py:PatternSpec` — A single catalog entry (metadata only — detection lives in patterns.py).
- `catalog.py:all_pattern_ids` — Return all pattern ids in catalog order.
- `missingness.py:little_mcar_test` — In-house Little's MCAR test over the numeric columns of ``df``.
- `missingness.py:diagnose_missingness` — Diagnose the missingness mechanism (MCAR / MAR / MNAR) per column.
- `patterns.py:scan_dq_patterns` — Run the 25-pattern DQ catalog detectors over a CSV and return findings.
- `patterns.py:detect_ocr_artifacts` — Detect OCR / scanning artifacts in text columns of a CSV.
- Owned files: `__init__.py`, `catalog.py`, `missingness.py`, `patterns.py`, `score.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
