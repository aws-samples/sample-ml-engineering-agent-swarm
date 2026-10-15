# src/amlas/tools/imputation

## Purpose

Select and evaluate numeric, categorical, temporal and LLM-assisted missing-value methods.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `_common.py:read_csv_or_error` — Load a CSV, returning ``(df, None)`` on success or ``(None, error_dict)``.
- `categorical.py:impute_categorical_frame` — Impute categorical ``columns``; return (frame, warnings, n_filled).
- `categorical.py:impute_categorical` — Impute missing categorical / discrete-text values (master plan §6).
- `engine.py:recommend_imputation_plan` — Recommend a per-column imputation method using mechanism diagnosis + steering.
- `evaluate.py:evaluate_methods` — Score each candidate method's recovery of masked-known cells for ``column``.
- `evaluate.py:evaluate_imputation` — Mask-and-score self-evaluation: rank imputation methods by recovery.
- The direct Bedrock extrapolation caller resolves the configured bounded role,
  filters sampling fields through shared model compatibility, and returns only
  visible response text.
- Owned files: `__init__.py`, `_common.py`, `categorical.py`, `engine.py`, `evaluate.py`, `llm_extrapolate.py`, `numeric.py`, `timeseries.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
