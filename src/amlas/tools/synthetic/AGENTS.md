# src/amlas/tools/synthetic

## Purpose

Generate or augment rows and evaluate fidelity, utility and privacy evidence.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Keep generation and validation separate; fidelity alone is insufficient for acceptance.
- Preserve the existing fidelity, utility and privacy checks and domain-steering constraints.
- `_common.py:canonical_generator` — Map a generator alias to a canonical SDV generator name (identity if canonical).
- `_common.py:synthetic_output_path` — Deterministic write-back path next to the input CSV.
- `generate.py:synthesize_tabular_frame` — Core SDV synthesis (plain function reused by the agent + validator).
- `generate.py:synthesize_tabular` — Generate realistic synthetic rows from a reference dataset via SDV (master plan §7).
- Owned files: `__init__.py`, `_common.py`, `generate.py`, `llm_rows.py`, `requirement.py`, `steering.py`, `validate.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
