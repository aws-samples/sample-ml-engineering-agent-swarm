# src/amlas/golden

## Purpose

Load evaluation fixtures and retain frozen train/test split manifests for repeatable grading.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Preserve deterministic split assignments and manifest identities; changing a split changes the evaluation oracle.
- `ai4i.py:SplitManifest` — Frozen split description — Arbiter verifies agent didn't fit on held-out data.
- `ai4i.py:collapse_failure_modes` — Collapse 5 binary failure-mode columns into one multi-class target.
- `railsem19.py:load_railsem19` — Index a local RailSem19 unpacked directory.
- Owned files: `__init__.py`, `ai4i.py`, `railsem19.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
