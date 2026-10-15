# src/amlas/sagemaker_entrypoints

## Purpose

Run managed tabular training/processing entrypoints and write their expected result artifacts.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Honor managed training input/output channels and write the expected summary/prediction artifacts.
- Runtime credentials come from the managed role chain, not a developer profile name.
- `tabular_train.py:run_training` — Fit ``algo`` on the full train split, predict the test split, write y_pred_test.npy.
- `tabular_train.py:resolve_sagemaker_paths` — Resolve the train/test dirs + output dir from SageMaker's SM_* env vars.
- Owned files: `__init__.py`, `tabular_train.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
