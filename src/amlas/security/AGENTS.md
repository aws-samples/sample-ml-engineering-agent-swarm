# src/amlas/security

## Purpose

Enforce shared model-artifact loading boundaries across application layers.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Load model artifacts only through the declared validated boundaries in `model_artifacts.py`.
- Do not weaken untrusted-artifact checks to make a model deserialize.
- `model_artifacts.py:UntrustedArtifactError` — Raised when an estimator artifact fails a trust-boundary check.
- `model_artifacts.py:dump_model_artifact` — Persist ``model`` in Skops format and return the resolved artifact path.
- Owned files: `__init__.py`, `model_artifacts.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
