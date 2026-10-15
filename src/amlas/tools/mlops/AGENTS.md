# src/amlas/tools/mlops

## Purpose

Expose managed ML registry, pipeline, feature-store, explainability and monitoring operations.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `_common.py:sanitize_name` — Coerce an arbitrary string into a valid SageMaker resource name.
- `_common.py:make_sm_client` — Return an injected SageMaker client, else build a default boto3 one.
- `clarify.py:build_clarify_config` — Assemble + validate a Clarify analysis config (pure, no AWS).
- `clarify.py:run_clarify` — Run a SageMaker Clarify bias + SHAP analysis (master plan §8 D4).
- `feature_store.py:infer_feature_definitions` — Map pandas dtypes → SageMaker FeatureDefinitions (pure, no AWS).
- `feature_store.py:sanitize_feature_name` — Coerce a column name to a valid SageMaker FeatureName (pure, no AWS).
- Owned files: `__init__.py`, `_common.py`, `clarify.py`, `feature_store.py`, `kpis.py`, `monitor.py`, `pipelines.py`, `registry.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
