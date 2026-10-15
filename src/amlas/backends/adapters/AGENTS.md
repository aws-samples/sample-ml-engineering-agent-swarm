# src/amlas/backends/adapters

## Purpose

Implement and register concrete local and managed-service backend adapters.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Import registration belongs in `__init__.py`; preserve adapter names and declared port conformance.
- Propagate service failures and retain accepted native handles; configuration/Region comes through the shared selectors.
- `_util.py:raw_tool` — Return the undecorated callable behind a Strands ``@tool`` (or the fn itself).
- `automl.py:LocalSweepBackend` — Local/offline AutoMLBackend — a tiny seeded quick-sweep over the ``train_model`` IP.
- `automl.py:AutopilotV2Backend` — Managed AutoMLBackend — orchestrates SageMaker Autopilot / AutoML v2 (T-AML-01).
- `bedrock_alt.py:BedrockFineTuneBackend` — Bedrock-native ``TrainingBackend`` — orchestrates a model-customization (fine-tune) job.
- `bedrock_alt.py:BedrockHostedBackend` — Bedrock-native ``DeploymentBackend`` — hosts via provisioned-throughput (not a SM endpoint).
- Owned files: `__init__.py`, `_util.py`, `automl.py`, `bedrock_alt.py`, `dataquality.py`, `deployment.py`, `etl.py`, `evaluation.py`, `explainability.py`, `featurestore.py`, `governance.py`, `labeling.py`, `lake.py`, `modelhub.py`, `monitoring.py`, `orchestration.py`, `orchestration_pipeline.py`, `pipeline.py`, `query.py`, `registry_be.py`, `retrieval.py`, `stubs.py`, `tracking.py`, `training.py`, `tuning.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
