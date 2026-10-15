# src/amlas/flows

## Purpose

Declare and register reusable ML lifecycle flows with execution and approval metadata.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Registered flow identifiers and spend/irreversibility metadata are dispatch contracts.
- Use typed backend ports; report execution evidence and grading separately from a workflow status.
- Solution survey derives omitted text modality for generation tasks and rejects
  an explicit contradictory modality before catalog calls.
- `__init__.py:Flow` — A registered flow descriptor (``flow_id`` → metadata).
- `__init__.py:register_flow` — Register ``flow`` under its ``flow_id`` (idempotent — last write wins).
- `_util.py:raw` — Return the undecorated callable behind a Strands ``@tool`` (or the fn itself).
- `baseline_train_flow.py:BaselineTrainFlowResult` — The invariant output contract of one ``baseline_train_flow`` run.
- Owned files: `__init__.py`, `_util.py`, `baseline_train_flow.py`, `business_framing_flow.py`, `data_collection_flow.py`, `data_quality_flow.py`, `deploy_flow.py`, `devops_git_flow.py`, `evaluation_flow.py`, `feature_engineering_flow.py`, `hpo_flow.py`, `improve_retrain_flow.py`, `imputation_flow.py`, `monitor_flow.py`, `readiness_flow.py`, `research_intel_flow.py`, `research_scan_flow.py`, `solution_survey_flow.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
