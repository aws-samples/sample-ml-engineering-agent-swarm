# prompts/leads

## Purpose

Declare and build the seven lifecycle Lead agents and their shared tool/prompt contracts.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Keep tool/prompt declarations in `declarations.py`; wrappers must not silently add capabilities or change guardrail behavior.
- Owned files: `data_engineering_lead.md`, `deploy_serving_lead.md`, `fe_lead.md`, `ml_execution_lead.md`, `mlops_lead.md`, `monitoring_lead.md`, `pipeline_orchestrator.md`, `research_lead.md`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
