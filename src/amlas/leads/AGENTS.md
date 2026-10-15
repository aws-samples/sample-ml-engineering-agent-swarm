# src/amlas/leads

## Purpose

Declare and build the seven lifecycle Lead agents and their shared tool/prompt contracts.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Keep tool/prompt declarations in `declarations.py`; wrappers must not silently add capabilities or change guardrail behavior.
- `data_eng.py:build_data_engineering_lead` — Build the Data Engineering Lead (D0) agent.
- `declarations.py:declared_tool_contract` — Return ``(description, input_schema)`` for a directly-bound ``@tool`` object.
- `declarations.py:declared_tool_name` — Return the name a directly-bound ``@tool`` object registers itself under.
- `deploy_serving_lead.py:build_deploy_serving_lead` — Build the Deploy/Serving Lead (D3) agent.
- `feature_eng.py:build_fe_lead` — Build the FE Lead (D1) agent.
- Owned files: `__init__.py`, `data_eng.py`, `declarations.py`, `deploy_serving_lead.py`, `feature_eng.py`, `ml_execution.py`, `mlops_lead.py`, `monitoring_lead.py`, `registry.py`, `research_lead.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
