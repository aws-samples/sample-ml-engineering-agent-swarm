# src/amlas/subagents

## Purpose

Declare bounded specialist agents and their reusable tools/prompts.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Keep specialist tool/prompt declarations centralized and bounded; wrappers must reflect those declarations.
- `business_goal_advisor.py:build_business_goal_advisor` — Build the business_goal_advisor sub-agent (R1.a) — Stage-1 problem/KPI/go-no-go reasoning.
- `compute_placement.py:agent_tool_names` — Return the tool names bound to a Strands ``Agent`` (best-effort, version-tolerant).
- `data_collection_agent.py:build_data_collection_agent` — Build the data_collection_agent sub-agent (DE.a) — Stage-4 lake inventory via QueryBackend (read).
- `data_integrity.py:build_data_integrity_agent` — Build the Data Integrity sub-agent.
- `declarations.py:tool_name_of` — The name a ``@tool`` object registers itself under.
- `declarations.py:tool_contract_of` — ``(name, description, input_schema)`` for a ``@tool``, readable offline.
- Owned files: `__init__.py`, `algo_hypothesis.py`, `algorithm_specialist.py`, `business_goal_advisor.py`, `compute_placement.py`, `data_collection_agent.py`, `data_integrity.py`, `declarations.py`, `deployment_agent.py`, `devops_git_agent.py`, `dq_assessment.py`, `feature_hypothesis.py`, `fine_tune.py`, `framing.py`, `governance_agent.py`, `improvement_agent.py`, `imputation_strategist.py`, `labeling_strategist.py`, `llm_extrapolation.py`, `monitoring_agent.py`, `problem_framing_advisor.py`, `readiness_review_agent.py`, `registry.py`, `research_curator.py`, `research_discovery.py`, `research_scout.py`, `sagemaker_ops.py`, `solution_fit_advisor.py`, `solution_survey_agent.py`, `spec_writer.py`, `synthetic_data.py`, `tabular_fe.py`, `trainer_tuner.py`, `vision_fe.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

| Path | Covers |
| --- | --- |
| `fine_tune_a2a/AGENTS.md` | Expose the separate managed fine-tuning sub-runtime and its invocation boundary. |
