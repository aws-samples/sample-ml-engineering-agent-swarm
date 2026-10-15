# prompts/subagents

## Purpose

Declare bounded specialist agents and their reusable tools/prompts.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Keep specialist tool/prompt declarations centralized and bounded; wrappers must reflect those declarations.
- Spec Writer preserves generation intent and supplied metadata, and labels
  unmeasured baselines instead of substituting classification or fabricated scores.
- Owned files: `algo_specialist.md`, `business_goal_advisor.md`, `data_collection_agent.md`, `data_integrity.md`, `deployment_agent.md`, `devops_git_agent.md`, `dq_assessment.md`, `fine_tune.md`, `governance_agent.md`, `improvement_agent.md`, `imputation_strategist.md`, `labeling_strategist.md`, `llm_extrapolation.md`, `monitoring_agent.md`, `problem_framing_advisor.md`, `readiness_review_agent.md`, `research_curator.md`, `research_discovery.md`, `research_scout.md`, `sagemaker_ops.md`, `solution_fit_advisor.md`, `solution_survey_agent.md`, `spec_writer.md`, `synthetic_data.md`, `tabular_fe.md`, `trainer_tuner.md`, `vision_fe.md`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
