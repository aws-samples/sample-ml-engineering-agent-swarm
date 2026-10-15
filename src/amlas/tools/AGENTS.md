# src/amlas/tools

## Purpose

Expose agent-callable data, research, ML, deployment and operational tools.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- AWS readiness distinguishes permitted actions from selected configuration;
  configured references are not existence, ownership or authorization proof.
- Generation inference/recommendations preserve text intent and report research
  gaps; they must not fall back to tabular training or invent baseline scores.
- Inference preserves explicit classification/numeric prediction/forecasting objectives when a
  language model is the implementation. Negated task exclusions are not positive
  objective signals; generation aliases share the canonical recommendation path.
- Spec rendering preserves supplied dataset, splits, constraints and budget.
- `run_ml_capability` accepts only a typed caller policy from SDK-injected
  ToolContext; hidden context is excluded from the model tool schema. Prompt
  text and params JSON do not authorize execution. Other tools keep their guards.
- `agent_tools.py:data_verify_tool` — Produce a compact data profile (shape, dtypes, NaN%, sample) for agent consumption.
- `agent_tools.py:leakage_check_tool` — Run leakage checks against train/test frames.
- `aws_control.py:ensure_s3_prefix` — Ensure an S3 lake prefix exists (idempotent), so a job can write there.
- `aws_control.py:ensure_glue_database` — Ensure a Glue Data Catalog database exists (idempotent).
- `backend_neutral.py:submit_training_run` — Submit a training run for ``algo`` on ``train_uri`` (T-TRAIN-01, backend-neutral).
- `code_interpreter.py:register_staged_file` — Record a ``{sandbox_target: local_source}`` pair so FIX-22 can re-stage it on reprovision.
- `code_interpreter.py:sandbox_egress_enabled` — True only if sandbox network egress is EXPLICITLY enabled (SEC-A15; default OFF).
- Owned files: `__init__.py`, `agent_tools.py`, `aws_control.py`, `backend_neutral.py`, `code_interpreter.py`, `dashboard_tools.py`, `data_etl_neutral.py`, `data_quality_tools.py`, `data_tools.py`, `data_verify.py`, `decision_tools.py`, `deploy_neutral.py`, `dispatch_preflight.py`, `evaluation_tools.py`, `experiment_tools.py`, `fe_advanced_tools.py`, `fe_tools.py`, `fs_tools.py`, `git_neutral.py`, `git_tools.py`, `governance_m8_neutral.py`, `governance_neutral.py`, `hpo_tools.py`, `leakage_check.py`, `leakage_columns.py`, `local_tools.py`, `ml_neutral.py`, `monitor_neutral.py`, `orchestration_tools.py`, `pathref.py`, `pii_tools.py`, `research_neutral.py`, `research_tools.py`, `retrieval_neutral.py`, `sagemaker.py`, `sagemaker_source_bundle.py`, `scheduler_tools.py`, `spec_generation_tools.py`, `spec_heuristics.py`, `spec_tools.py`, `training_jobs.py`, `training_tools.py`, `uri_resolution.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

| Path | Covers |
| --- | --- |
| `data_quality/AGENTS.md` | Profile data quality through the catalog, patterns, missingness analysis and scorecard. |
| `imputation/AGENTS.md` | Select and evaluate numeric, categorical, temporal and LLM-assisted missing-value methods. |
| `lake/AGENTS.md` | Resolve partitioned medallion datasets and retain the lake read/write boundary. |
| `mlops/AGENTS.md` | Expose managed ML registry, pipeline, feature-store, explainability and monitoring operations. |
| `research/AGENTS.md` | Aggregate research-provider results and bounded URL retrieval with explicit source/coverage information. |
| `synthetic/AGENTS.md` | Generate or augment rows and evaluate fidelity, utility and privacy evidence. |
