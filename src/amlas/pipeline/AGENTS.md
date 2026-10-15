# src/amlas/pipeline

## Purpose

Coordinate local and agentic experiment pipeline execution and result summaries.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `agentic_runner.py:build_pipeline_agent` — Build the autonomous ML pipeline agent.
- `orchestrator.py:PipelineResult` — Full pipeline output containing all artifacts and intermediate results.
- `orchestrator.py:MLPipeline` — Orchestrates the full ML engineering pipeline.
- Owned files: `__init__.py`, `agentic_runner.py`, `orchestrator.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
