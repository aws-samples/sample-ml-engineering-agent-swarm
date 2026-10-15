# src/amlas/experiments

## Purpose

Configure and execute experiment plans and hyperparameter selections.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `config.py:ExperimentConfig` — Configuration for a single experiment run.
- `config.py:ExperimentPlan` — A plan containing multiple experiment configurations to run.
- `runner.py:ExperimentRunner` — Orchestrates running multiple training experiments and tracking results.
- Owned files: `__init__.py`, `config.py`, `hyperparams.py`, `runner.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
