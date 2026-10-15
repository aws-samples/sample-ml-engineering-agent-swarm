# src/amlas/cli/commands

## Purpose

Implement the CLI command groups for specifications, data, research, runs, results, status, configuration and capabilities.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `capability.py:list_cmd` — List every registered capability (id, gated, runnable) from the shared registry.
- `config.py:config` — Configuration — show, set, initialize workspace, check health.
- `config.py:show` — Show current configuration.
- `data.py:data` — Data management — describe, quality, impute, synth, lake, and more.
- `data.py:describe` — Describe a dataset (EDA summary).
- `info.py:info` — System info — runtime, version, health checks.
- `info.py:version` — Show the platform version.
- Owned files: `__init__.py`, `capability.py`, `config.py`, `data.py`, `info.py`, `mlops.py`, `research.py`, `results.py`, `run.py`, `spec.py`, `status.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
