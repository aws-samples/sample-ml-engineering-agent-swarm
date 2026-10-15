# src/amlas/tui/screens

## Purpose

Present the application tabs and route their actions to the existing tools/capability layer.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `chat.py:ChatInput` — Multi-line chat input: Enter submits, Ctrl+J / Shift+Enter add a newline.
- `chat.py:ChatPane` — Chat interface — conversational control for all platform features.
- `dashboard.py:DashboardPane` — Main dashboard — live overview + auto-monitor + observability + agent activity.
- `data.py:DataPane` — Data explorer screen for dataset management + DQ pattern scan.
- `lake.py:LakePane` — Medallion lakehouse browser — catalog, slice read, write-back.
- `mlops.py:MLOpsPane` — MLOps dashboard — registry table, monitor panel, KPI scorecard.
- Owned files: `__init__.py`, `chat.py`, `dashboard.py`, `data.py`, `lake.py`, `mlops.py`, `pipeline.py`, `results.py`, `specs.py`, `steering.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
