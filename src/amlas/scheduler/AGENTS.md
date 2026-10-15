# src/amlas/scheduler

## Purpose

Manage project queues, research cadence and scheduler/dashboard state.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `dashboard.py:ProjectDashboard` — Generates markdown dashboards for multi-project scheduler state.
- `project_manager.py:Project` — A single ML project in the scheduler queue.
- `project_manager.py:ProjectManager` — Thread-safe project queue manager with JSON persistence.
- `research_scheduler.py:normalize_cadence` — Normalize a cadence shorthand; unknown / empty → the weekly default (never raises).
- `research_scheduler.py:scan_id_for` — Return a stable, AWS-safe scan ID derived with SHA-256.
- `scheduler.py:ProjectScheduler` — Parallel project scheduler with concurrency limits.
- Owned files: `__init__.py`, `dashboard.py`, `project_manager.py`, `research_scheduler.py`, `scheduler.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
