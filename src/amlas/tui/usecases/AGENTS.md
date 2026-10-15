# src/amlas/tui/usecases

## Purpose

Define machine-readable data/ML use-case registries used by the interface.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `mlops_registry.py:get_mlops_use_case` — Return the :class:`UseCase` with ``id == uc_id`` (or ``None``).
- `mlops_registry.py:mlops_use_cases_for_category` — Return all MLOps use cases in ``category`` (registry order).
- `registry.py:UseCase` — One declarative, harness-drivable DQ/DE use case (master-plan §C.3).
- `registry.py:get_use_case` — Return the :class:`UseCase` with ``id == uc_id`` (or ``None``).
- Owned files: `__init__.py`, `mlops_registry.py`, `registry.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
