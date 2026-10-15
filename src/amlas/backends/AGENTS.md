# src/amlas/backends

## Purpose

Define typed backend ports, domain handles and adapter resolution at the workflow-to-compute boundary.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Flows depend on `ports.py` protocols and `types.py` domain handles; managed SDK details belong in adapters.
- `registry.py` resolves steering pins, explicit selectors and automatic choice; explicit unavailable choices must fail rather than silently switch.
- `hooks.py:HookContext` — The context a hook matcher/action inspects.
- `types.py:JobState` — Backend-neutral lifecycle state of an async job (training/processing/etc.).
- Owned files: `__init__.py`, `hooks.py`, `ports.py`, `registry.py`, `types.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

| Path | Covers |
| --- | --- |
| `adapters/AGENTS.md` | Implement and register concrete local and managed-service backend adapters. |
