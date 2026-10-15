# src/amlas/memory

## Purpose

Connect agent hooks to managed memory and session state.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `hooks.py:MemoryClientProto` — Minimal surface of `bedrock_agentcore.memory.MemoryClient` we actually use.
- `hooks.py:MemoryHookProvider` — HookProvider that pulls history on init and persists new messages on turn boundaries.
- Owned files: `__init__.py`, `hooks.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
