# src/amlas/tui/jobs

## Purpose

Manage background UI execution, job lifecycle and cancellation.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Keep worker completion/error/cancellation states observable to the UI.
- Do not perform blocking managed calls on the UI event loop or lose cancellation during polling delays.
- `manager.py:JobCancelled` — Raised by a worker ``fn`` (or detected via the cancel token) to end a job
- `manager.py:Job` — One unit of background work and its full lifecycle state (§3.1).
- Owned files: `__init__.py`, `manager.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
