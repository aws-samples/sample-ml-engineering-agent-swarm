# src/amlas/tui/context

## Purpose

Share validated UI context and project/session/trace state through the context bus.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `bus.py:set_active_bus` — Register (or clear) the process-local active ``ContextBus`` for the agent tool.
- `bus.py:get_active_bus` — Return the process-local active ``ContextBus`` (or ``None`` when headless).
- `models.py:ActiveSpec` — The spec the user is working on (Specs tab / chat ``/spec`` / agent).
- `models.py:ActiveDataset` — The dataset in focus (Data/Lake tab / chat ``/data`` / agent).
- `session_ledger.py:LedgerEntry` — One immutable event in the session timeline (all fields JSON-serializable).
- `session_ledger.py:SessionState` — The append-only timeline record (session id + started_at + ordered entries).
- Owned files: `__init__.py`, `bus.py`, `models.py`, `session_ledger.py`, `trace_reader.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
