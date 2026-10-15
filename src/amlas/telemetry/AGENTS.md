# src/amlas/telemetry

## Purpose

Write run traces, error records and execution state for inspection.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `error_log.py:resolve_runtime_mode` — Derive the current runtime mode from env (the "all 3 modes" tag source).
- `error_log.py:ErrorRecord` — One captured error — tagged with the runtime ``mode`` so all 3 modes share one surface.
- `trace.py:content_capture_enabled` — True only if OTEL GenAI message-content capture is EXPLICITLY enabled (SEC-A24).
- `trace.py:redact_for_telemetry` — Gate + redact message content for telemetry emission (SEC-A24).
- Owned files: `__init__.py`, `error_log.py`, `kanban.py`, `trace.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
