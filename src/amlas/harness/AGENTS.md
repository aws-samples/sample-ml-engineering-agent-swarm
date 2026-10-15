# src/amlas/harness

## Purpose

Provision and invoke opted-in managed Lead hosting and resolve its supported contracts.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `approvals.py:ApprovalDeniedError` — A gated tool call was refused, and the caller asked to fail rather than continue.
- `approvals.py:resolve_tool_calls` — Gate, then execute, each paused ``inline_function`` call — in order.
- `client.py:HarnessInvokeError` — Base class for harness invocation failures.
- `client.py:HarnessSdkTooOldError` — The installed SDK has no ``InvokeHarness`` operation.
- `leads.py:HarnessLeadError` — The harness path was requested for a Lead that cannot currently run on one.
- `provisioner.py:HarnessProvisionerError` — A lifecycle operation that must not be attempted, refused BEFORE calling AWS.
- `provisioner.py:HarnessDeletingError` — The harness is mid-deletion; nothing can be done with it until it drains.
- Owned files: `__init__.py`, `approvals.py`, `client.py`, `leads.py`, `provisioner.py`, `registry.py`, `spec.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
