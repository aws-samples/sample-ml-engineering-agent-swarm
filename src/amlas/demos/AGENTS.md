# src/amlas/demos

## Purpose

Provide the optional code-interpreter demonstration runner and its explicit staging/execution boundary.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `sandbox_runner.py:SandboxRunFailed` — Sandbox returned an error or unparseable output.
- `sandbox_runner.py:train_ai4i_in_sandbox` — Run FE + training in the AgentCore CI sandbox; return a typed result.
- Owned files: `__init__.py`, `sandbox_runner.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
