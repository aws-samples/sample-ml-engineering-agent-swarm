# src/amlas/subagents/fine_tune_a2a

## Purpose

Expose the separate managed fine-tuning sub-runtime and its invocation boundary.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `app.py:fine_tune_invocation` — A2A entry point — receives fine-tune requests, delegates to the fine_tune sub-agent.
- Owned files: `__init__.py`, `app.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
