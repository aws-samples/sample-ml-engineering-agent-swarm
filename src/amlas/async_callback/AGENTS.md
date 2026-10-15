# src/amlas/async_callback

## Purpose

Translate managed training events into callback/job-state handling and optional runtime resumption.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- An event callback is not evidence of end-to-end runtime resumption; target binding and runtime input shape remain separate contracts.
- Retain job-state correlation and propagate meaningful service failures without credential/payload logging.
- `lambda_handler.py:build_resume_payload` — The utf-8 JSON payload sent to ``invoke_agent_runtime`` (pure).
- `lambda_handler.py:build_invoke_kwargs` — The exact kwargs for ``bedrock-agentcore.invoke_agent_runtime`` (pure).
- Owned files: `__init__.py`, `lambda_handler.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
