# src/amlas/errors

## Purpose

Classify failures, produce user-facing error messages and apply bounded recovery/retry behavior.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `boundary.py:tool_error_boundary` — Wrap a tool fn so any exception becomes the standard error dict + is captured.
- `messages.py:format_user_error` — Return an actionable, human-facing message for ``exc`` (never a raw trace).
- `retry.py:backoff_delay` — Full-jitter exponential delay for ``attempt`` (0-indexed): ``uniform(0, nominal]``.
- `retry.py:retry_with_backoff` — Call ``fn`` with full-jitter exponential backoff on RETRYABLE failures.
- `taxonomy.py:ErrorCategory` — The disposition classes every amlas error falls into.
- `taxonomy.py:AmlasError` — Root of the amlas error taxonomy — carries classification + user-facing metadata.
- Owned files: `__init__.py`, `boundary.py`, `messages.py`, `retry.py`, `taxonomy.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
