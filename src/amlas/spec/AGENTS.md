# src/amlas/spec

## Purpose

Parse task specifications and acceptance configuration into validated models.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Validate inputs through `schema.py` and `loader.py`; report schema/acceptance errors rather than silently repairing intent.
- `text_generation` is a valid task type. Generation spec validation does not
  establish a supported trainer or grader. Unmeasured baseline scores remain null.
- `loader.py:SpecLoadError` — Raised when a task_spec.md cannot be parsed or validated.
- `loader.py:load_spec_from_text` — Parse a spec string. Raises SpecLoadError on any parse or validation failure.
- Owned files: `__init__.py`, `loader.py`, `schema.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
