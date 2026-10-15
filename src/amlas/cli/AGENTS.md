# src/amlas/cli

## Purpose

Expose the command-line application and its shared configuration/display entry points.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- `config.py:CLIConfig` — CLI configuration manager.
- `config.py:init_workspace` — Initialize a new workspace with default directory structure.
- `display.py:print_success` — Print a success message with green checkmark.
- `display.py:print_error` — Print an error message with red X.
- `main.py:cli` — Amlas ML Engineering Platform — Terminal Control System.
- Owned files: `__init__.py`, `__main__.py`, `config.py`, `display.py`, `main.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

| Path | Covers |
| --- | --- |
| `commands/AGENTS.md` | Implement the CLI command groups for specifications, data, research, runs, results, status, configuration and capabilities. |
