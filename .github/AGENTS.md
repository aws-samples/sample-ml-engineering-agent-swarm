# .github

## Purpose

Own the public hosting automation and its least-privilege workflow boundary.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Maintain the interfaces and file formats owned by the modules listed below; parent contracts apply.
- Owned files: .

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

| Path | Covers |
| --- | --- |
| `workflows/AGENTS.md` | Run public-release checks with immutable action references and minimal permissions. |
