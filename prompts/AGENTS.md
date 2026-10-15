# prompts

## Purpose

Own the checked-in prompt templates used to build agent instructions.

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
| `harness/AGENTS.md` | Provision and invoke opted-in managed Lead hosting and resolve its supported contracts. |
| `leads/AGENTS.md` | Declare and build the seven lifecycle Lead agents and their shared tool/prompt contracts. |
| `preambles/AGENTS.md` | Maintained preambles boundary |
| `research/AGENTS.md` | Aggregate research-provider results and bounded URL retrieval with explicit source/coverage information. |
| `subagents/AGENTS.md` | Declare bounded specialist agents and their reusable tools/prompts. |
