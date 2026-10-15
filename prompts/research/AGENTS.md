# prompts/research

## Purpose

Aggregate research-provider results and bounded URL retrieval with explicit source/coverage information.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Keep provider/source provenance and bounded coverage in returned research records.
- HTTP(S) URL retrieval retains public-address checks, bounded fetches and HTTPS certificate verification.
- Owned files: `algo_hypothesis.md`, `feature_hypothesis.md`, `framing.md`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
