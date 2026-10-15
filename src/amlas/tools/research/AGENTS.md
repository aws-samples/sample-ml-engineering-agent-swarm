# src/amlas/tools/research

## Purpose

Aggregate research-provider results and bounded URL retrieval with explicit source/coverage information.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Keep provider/source provenance and bounded coverage in returned research records.
- HTTP(S) URL retrieval retains public-address checks, bounded fetches and HTTPS certificate verification.
- `aggregator.py:MergedResult` — A ResearchResult enriched with the set of sources that contributed it.
- `aggregator.py:research_aggregate` — Fan out `query` to all sources concurrently, dedupe, rank.
- `arxiv.py:ArxivError` — Raised on HTTP failure or unparseable Atom response.
- `arxiv.py:parse_arxiv_atom` — Parse an arxiv Atom-feed response into normalized ResearchResults.
- Owned files: `__init__.py`, `aggregator.py`, `arxiv.py`, `papers_with_code.py`, `public_fetch.py`, `semantic_scholar.py`, `strands_tools.py`, `tavily.py`, `types.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
