# src/amlas/pricing

## Purpose

Cache public AWS pricing inputs used for cost estimates.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Cached rates are estimation inputs, not metered charges or an account budget.
- Preserve explicit cache/source metadata and bounded fallback behavior in `cache.py`.
- `cache.py:PriceBook` — A resolved price book + provenance for a cost estimate.
- `cache.py:load_price_book` — Load the newest price snapshot; refresh via Tavily if stale (>30d) or absent.
- Owned files: `__init__.py`, `cache.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
