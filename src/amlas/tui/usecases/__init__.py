"""Machine-readable top-100 data-quality & data-engineering use-case registry.

This package is the **single source of truth** for the 100 DQ/DE use cases the
Phase-8 E2E harness drives through the TUI (chat -> ContextBus -> panels ->
assertions). The two companion Markdown documents
(``docs/use-cases/data-quality-engineering-100-stories.md`` and
``…-100-tdds.md``) are generated to share IDENTICAL ids/titles/prompts with
:data:`amlas.tui.usecases.registry.USE_CASES`.

See :mod:`amlas.tui.usecases.registry` for the :class:`UseCase` schema
(grounded in master-plan §C.3) and the taxonomy constants.
"""

from __future__ import annotations

from amlas.tui.usecases.registry import (
    CANONICAL_EVENTS,
    CATEGORIES,
    CATEGORY_COUNTS,
    USE_CASES,
    UseCase,
    category_counts,
    get_use_case,
    use_cases_for_category,
)

__all__ = [
    "CANONICAL_EVENTS",
    "CATEGORIES",
    "CATEGORY_COUNTS",
    "USE_CASES",
    "UseCase",
    "category_counts",
    "get_use_case",
    "use_cases_for_category",
]
