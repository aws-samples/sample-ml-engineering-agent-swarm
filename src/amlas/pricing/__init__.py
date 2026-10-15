"""AWS pricing cache for cost ESTIMATES (FIX-25).

A small, self-refreshing price book used by ``estimate_dispatch_cost`` to give the chat
agent a defensible cost/runtime ESTIMATE before a real cloud dispatch. This is deliberately
an *estimate* surface, not a billing source:

* Prices may be cached in a dated local JSON snapshot
  ``aws-pricing-sample-<YYYY-MM-DD>.json`` under ``knowledge-base/pricing/``.
* On read, the newest snapshot's ``checked_at`` timestamp is compared to *now*; if it is
  older than 30 days (or no snapshot exists), a best-effort **Tavily** refresh runs and writes
  a new dated snapshot. Refresh is fully degradable — no Tavily key / no network / a parse
  failure simply keeps the last snapshot (or the baked-in default), never raising.
* ``now`` and the Tavily searcher are INJECTABLE so the whole module is directly verifiable
  hermetically (no real clock, no real network in CI).
"""

from .cache import (
    DEFAULT_PRICE_BOOK,
    PRICING_CACHE_DIR,
    STALENESS_DAYS,
    PriceBook,
    load_price_book,
    refresh_price_book_via_tavily,
)

__all__ = [
    "DEFAULT_PRICE_BOOK",
    "PRICING_CACHE_DIR",
    "STALENESS_DAYS",
    "PriceBook",
    "load_price_book",
    "refresh_price_book_via_tavily",
]
