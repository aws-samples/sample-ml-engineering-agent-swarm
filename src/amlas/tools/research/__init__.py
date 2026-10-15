"""Research / intelligence layer — replaces the v1 `web_retrieve` stub.

Sources (each module is a thin client that returns normalized `ResearchResult`s):

    arxiv.py            — arxiv.org Atom-feed REST API (free)
    tavily.py           — keyed Tavily or optional rate-limited keyless Search
    public_fetch.py     — bounded fetch of supplied public URLs; untrusted text
    semantic_scholar.py — Semantic Scholar Graph API (free, rate-limited)
    papers_with_code.py — Papers with Code API (free)

The aggregator fans out to enabled sources in parallel, dedupes by canonical
identifier (arxiv ID > DOI > canonicalized URL), and ranks by source-mix +
recency. Strands `@tool` wrappers live in `strands_tools.py` so agents only
see one entry point.

Per ADR-0001, these are thin wrappers (≤200 LOC each) over established
external APIs — not custom search infrastructure.
"""

from .types import ResearchResult, ResearchSource, canonicalize_url

__all__ = ["ResearchResult", "ResearchSource", "canonicalize_url"]
