"""Semantic Scholar Graph API search source.

Docs: https://api.semanticscholar.org/api-docs/graph

Free tier rate-limits aggressively (~100 req/5 min). Optional `SEMANTIC_SCHOLAR_API_KEY`
unlocks higher quotas (sent as `x-api-key` header).
"""

from __future__ import annotations

import math
import os
from typing import Any

import httpx

from .types import ResearchResult, ResearchSource

SS_API_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"
_MAX_RESULTS_CAP = 25
_DEFAULT_TIMEOUT = 15.0
_FIELDS = "title,abstract,url,year,citationCount,externalIds,authors"


class SemanticScholarError(RuntimeError):
    """HTTP failure or unparseable response."""


def _citation_score(cit: int | None) -> float | None:
    """log10(1 + citations) — bounded, comparable across sources."""
    if cit is None:
        return None
    try:
        return float(round(math.log10(1 + max(0, int(cit))), 4))
    except (TypeError, ValueError):
        return None


def parse_semantic_scholar_response(payload: dict[str, Any]) -> list[ResearchResult]:
    out: list[ResearchResult] = []
    for p in payload.get("data", []) or []:
        ext = p.get("externalIds") or {}
        identifiers: dict[str, str] = {}
        if p.get("paperId"):
            identifiers["ssid"] = str(p["paperId"])
        if ext.get("ArXiv"):
            identifiers["arxiv_id"] = str(ext["ArXiv"])
        if ext.get("DOI"):
            identifiers["doi"] = str(ext["DOI"])

        authors = tuple(
            (a.get("name") or "").strip()
            for a in (p.get("authors") or [])
            if a.get("name")
        )

        year = p.get("year")
        out.append(
            ResearchResult(
                source=ResearchSource.SEMANTIC_SCHOLAR,
                title=p.get("title", "") or "",
                url=p.get("url", "") or "",
                snippet=p.get("abstract") or "",
                score=_citation_score(p.get("citationCount")),
                published=str(year) if year else None,
                authors=authors,
                identifiers=identifiers,
            )
        )
    return out


def semantic_scholar_search(
    query: str,
    *,
    max_results: int = 10,
    api_key: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> list[ResearchResult]:
    """Search Semantic Scholar Graph API. Free tier; key unlocks higher quotas.

    Raises:
        SemanticScholarError: HTTP failure (incl. 429), parse error.
    """
    capped = max(1, min(int(max_results), _MAX_RESULTS_CAP))
    headers: dict[str, str] = {}
    key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if key:
        headers["x-api-key"] = key

    params: dict[str, str | int] = {"query": query, "limit": capped, "fields": _FIELDS}

    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(SS_API_BASE, params=params, headers=headers, timeout=timeout)
    except httpx.HTTPError as e:
        raise SemanticScholarError(f"semantic scholar request failed: {e}") from e

    if r.status_code != 200:
        raise SemanticScholarError(
            f"semantic scholar HTTP {r.status_code}: {r.text[:200]}"
        )

    try:
        return parse_semantic_scholar_response(r.json())
    except (ValueError, KeyError) as e:
        raise SemanticScholarError(f"semantic scholar parse failed: {e}") from e
