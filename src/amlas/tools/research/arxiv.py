"""arxiv.org search source — Atom-feed REST API, free, no key.

API docs: https://info.arxiv.org/help/api/user-manual.html
"""

from __future__ import annotations

import re

# SEC-16 / R-013: the remote Atom response and its parser exceptions stay behind
# the defusedxml boundary, which rejects unsafe DTD/entity expansion constructs.
import defusedxml.ElementTree as ET
import httpx

from .types import ResearchResult, ResearchSource

ARXIV_API_BASE = "https://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom"}
_MAX_RESULTS_CAP = 50  # bounded to keep agent prompts manageable
_DEFAULT_TIMEOUT = 30.0  # arxiv's slow path is real; 15s spuriously failed long queries
_RETRYABLE_EXC = (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ReadError)
_STOPWORDS = frozenset({
    # Function words
    "the", "a", "an", "of", "and", "or", "for", "to", "in", "on", "with", "by",
    "is", "are", "be", "from", "this", "that", "these", "those", "it",
    # Generic ML jargon — appear in too many papers to narrow usefully
    "classification", "regression", "prediction", "predict", "model", "models",
    "machine", "learning", "method", "methods", "approach", "deep",
    "neural", "network", "networks", "f1", "macro", "micro", "metric", "metrics",
    "data", "dataset", "datasets",
})
_MAX_FIRST_QUERY_TOKENS = 5  # arxiv AND beyond ~5 terms reliably finds nothing


def _build_search_query(query: str, categories: list[str] | None) -> str:
    """ANDed token query (one `all:term` per significant token) + optional cat filters.

    Phrase-matching the whole query (`all:"..."`) is too restrictive — long natural-
    language queries find nothing. Per-token AND mirrors what the arxiv search UI does.
    """
    tokens = [t.strip(".,;:()[]\"'") for t in query.lower().split()]
    tokens = [t for t in tokens if t and t not in _STOPWORDS]
    parts: list[str] = [f"all:{t}" for t in tokens]
    for cat in categories or []:
        parts.append(f"cat:{cat}")
    return " AND ".join(parts) if parts else "all:*"


class ArxivError(RuntimeError):
    """Raised on HTTP failure or unparseable Atom response."""


def parse_arxiv_atom(xml_text: str) -> list[ResearchResult]:
    """Parse an arxiv Atom-feed response into normalized ResearchResults."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as error:
        raise ArxivError(f"unparseable arxiv response: {error}") from error

    out: list[ResearchResult] = []
    for entry in root.findall("atom:entry", _NS):
        title_el = entry.find("atom:title", _NS)
        summary_el = entry.find("atom:summary", _NS)
        id_el = entry.find("atom:id", _NS)
        published_el = entry.find("atom:published", _NS)

        if title_el is None or id_el is None:
            continue

        title = (title_el.text or "").strip()
        snippet = " ".join((summary_el.text or "").split()) if summary_el is not None else ""
        url = (id_el.text or "").strip()
        published = ""
        if published_el is not None and published_el.text:
            published = published_el.text.split("T")[0]  # ISO date only

        # arxiv ID = trailing path component minus version (`v3`).
        m = re.search(r"/abs/([^/]+?)(v\d+)?$", url)
        arxiv_id = m.group(1) if m else ""

        authors = [
            (a.findtext("atom:name", default="", namespaces=_NS) or "").strip()
            for a in entry.findall("atom:author", _NS)
        ]
        authors = [a for a in authors if a]

        out.append(
            ResearchResult(
                source=ResearchSource.ARXIV,
                title=title,
                url=url,
                snippet=snippet,
                published=published or None,
                authors=tuple(authors),
                identifiers={"arxiv_id": arxiv_id} if arxiv_id else {},
            )
        )
    return out


def arxiv_search(
    query: str,
    *,
    max_results: int = 10,
    categories: list[str] | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> list[ResearchResult]:
    """Search arxiv. Returns normalized ResearchResults.

    Args:
        query: free-text search; we wrap as `all:"<query>"`.
        max_results: capped at 50.
        categories: optional list of arxiv categories (e.g. ["cs.LG", "stat.ML"]).
            ANDed with the query as `cat:<category>`.

    Raises:
        ArxivError: HTTP failure, non-2xx, or unparseable response.
    """
    capped = max(1, min(int(max_results), _MAX_RESULTS_CAP))

    # Progressive widening — start with all significant tokens AND'd; if that
    # returns 0, drop the rightmost token and retry. Up to 3 attempts.
    tokens = [t.strip(".,;:()[]\"'") for t in query.lower().split()]
    tokens = [t for t in tokens if t and t not in _STOPWORDS]
    if not tokens:
        return []
    # Cap initial token count — beyond ~5 ANDed terms arxiv reliably returns 0.
    tokens = tokens[:_MAX_FIRST_QUERY_TOKENS]

    last_exc: Exception | None = None
    parsed: list[ResearchResult] = []
    for shrink in range(min(3, max(1, len(tokens) - 1))):
        active_tokens = tokens[: max(2, len(tokens) - shrink)]
        parts = [f"all:{t}" for t in active_tokens]
        for cat in categories or []:
            parts.append(f"cat:{cat}")
        params: dict[str, str | int] = {
            "search_query": " AND ".join(parts) if parts else "all:*",
            "max_results": capped,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        for _ in range(2):  # one retry on timeout per shrink level
            try:
                with httpx.Client(timeout=timeout) as client:
                    r = client.get(ARXIV_API_BASE, params=params, timeout=timeout)
                break
            except _RETRYABLE_EXC as e:
                last_exc = e
                continue
            except httpx.HTTPError as e:
                raise ArxivError(f"arxiv request failed: {e}") from e
        else:
            raise ArxivError(f"arxiv request failed after retry: {last_exc}") from last_exc

        if r.status_code != 200:
            raise ArxivError(f"arxiv returned HTTP {r.status_code}: {r.text[:200]}")
        parsed = parse_arxiv_atom(r.text)
        if parsed:
            break  # widened enough — stop here

    return parsed
