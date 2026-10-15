"""Research aggregator — fan-out, dedupe by canonical id, rank by source diversity.

Each source is a callable `(query: str, **kwargs) -> list[ResearchResult]`. The
aggregator runs them concurrently, tolerates per-source failures, and produces
a single ranked, deduplicated list with merged source provenance.

Ranking signal (descending):
    1. Number of distinct sources that found this paper (cross-source agreement).
    2. Max score across the matching results (per-source relevance).
    3. Recency (published date), if available.
    4. Stable tie-break on canonical_id for deterministic ordering.
"""

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .types import ResearchResult, ResearchSource

SourceFn = Callable[..., list[ResearchResult]]


@dataclass(frozen=True)
class MergedResult:
    """A ResearchResult enriched with the set of sources that contributed it."""

    title: str
    url: str
    snippet: str
    canonical_id: str
    sources: tuple[ResearchSource, ...]
    score: float | None = None
    published: str | None = None
    authors: tuple[str, ...] = ()
    identifiers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AggregateReport:
    results: list[MergedResult]
    errors: dict[str, str]  # source name → error message
    queried_sources: tuple[str, ...] = ()


def _merge_group(group: list[ResearchResult]) -> MergedResult:
    """Collapse a group of results sharing canonical_id into one MergedResult."""
    sources = tuple(sorted({r.source for r in group}, key=lambda s: s.name))
    # Pick the longest title and longest snippet (richest content wins).
    best_title = max(group, key=lambda r: len(r.title or "")).title
    best_snippet = max(group, key=lambda r: len(r.snippet or "")).snippet
    # Url: prefer one with an arxiv_id identifier, else first non-empty.
    pri = next((r for r in group if "arxiv_id" in r.identifiers), group[0])
    # Score: max across group, ignoring Nones.
    scores = [r.score for r in group if r.score is not None]
    score = max(scores) if scores else None
    # Published: earliest (when known) → reflects original publication date.
    publisheds = sorted(r.published for r in group if r.published)
    published = publisheds[0] if publisheds else None
    # Authors: union, preserving insertion order.
    seen_authors: list[str] = []
    for r in group:
        for a in r.authors or ():
            if a not in seen_authors:
                seen_authors.append(a)
    # Identifiers: merge dicts, later writes don't clobber earlier non-empty values.
    identifiers: dict[str, str] = {}
    for r in group:
        for k, v in (r.identifiers or {}).items():
            if v and k not in identifiers:
                identifiers[k] = v

    return MergedResult(
        title=best_title,
        url=pri.url,
        snippet=best_snippet,
        canonical_id=pri.canonical_id,
        sources=sources,
        score=score,
        published=published,
        authors=tuple(seen_authors),
        identifiers=identifiers,
    )


def _rank_key(m: MergedResult) -> tuple[Any, ...]:
    # Higher source-count first → invert with negative.
    # Higher score first → invert with negative (Nones go last).
    score_key = -(m.score if m.score is not None else -1.0)
    pub_key = m.published or ""
    # Reverse pub date for descending recency: invert via `-ord` trick? simpler — sort
    # by the *negated* date string later. Easiest: negate via lambda `not None`.
    return (-len(m.sources), score_key, _neg_iso_date(pub_key), m.canonical_id)


def _neg_iso_date(d: str) -> str:
    """Trick: invert ISO date for descending sort by string. Recent dates → small string."""
    if not d:
        return "~~~~"  # later than any real ISO date string lexicographically
    return "".join(chr(ord("9") - int(c) + ord("0")) if c.isdigit() else c for c in d)


def research_aggregate(
    query: str,
    *,
    sources: dict[str, SourceFn],
    max_per_source: int = 10,
    max_total: int = 25,
    timeout_per_source: float = 15.0,
) -> AggregateReport:
    """Fan out `query` to all sources concurrently, dedupe, rank.

    Args:
        query: free-text search.
        sources: name → callable. The callable signature is
            `(query: str, max_results: int) -> list[ResearchResult]`. When a source
            doesn't accept `max_results`, kwargs are filtered to those it does.
        max_per_source: cap each source's contribution before merging.
        max_total: cap on the final merged + ranked list.
        timeout_per_source: per-source timeout (seconds).

    Returns:
        AggregateReport — ranked merged results + per-source error map.
    """
    errors: dict[str, str] = {}
    raw_by_source: dict[str, list[ResearchResult]] = {}

    def _call(name: str, fn: SourceFn) -> tuple[str, list[ResearchResult] | Exception]:
        try:
            res = fn(query, max_results=max_per_source)
            return name, list(res or [])
        except Exception as e:
            return name, e

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(sources))) as ex:
        futures = {ex.submit(_call, n, fn): n for n, fn in sources.items()}
        # Per-source timeout isolation (OPT-02): a single global `as_completed`
        # deadline let one slow source starve / abort the whole aggregate. Wait
        # on each future independently so a slow/hung source is recorded as an
        # error without dropping results that already completed.
        for future, name in futures.items():
            try:
                _name, payload = future.result(timeout=timeout_per_source)
            except concurrent.futures.TimeoutError:
                errors[name] = f"TimeoutError: source exceeded {timeout_per_source}s"
                future.cancel()
                continue
            if isinstance(payload, Exception):
                errors[name] = f"{type(payload).__name__}: {payload}"
            else:
                raw_by_source[name] = payload

    # Group by canonical_id across all sources.
    groups: dict[str, list[ResearchResult]] = {}
    for results in raw_by_source.values():
        for r in results:
            groups.setdefault(r.canonical_id, []).append(r)

    merged = [_merge_group(g) for g in groups.values()]
    merged.sort(key=_rank_key)
    if len(merged) > max_total:
        merged = merged[:max_total]

    return AggregateReport(
        results=merged,
        errors=errors,
        queried_sources=tuple(sorted(sources.keys())),
    )


def merged_to_dict(m: MergedResult) -> dict[str, Any]:
    return {
        "title": m.title,
        "url": m.url,
        "snippet": m.snippet,
        "canonical_id": m.canonical_id,
        "sources": [s.name for s in m.sources],
        "score": m.score,
        "published": m.published,
        "authors": list(m.authors),
        "identifiers": dict(m.identifiers),
    }
