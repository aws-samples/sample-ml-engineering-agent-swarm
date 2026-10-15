"""Normalized result schema used by every research source."""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

# Tracking / share parameters that should be dropped before two URLs are compared.
_TRACKING_PARAMS = frozenset(
    {
        "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        "ref", "ref_src", "ref_url", "fbclid", "gclid", "mc_cid", "mc_eid",
    }
)


class ResearchSource(enum.StrEnum):
    """Provenance for a single result. `str` mixin → JSON-friendly."""

    ARXIV = "ARXIV"
    TAVILY = "TAVILY"
    SEMANTIC_SCHOLAR = "SEMANTIC_SCHOLAR"
    PAPERS_WITH_CODE = "PAPERS_WITH_CODE"


def canonicalize_url(url: str) -> str:
    """Normalize a URL so two cosmetic variants compare equal.

    - Lowercase scheme + host
    - Strip tracking parameters (utm_*, ref, fbclid, ...)
    - Drop trailing slash on the path
    - For arxiv abstract pages, strip the version suffix (`v3` etc.)
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/")
    if netloc == "arxiv.org" and path.startswith("/abs/"):
        # `/abs/2401.12345v3` → `/abs/2401.12345`
        prefix = "/abs/"
        rest = path[len(prefix):]
        if "v" in rest:
            base, _, ver = rest.rpartition("v")
            if ver.isdigit():
                rest = base
        path = prefix + rest
    query_pairs = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
                   if k not in _TRACKING_PARAMS]
    query = urlencode(query_pairs)
    return urlunparse((scheme, netloc, path, parsed.params, query, parsed.fragment))


@dataclass(frozen=True)
class ResearchResult:
    """One research finding, normalized across sources.

    Required: source, title, url, snippet.
    Optional: score (source-specific relevance), published (ISO date), authors,
    identifiers (arxiv_id, doi, ssid, pwc_id).
    """

    source: ResearchSource
    title: str
    url: str
    snippet: str
    score: float | None = None
    published: str | None = None
    authors: tuple[str, ...] | list[str] = field(default_factory=tuple)
    identifiers: dict[str, str] = field(default_factory=dict)

    @property
    def canonical_id(self) -> str:
        """ID used for cross-source dedup. Preference order:
        arxiv ID > DOI > canonicalized URL."""
        if "arxiv_id" in self.identifiers:
            return f"arxiv:{self.identifiers['arxiv_id']}"
        if "doi" in self.identifiers:
            return f"doi:{self.identifiers['doi']}"
        return canonicalize_url(self.url)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["source"] = self.source.name
        d["authors"] = list(self.authors)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResearchResult:
        return cls(
            source=ResearchSource[d["source"]] if isinstance(d["source"], str)
            else d["source"],
            title=d["title"],
            url=d["url"],
            snippet=d["snippet"],
            score=d.get("score"),
            published=d.get("published"),
            authors=list(d.get("authors") or []),
            identifiers=dict(d.get("identifiers") or {}),
        )
