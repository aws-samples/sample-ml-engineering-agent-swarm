"""Papers with Code search source.

DEPRECATED 2025-Q1 — the public REST API was decommissioned by Meta and the
endpoint now 302s to https://huggingface.co/papers/trending. The module + unit
tests are retained against fixture data so we can re-enable instantly if a
maintained fork emerges; it is NOT part of the default source set
(see `strands_tools._build_default_sources`).

Docs (historical): https://paperswithcode.com/api/v1/docs/
"""

from __future__ import annotations

from typing import Any

import httpx

from .types import ResearchResult, ResearchSource

PWC_API_BASE = "https://paperswithcode.com/api/v1/papers/"
_MAX_RESULTS_CAP = 25
_DEFAULT_TIMEOUT = 15.0
# SEC-A17 (SSRF / CWE-918): this is the ONE research client that follows redirects (the API
# 302-canonicalizes query strings). Cap the redirect chain so a redirect loop / redirect-to-
# arbitrary-host cannot be driven indefinitely. The base host is fixed (paperswithcode.com),
# the caller controls only query params, and the module is deprecated/off the default source
# set — so a small bound is sufficient defense-in-depth.
_MAX_REDIRECTS = 3


class PapersWithCodeError(RuntimeError):
    """HTTP failure or unparseable response."""


def parse_pwc_response(payload: dict[str, Any]) -> list[ResearchResult]:
    out: list[ResearchResult] = []
    for p in payload.get("results", []) or []:
        identifiers: dict[str, str] = {}
        if p.get("id"):
            identifiers["pwc_id"] = str(p["id"])
        if p.get("arxiv_id"):
            identifiers["arxiv_id"] = str(p["arxiv_id"])

        out.append(
            ResearchResult(
                source=ResearchSource.PAPERS_WITH_CODE,
                title=p.get("title", "") or "",
                url=p.get("url_abs", "") or "",
                snippet=p.get("abstract") or "",
                published=p.get("published") or None,
                authors=tuple(p.get("authors") or []),
                identifiers=identifiers,
            )
        )
    return out


def papers_with_code_search(
    query: str,
    *,
    max_results: int = 10,
    timeout: float = _DEFAULT_TIMEOUT,
) -> list[ResearchResult]:
    capped = max(1, min(int(max_results), _MAX_RESULTS_CAP))
    params: dict[str, str | int] = {"q": query, "items_per_page": capped}

    try:
        # `follow_redirects=True` — Papers with Code's /api/v1/papers/ canonicalizes
        # query strings via 302; without follow we get a useless redirect response.
        # SEC-A17: bound the redirect chain (max_redirects) so a redirect loop / chain to an
        # arbitrary host cannot be driven indefinitely (SSRF defense-in-depth on the one
        # redirect-following research client).
        with httpx.Client(
            timeout=timeout, follow_redirects=True, max_redirects=_MAX_REDIRECTS
        ) as client:
            r = client.get(PWC_API_BASE, params=params, timeout=timeout)
    except httpx.HTTPError as e:
        raise PapersWithCodeError(f"papers-with-code request failed: {e}") from e

    if r.status_code != 200:
        raise PapersWithCodeError(
            f"papers-with-code HTTP {r.status_code}: {r.text[:200]}"
        )

    try:
        return parse_pwc_response(r.json())
    except (ValueError, KeyError) as e:
        raise PapersWithCodeError(f"papers-with-code parse failed: {e}") from e
