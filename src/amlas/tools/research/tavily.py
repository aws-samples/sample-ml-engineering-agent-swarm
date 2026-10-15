"""Tavily search source — keyed API or optional rate-limited keyless Search.

Docs: https://docs.tavily.com/documentation/keyless

A valid `TAVILY_API_KEY` uses normal keyed Search. Absent or placeholder keys
try optional rate-limited keyless Search; set `AMLAS_TAVILY_KEYLESS=false` to
disable it. Provider failures remain explicit; supplied public URLs can still be fetched.
"""
# ALTERNATIVE: Amazon Bedrock AgentCore Web Search offers an AWS-native managed
# web-search capability; this optional Tavily integration exists for adopters
# who already hold a Tavily key. The key lives in Secrets Manager, never code.

from __future__ import annotations

import os
from typing import Any

import httpx

from ...cloud.secrets import _is_placeholder
from .types import ResearchResult, ResearchSource

TAVILY_API_BASE = "https://api.tavily.com/search"
_MAX_RESULTS_CAP = 20
_DEFAULT_TIMEOUT = 15.0


class TavilyError(RuntimeError):
    """HTTP failure or unparseable response."""


class TavilyMissingKeyError(TavilyError):
    """`TAVILY_API_KEY` is not set in the environment."""


def parse_tavily_response(payload: dict[str, Any], *, access_mode: str | None = None) -> list[ResearchResult]:
    out: list[ResearchResult] = []
    for r in payload.get("results", []) or []:
        out.append(
            ResearchResult(
                source=ResearchSource.TAVILY,
                title=r.get("title", "") or "",
                url=r.get("url", "") or "",
                snippet=r.get("content", "") or "",
                score=float(r["score"]) if r.get("score") is not None else None,
                published=r.get("published_date") or None,
                identifiers={"access_mode": access_mode} if access_mode else {},
            )
        )
    return out


def tavily_search(
    query: str,
    *,
    max_results: int = 5,
    search_depth: str = "basic",
    api_key: str | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> list[ResearchResult]:
    """Search via Tavily.

    Args:
        query: free-text query.
        max_results: capped at 20.
        search_depth: "basic" (default, fast) or "advanced" (deeper, slower, costlier).
        api_key: override `TAVILY_API_KEY` env.

    Raises:
        TavilyMissingKeyError: no valid key and optional keyless Search disabled.
        TavilyError: non-2xx, network failure, or malformed payload.
    """
    key = api_key or os.environ.get("TAVILY_API_KEY")
    if key and _is_placeholder(key):
        key = None
    keyless = not key and os.environ.get("AMLAS_TAVILY_KEYLESS", "true").lower() in ("1", "true", "yes", "on")
    if not key and not keyless:
        raise TavilyMissingKeyError(
            "Tavily key is absent and optional keyless search is disabled; fetch supplied public URLs instead"
        )
    access_mode = "keyed" if key else "keyless"
    headers = {} if key else {"X-Tavily-Access-Mode": "keyless"}

    capped = max(1, min(int(max_results), _MAX_RESULTS_CAP))
    body = {
        "query": query,
        "max_results": capped,
        "search_depth": search_depth,
    }

    if key:
        body["api_key"] = key

    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(TAVILY_API_BASE, json=body, headers=headers, timeout=timeout)
    except httpx.HTTPError as e:
        # Avoid leaking the request body (which contains the api_key) in the message.
        raise TavilyError(f"tavily request failed: {type(e).__name__}") from None

    if r.status_code != 200:
        raise TavilyError(
            f"tavily {access_mode} search returned HTTP {r.status_code}; source coverage is incomplete"
        )

    try:
        return parse_tavily_response(r.json(), access_mode=access_mode)
    except (ValueError, KeyError):
        raise TavilyError("tavily response parse failed; no results fabricated") from None
