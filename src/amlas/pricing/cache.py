"""Dated AWS price-book snapshots with a 30-day Tavily self-refresh (FIX-25).

See the package docstring. Time and network seams are injectable for deterministic
offline operation.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Runtime snapshot store. Snapshots are local artifacts and are never committed.
_REPO_ROOT = Path(__file__).resolve().parents[3]
PRICING_CACHE_DIR = _REPO_ROOT / "knowledge-base" / "pricing"
_SNAPSHOT_GLOB = "aws-pricing-sample-*.json"
_SNAPSHOT_DATE_RE = re.compile(r"aws-pricing-sample-(\d{4}-\d{2}-\d{2})\.json$")

# A snapshot older than this triggers a best-effort refresh.
STALENESS_DAYS = 30

# Baked-in default (us-west-2 on-demand ballparks, USD). Deliberately a rough ESTIMATE
# basis, not a billing quote — the whole surface is caveated as an estimate. These back the
# very first run before any snapshot is written, and any field a refresh fails to update.
DEFAULT_PRICE_BOOK: dict[str, Any] = {
    "checked_at": "1970-01-01T00:00:00+00:00",  # epoch → always stale until refreshed/written
    "region": "us-west-2",
    "currency": "USD",
    "source": "baked-in default (rough estimate basis; refresh via Tavily)",
    "prices": {
        # Bedrock — per 1K tokens (Opus tier; the swarm's Lead model).
        "bedrock_opus_input_per_1k": 0.015,
        "bedrock_opus_output_per_1k": 0.075,
        # SageMaker training — per instance-hour (common small training instance).
        "sagemaker_ml_m5_xlarge_hour": 0.23,
        # S3 — per GB-month (session payload staging; negligible for a single run).
        "s3_standard_gb_month": 0.023,
    },
}


@dataclass
class PriceBook:
    """A resolved price book + provenance for a cost estimate."""

    prices: dict[str, float]
    checked_at: str
    region: str = "us-west-2"
    currency: str = "USD"
    source: str = ""
    path: str | None = None          # snapshot file it came from (None → baked-in default)
    stale: bool = False              # True when older than STALENESS_DAYS at load time
    refreshed: bool = False          # True when this load triggered a successful refresh
    warnings: list[str] = field(default_factory=list)

    def price(self, key: str, default: float = 0.0) -> float:
        """Return a price by key, falling back to the baked-in default then ``default``."""
        if key in self.prices:
            return float(self.prices[key])
        baked = DEFAULT_PRICE_BOOK["prices"].get(key)
        return float(baked) if baked is not None else float(default)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _parse_iso(ts: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(ts)
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    except Exception:
        return None


def _newest_snapshot(cache_dir: Path) -> Path | None:
    """Return the snapshot file with the newest date in its NAME, or None."""
    try:
        candidates = sorted(cache_dir.glob(_SNAPSHOT_GLOB))
    except Exception:
        return None
    dated: list[tuple[str, Path]] = []
    for p in candidates:
        m = _SNAPSHOT_DATE_RE.search(p.name)
        if m:
            dated.append((m.group(1), p))
    if not dated:
        return None
    dated.sort(key=lambda t: t[0])
    return dated[-1][1]


def _read_snapshot(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None
    except Exception:
        logger.debug("pricing snapshot unreadable: %s", path, exc_info=True)
        return None


def _is_stale(checked_at: str, now: datetime) -> bool:
    dt = _parse_iso(checked_at)
    if dt is None:
        return True
    return (now - dt) > timedelta(days=STALENESS_DAYS)


def load_price_book(
    *,
    cache_dir: Path | None = None,
    now: Callable[[], datetime] | None = None,
    tavily_search: Callable[..., Any] | None = None,
    allow_refresh: bool = True,
) -> PriceBook:
    """Load the newest price snapshot; refresh via Tavily if stale (>30d) or absent.

    Args:
        cache_dir: snapshot directory (defaults to ``knowledge-base/pricing``).
        now: injectable clock returning an aware ``datetime`` (defaults to UTC now).
        tavily_search: injectable Tavily searcher (defaults to the real one lazily). Only
            called when a refresh is needed AND ``allow_refresh`` is True.
        allow_refresh: set False to read-only (no network) — used by tests / offline callers.

    Never raises: a missing/corrupt snapshot or a failed refresh degrades to the last good
    snapshot, else the baked-in default. The returned ``PriceBook`` records provenance
    (``path``/``source``/``stale``/``refreshed``/``warnings``).
    """
    cache_dir = cache_dir or PRICING_CACHE_DIR
    clock = now or _utcnow
    current = clock()

    snap_path = _newest_snapshot(cache_dir)
    data = _read_snapshot(snap_path) if snap_path else None
    warnings: list[str] = []

    if data is None:
        data = dict(DEFAULT_PRICE_BOOK)
        snap_path = None
        warnings.append("no pricing snapshot found — using baked-in default estimate basis")

    stale = _is_stale(str(data.get("checked_at", "")), current)
    refreshed = False

    if stale and allow_refresh:
        new_data = refresh_price_book_via_tavily(
            base=data, cache_dir=cache_dir, now=clock, tavily_search=tavily_search
        )
        if new_data is not None:
            data = new_data
            refreshed = True
            stale = False
            snap_path = _newest_snapshot(cache_dir)
        else:
            warnings.append(
                "pricing snapshot is >30d old and the Tavily refresh did not succeed — "
                "estimate uses the last known prices (may be out of date)"
            )

    return PriceBook(
        prices={k: float(v) for k, v in (data.get("prices") or {}).items()},
        checked_at=str(data.get("checked_at", "")),
        region=str(data.get("region", "us-west-2")),
        currency=str(data.get("currency", "USD")),
        source=str(data.get("source", "")),
        path=str(snap_path) if snap_path else None,
        stale=stale,
        refreshed=refreshed,
        warnings=warnings,
    )


# Price keys → the Tavily query used to refresh each. Kept explicit so a refresh only ever
# updates known keys (a bad search can never inject arbitrary fields).
_REFRESH_QUERIES: dict[str, str] = {
    "bedrock_opus_input_per_1k": "AWS Bedrock Claude Opus input token price per 1000 tokens us-west-2",
    "bedrock_opus_output_per_1k": "AWS Bedrock Claude Opus output token price per 1000 tokens us-west-2",
    "sagemaker_ml_m5_xlarge_hour": "AWS SageMaker ml.m5.xlarge training price per hour us-west-2",
}

# A price in USD extracted from search text must fall in a sane band, else it's ignored
# (guards against parsing a random number out of a snippet).
_PRICE_BAND = (0.0001, 1000.0)


def _extract_price(text: str) -> float | None:
    """Pull the first plausible ``$X.YZ`` USD figure from search text (band-guarded)."""
    for m in re.finditer(r"\$\s*([0-9]+(?:\.[0-9]+)?)", text or ""):
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        if _PRICE_BAND[0] <= val <= _PRICE_BAND[1]:
            return val
    return None


def refresh_price_book_via_tavily(
    *,
    base: dict[str, Any],
    cache_dir: Path | None = None,
    now: Callable[[], datetime] | None = None,
    tavily_search: Callable[..., Any] | None = None,
) -> dict[str, Any] | None:
    """Best-effort Tavily refresh → write a new dated snapshot. Returns the new book or None.

    Starts from ``base`` (so any key Tavily can't refresh keeps its prior value), queries each
    known price key, band-checks the extracted figure, stamps ``checked_at`` = now, and writes
    ``aws-pricing-sample-<date>.json``. Returns None (no write) if NOTHING could be refreshed
    or the Tavily key/network is unavailable — the caller then keeps the prior snapshot.
    """
    clock = now or _utcnow
    current = clock()

    searcher = tavily_search
    if searcher is None:
        try:
            from ..tools.research.tavily import tavily_search as _real
            searcher = _real
        except Exception:
            return None

    new_prices: dict[str, float] = dict(base.get("prices") or {})
    any_updated = False
    for key, query in _REFRESH_QUERIES.items():
        try:
            results = searcher(query, max_results=3)
        except Exception:  # nosec B112
            continue  # missing key / network / API error — skip this key
        for r in results or []:
            snippet = getattr(r, "snippet", "") or getattr(r, "title", "") or ""
            price = _extract_price(snippet)
            if price is not None:
                new_prices[key] = price
                any_updated = True
                break

    if not any_updated:
        return None

    new_book = {
        "checked_at": current.isoformat(),
        "region": base.get("region", "us-west-2"),
        "currency": base.get("currency", "USD"),
        "source": "tavily refresh (estimate basis; not a billing quote)",
        "prices": new_prices,
    }

    try:
        cache_dir = cache_dir or PRICING_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        date_tag = current.strftime("%Y-%m-%d")
        out = cache_dir / f"aws-pricing-sample-{date_tag}.json"
        out.write_text(json.dumps(new_book, indent=2) + "\n")
    except Exception:
        logger.debug("failed to write refreshed pricing snapshot", exc_info=True)
        # Still return the in-memory book so the estimate uses fresh numbers this run.

    return new_book
