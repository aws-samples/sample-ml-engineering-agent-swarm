"""``research_scan_flow`` — the Stage-0 SCHEDULED research scan (gap-closure §G5).

The schedulable sibling of [`research_intel_flow`](research_intel_flow.py:35): where the latter is
the **on-demand** Stage-0 scan, this flow runs ONE *scheduled tick* of the
[`ResearchScanScheduler`](../scheduler/research_scheduler.py) — invoke the existing research scan
per query, aggregate the results into one deduped digest, and record/deliver it. Registered as a
dispatchable capability (``research_scan_flow``) so it runs on ALL four surfaces (Chat/TUI/CLI/
Python-API) through the ONE capability funnel with zero per-surface code (ADR-010).

All CHEAP_READ (read-only public research sources; no AWS spend) — the public ``run_*`` entrypoint
runs without an approval prompt, mirroring ``research_intel_flow``. The hosted *cadence* (the
recurring trigger beyond this single tick) is the deploy-time EventBridge rule in
``infra/cdk_stacks/research_scan_stack.py``; this flow is the tickable unit that rule (or a
long-running process / a manual surface invocation) drives.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from ..scheduler.research_scheduler import ResearchScanScheduler

__all__ = ["run_research_scan_flow"]


def run_research_scan_flow(
    *,
    queries: list[str] | str | None = None,
    query: str = "",
    cadence: str = "weekly",
    index_name: str = "research",
    max_results: int = 5,
    theme: str = "",
    documents: list[dict[str, Any]] | None = None,
    index_corpus: bool | None = None,
    clock: Callable[[], datetime] | None = None,
    scan_fn: Callable[..., Any] | None = None,
    sink: Callable[[dict[str, Any]], Any] | None = None,
    seed: int = 0,
    **_ignored: Any,
) -> dict[str, Any]:
    """Run ONE scheduled research-scan tick and return the aggregated digest (CHEAP_READ).

    Builds a :class:`ResearchScanScheduler` for the configured ``queries`` + ``cadence`` and fires a
    single (always-due-on-first-call) tick: it invokes the existing Stage-0 scan per query,
    aggregates a deduped digest, and (optionally) delivers it through ``sink``. Surface-dispatchable
    by convention as ``research_scan_flow`` (the capability funnel resolves ``run_research_scan_flow``).

    Args:
        queries: The research queries to scan (a list, or a single string). Falls back to ``query``
            (a single query) or a sensible default so a bare invocation still produces a digest.
        query: Convenience single-query alias (used when ``queries`` is not supplied).
        cadence: ``hourly`` | ``daily`` | ``weekly`` | ``monthly`` (unknown → ``weekly``). Records the
            cadence the hosted EventBridge rule would use; the open §G5 decision confirms the final one.
        index_name / max_results / theme: forwarded to the underlying scan.
        clock / scan_fn / sink / seed: injection seams (callers may pass a fake clock + fake scan; no sleeps,
            no web). ``scan_fn`` defaults to the real ``run_research_intel_flow``.

    Returns:
        The aggregated digest record dict (``scan_id``, ``cadence``, ``tick``, ``digest``,
        ``findings``, ``backlog``, ``total_finding_count``, ``new_finding_count``, ``next_run`` …)
        with ``status="completed"``.
    """
    # Normalize the query input into a non-empty list.
    if isinstance(queries, str):
        query_list = [queries]
    elif queries:
        query_list = [str(q) for q in queries]
    elif query:
        query_list = [query]
    else:
        query_list = ["ML model monitoring drift detection predictive maintenance"]

    # Forward optional scan controls (index_corpus / documents) to the underlying scan so a
    # surface invocation can keep the tick light + fully offline (e.g. index_corpus=False).
    scan_kwargs: dict[str, Any] = {}
    if documents is not None:
        scan_kwargs["documents"] = documents
    if index_corpus is not None:
        scan_kwargs["index_corpus"] = index_corpus

    kwargs: dict[str, Any] = {
        "queries": query_list,
        "cadence": cadence,
        "index_name": index_name,
        "max_results": max_results,
        "theme": theme,
        "seed": seed,
    }
    if scan_kwargs:
        kwargs["scan_kwargs"] = scan_kwargs
    if clock is not None:
        kwargs["clock"] = clock
    if scan_fn is not None:
        kwargs["scan_fn"] = scan_fn
    if sink is not None:
        kwargs["sink"] = sink

    scheduler = ResearchScanScheduler(**kwargs)
    record = scheduler.tick()
    if record is None:  # defensive — the first tick is always due
        return {"status": "skipped", "cadence": scheduler.cadence,
                "scan_id": scheduler.scan_id, "next_run": scheduler.next_run_time().isoformat(),
                "reason": "not due"}
    return {"status": "completed", **record}
