"""``ResearchScanScheduler`` — the Stage-0 scheduled/recurring research scan (gap-closure §G5).

Beyond the on-demand [`run_research_intel_flow`](../flows/research_intel_flow.py:35), this adds a
**schedulable, cadence-driven** orchestration tick: on each *due* tick it invokes the existing
research scan once per configured query, aggregates the per-query results into ONE deduped digest,
records/delivers it (optional sink), and advances the next-run time by the cadence interval. It
closes the "every Monday scan + digest" story (master plan §G5).

Design invariants (the master plan §G5 contract + the TDD test):

  * **Reuse, not re-implement** — ``scan_fn`` defaults to ``run_research_intel_flow`` (the existing
    Stage-0 scan); the scheduler only adds the *cadence + aggregation + idempotency* layer (it does
    NOT overload :class:`ProjectScheduler`, which runs ML *projects* — scheduler/scheduler.py:50).
  * **Injectable clock** — no real ``time.sleep``; the cadence is evaluated against an injected
    ``clock() -> datetime`` so ticks are deterministic + directly verifiable without waiting.
  * **Idempotent** — (1) a tick before the next cadence boundary is a no-op (the scan is NOT
    re-invoked); (2) findings already surfaced in a prior tick are deduped (``new_finding_count``
    counts only the never-before-seen ones).
  * **Deterministic** — seeded; the aggregated findings are sorted by a stable key, so two identical
    runs produce a byte-for-byte-identical digest + a stable, AWS-rule-name-safe ``scan_id``.
  * **CHEAP_READ** — the scan hits read-only public research sources (no AWS spend); the
    ``schedule_research_scan`` action is already classified CHEAP_READ (approvals.py:91), and the
    ``research_scan_flow`` capability is registered non-gated so it never prompts for approval.

The hosted cadence is realized at deploy-time by an EventBridge schedule rule (CDK) that triggers
the same scan on a cron — see ``infra/cdk_stacks/research_scan_stack.py`` (the parameterized
cadence + digest destination). This in-process scheduler is the tickable core that the rule (or a
long-running process) drives.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

__all__ = ["CADENCE_INTERVALS", "ResearchScanScheduler", "normalize_cadence", "scan_id_for"]

# The supported cadence shorthands → their interval. ``weekly`` is the sensible default the
# §G5 open-decision proposes (e.g. "weekly Monday"); the human confirms the final cadence.
CADENCE_INTERVALS: dict[str, timedelta] = {
    "hourly": timedelta(hours=1),
    "daily": timedelta(days=1),
    "weekly": timedelta(weeks=1),
    "monthly": timedelta(days=30),
}
_DEFAULT_CADENCE = "weekly"

# AWS resource/rule name safety: hyphens, no underscores ([a-zA-Z0-9]{1,63}(-*[a-zA-Z0-9+.-]){0,63}).
_NAME_SANITIZE = re.compile(r"[^a-zA-Z0-9]+")


def normalize_cadence(cadence: str | None) -> str:
    """Normalize a cadence shorthand; unknown / empty → the weekly default (never raises)."""
    c = (cadence or "").strip().lower()
    return c if c in CADENCE_INTERVALS else _DEFAULT_CADENCE


def scan_id_for(queries: list[str], cadence: str) -> str:
    """Return a stable, AWS-safe scan ID derived with SHA-256."""
    payload = ("|".join(sorted(queries)) + "@" + cadence).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:10]
    return f"research-scan-{cadence}-{digest}"


def _finding_key(finding: dict[str, Any]) -> str:
    """Stable dedupe/sort key for a finding — url first (canonical), else title, else snippet."""
    for k in ("url", "title", "snippet"):
        v = str(finding.get(k, "")).strip()
        if v:
            return v
    return repr(sorted(finding.items()))


def _default_clock() -> datetime:
    """Wall-clock now (UTC) — replaced by an injected clock in tests (no sleeps)."""
    return datetime.now(UTC)


@dataclass
class ResearchScanScheduler:
    """A cadence-driven scheduler that periodically runs the Stage-0 research scan + digests it.

    Args:
        queries: The research queries to scan each tick (one scan invocation per query).
        cadence: ``hourly`` | ``daily`` | ``weekly`` | ``monthly`` (unknown → ``weekly``).
        clock: Injected ``() -> datetime`` (UTC); defaults to wall-clock. Callers may pass a fake clock
            so cadence evaluation is deterministic with NO real waiting.
        scan_fn: The scan callable invoked per query; defaults to ``run_research_intel_flow`` (the
            existing Stage-0 scan) — reuse, not re-implementation.
        sink: Optional ``(record: dict) -> Any`` delivery hook (S3 / channel / kanban — the §G5
            open decision); when supplied the tick record's ``delivered`` flag is set True.
        index_name / max_results / theme: forwarded to the scan.
        seed: Determinism seed (kept for parity with the seeded-RNG convention; aggregation is
            already deterministic via sorted finding keys).
    """

    queries: list[str]
    cadence: str = _DEFAULT_CADENCE
    clock: Callable[[], datetime] = _default_clock
    scan_fn: Callable[..., Any] | None = None
    sink: Callable[[dict[str, Any]], Any] | None = None
    index_name: str = "research"
    max_results: int = 5
    theme: str = ""
    seed: int = 0
    scan_kwargs: dict[str, Any] | None = None

    # internal state (not constructor args)
    _next_run: datetime | None = field(default=None, init=False, repr=False)
    _tick_count: int = field(default=0, init=False, repr=False)
    _seen_keys: set[str] = field(default_factory=set, init=False, repr=False)

    def __post_init__(self) -> None:
        self.cadence = normalize_cadence(self.cadence)
        if self.scan_fn is None:
            # Lazy import keeps the scheduler import cheap + avoids a cycle at module load.
            from ..flows.research_intel_flow import run_research_intel_flow

            self.scan_fn = run_research_intel_flow
        # The first tick is due immediately (next_run == start) so a freshly-registered
        # schedule fires on its first evaluation, then settles onto the cadence grid.
        self._next_run = self.clock()

    # ── cadence introspection ────────────────────────────────────────────────
    @property
    def interval(self) -> timedelta:
        """The cadence interval (a ``timedelta``)."""
        return CADENCE_INTERVALS[self.cadence]

    def next_run_time(self) -> datetime:
        """The next scheduled run time (the cadence boundary the clock must reach to fire)."""
        assert self._next_run is not None
        return self._next_run

    def due(self, *, now: datetime | None = None) -> bool:
        """True when the clock has reached/passed the next-run boundary (a tick would fire)."""
        moment = now or self.clock()
        return self._next_run is not None and moment >= self._next_run

    @property
    def scan_id(self) -> str:
        """The stable, AWS-rule-name-safe scan id for this schedule."""
        return scan_id_for(self.queries, self.cadence)

    # ── the tick (the periodic beat) ──────────────────────────────────────────
    def tick(self, *, now: datetime | None = None) -> dict[str, Any] | None:
        """Run ONE scheduled tick IF due: scan each query → aggregate digest → record/deliver.

        Returns the aggregated digest record dict when it fired, or ``None`` when not due (a
        no-op — the scan is NOT re-invoked, the cadence is respected). Advances ``next_run`` by
        exactly one cadence interval on a fire.
        """
        moment = now or self.clock()
        if not self.due(now=moment):
            return None  # not due → no-op (idempotent within the cadence window)

        fired_at = moment
        self._tick_count += 1

        # ── scan each query (the existing Stage-0 scan), collect findings + backlog ──
        per_query: list[dict[str, Any]] = []
        all_findings: list[dict[str, Any]] = []
        all_backlog: list[dict[str, Any]] = []
        assert self.scan_fn is not None
        extra = dict(self.scan_kwargs or {})
        for query in self.queries:
            res = self.scan_fn(query=query, index_name=self.index_name,
                               max_results=self.max_results, theme=self.theme, **extra)
            findings = _as_list(res, "findings")
            backlog = _as_list(res, "backlog")
            all_findings.extend(findings)
            all_backlog.extend(backlog)
            detail = _as_attr(res, "detail", {}) or {}
            sources = detail.get("sources", []) if isinstance(detail, dict) else []
            per_query.append({"query": query, "finding_count": len(findings),
                              "status": _as_attr(res, "status", "completed"),
                              "sources": list(sources)})

        # ── aggregate + dedupe (deterministic: sorted by stable key) ──
        unique: dict[str, dict[str, Any]] = {}
        for f in all_findings:
            unique.setdefault(_finding_key(f), f)
        aggregated = [unique[k] for k in sorted(unique.keys())]

        new_keys = [k for k in unique if k not in self._seen_keys]
        new_finding_count = len(new_keys)
        self._seen_keys.update(unique.keys())

        backlog_unique: dict[str, dict[str, Any]] = {}
        for b in all_backlog:
            backlog_unique.setdefault(_finding_key(b), b)
        backlog = [backlog_unique[k] for k in sorted(backlog_unique.keys())]

        digest_md = _render_digest(self.theme or ", ".join(self.queries), aggregated,
                                   per_query, fired_at)

        # advance the cadence grid by exactly one interval (deterministic).
        self._next_run = fired_at + self.interval

        record: dict[str, Any] = {
            "scan_id": self.scan_id,
            "cadence": self.cadence,
            "tick": self._tick_count,
            "fired_at": fired_at.isoformat(),
            "next_run": self._next_run.isoformat(),
            "queries": list(self.queries),
            "per_query": per_query,
            "findings": aggregated,
            "backlog": backlog,
            "total_finding_count": len(aggregated),
            "new_finding_count": new_finding_count,
            "digest": digest_md,
            "delivered": False,
        }

        # ── record / deliver the digest (optional sink: S3 / channel / kanban) ──
        if self.sink is not None:
            self.sink(record)
            record["delivered"] = True

        return record


def _as_attr(res: Any, name: str, default: Any) -> Any:
    """Read ``name`` from a result object (dataclass) or dict, with a default."""
    if isinstance(res, dict):
        return res.get(name, default)
    return getattr(res, name, default)


def _as_list(res: Any, name: str) -> list[dict[str, Any]]:
    """Read a list field from a result object/dict (empty list when absent/None)."""
    val = _as_attr(res, name, None)
    return list(val) if isinstance(val, list) else []


def _render_digest(theme: str, findings: list[dict[str, Any]],
                   per_query: list[dict[str, Any]], fired_at: datetime) -> str:
    """Compile a deterministic markdown digest from the aggregated findings (sorted)."""
    lines = [f"# Scheduled research digest: {theme}", "",
             f"- fired_at: {fired_at.isoformat()}",
             f"- {len(findings)} unique findings across {len(per_query)} queries.", ""]
    for i, f in enumerate(findings, 1):
        title = str(f.get("title") or f.get("url") or "(untitled)")
        snippet = str(f.get("snippet", ""))[:120]
        lines.append(f"{i}. {title} — {snippet}")
    return "\n".join(lines)
