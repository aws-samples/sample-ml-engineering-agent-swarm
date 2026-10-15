"""Traceability — the executable story→flow→tool→backend→adapter→run chain (doc 05 §8.5 / roadmap M8).

This makes the doc-00 traceability matrix **executable**: every capability invocation records a
:class:`TraceRecord` linking the full lineage chain —

    Story ID → flow id → tool id(s) → backend port → adapter (+ resolution reason) → run/experiment id

— so the Phase-8 join (story↔tool↔backend↔run, doc 05 §8.1) is no longer a static spreadsheet but a
queryable record emitted at runtime. It complements (does not replace) the M0 registry
resolution-trace (``ResolvedBackend.port → adapter → reason``): this module *aggregates* that
per-port resolution trace into one end-to-end chain per run and surfaces it as:

  * a queryable in-process :class:`TraceLedger` (``record`` + ``get`` + ``query`` + ``matrix``);
  * an OTEL span attribute set (:func:`as_span_attributes`) the ``telemetry/trace.py`` ``TraceSpan``
    carries (the doc 05 §8.1 ``port→adapter(+reason)`` + AWS-resource-ARN span fields);
  * a small fetch tool (``tools/trace_neutral.py`` ``fetch_run_trace``) so an agent / surface can
    ask "show me the lineage for run X."

Pure-Python, boto3-free, deterministic — the trace is descriptive metadata, not an AWS call.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "LEDGER",
    "BackendHop",
    "TraceLedger",
    "TraceRecord",
    "as_span_attributes",
    "record_capability_trace",
]


# ──────────────────────────────────────────────────────────────────────────────
# Value types — one backend resolution hop + the full per-run chain
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class BackendHop:
    """One port→adapter resolution hop in a run's lineage (the M0 registry trace, aggregated).

    ``port`` is the Protocol name (``"TrainingBackend"``); ``adapter`` the bound selector
    (``"sagemaker"`` / ``"local"`` / ``"bedrock"``); ``reason`` the registry's resolution reason
    (``"explicit env TRAINING_BACKEND=bedrock"`` …); ``resource_arn`` the AWS resource the adapter
    touched when known (the doc 05 §8.1 ARN span field).
    """

    port: str
    adapter: str
    reason: str = ""
    resource_arn: str = ""


@dataclass
class TraceRecord:
    """The executable story→flow→tool→backend→adapter→run lineage for ONE capability invocation.

    The doc-00 traceability-matrix row, emitted at runtime: ``story_id`` (the user-story served) →
    ``flow_id`` (the dispatched flow) → ``tool_ids`` (the tools it called) → ``backends`` (the
    port→adapter hops) → ``run_id`` (the leaderboard/kanban run/experiment id). ``status`` records
    the invocation outcome; ``detail`` carries anything extra (surface, cost, etc.).
    """

    trace_id: str
    story_id: str
    flow_id: str
    tool_ids: list[str] = field(default_factory=list)
    backends: list[BackendHop] = field(default_factory=list)
    run_id: str = ""
    status: str = "completed"
    ts: float = 0.0
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain JSON-able dict (for the fetch tool + the OTEL span)."""
        return {
            "trace_id": self.trace_id,
            "story_id": self.story_id,
            "flow_id": self.flow_id,
            "tool_ids": list(self.tool_ids),
            "backends": [
                {"port": h.port, "adapter": h.adapter, "reason": h.reason, "resource_arn": h.resource_arn}
                for h in self.backends
            ],
            "run_id": self.run_id,
            "status": self.status,
            "ts": self.ts,
            "detail": dict(self.detail),
        }

    @property
    def chain(self) -> str:
        """The human-readable lineage chain: ``story → flow → tool,tool → port:adapter → run``."""
        tools = ",".join(self.tool_ids) or "-"
        hops = " ".join(f"{h.port}:{h.adapter}" for h in self.backends) or "-"
        return f"{self.story_id} → {self.flow_id} → {tools} → {hops} → {self.run_id or '-'}"


# ──────────────────────────────────────────────────────────────────────────────
# The ledger — record / get / query / matrix (in-process; optional JSONL sink)
# ──────────────────────────────────────────────────────────────────────────────
class TraceLedger:
    """Process-wide queryable trace ledger — the executable traceability matrix (doc 05 §8.5).

    ``record`` appends a :class:`TraceRecord`; ``get`` fetches by run/trace id; ``query`` filters by
    story/flow; ``matrix`` returns the full story→tool→backend→run join (the Phase-8 matrix). An
    optional ``path`` mirrors each record to a JSONL sink (``runs/<run>/traceability.jsonl``-style)
    so the trace survives the process — but the in-process list is the source of truth for isolated validation.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._records: list[TraceRecord] = []
        self._path = Path(path) if path else None

    def record(self, rec: TraceRecord) -> TraceRecord:
        """Append ``rec`` (stamping ts if unset) and mirror to the JSONL sink when configured."""
        if not rec.ts:
            rec.ts = time.time()
        self._records.append(rec)
        if self._path is not None:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec.to_dict(), default=str) + "\n")
            except Exception:  # nosec B110
                pass
        return rec

    def get(self, *, run_id: str | None = None, trace_id: str | None = None) -> TraceRecord | None:
        """Fetch the most recent record matching ``run_id`` OR ``trace_id`` (or ``None``)."""
        for rec in reversed(self._records):
            if trace_id and rec.trace_id == trace_id:
                return rec
            if run_id and rec.run_id == run_id:
                return rec
        return None

    def query(self, *, story_id: str | None = None, flow_id: str | None = None) -> list[TraceRecord]:
        """Return all records matching the given ``story_id`` / ``flow_id`` filters (AND)."""
        out = self._records
        if story_id:
            out = [r for r in out if r.story_id == story_id]
        if flow_id:
            out = [r for r in out if r.flow_id == flow_id]
        return list(out)

    def matrix(self) -> list[dict[str, Any]]:
        """Return the executable traceability matrix — one joined row per recorded invocation."""
        return [
            {
                "story_id": r.story_id,
                "flow_id": r.flow_id,
                "tool_ids": list(r.tool_ids),
                "ports": [h.port for h in r.backends],
                "adapters": [h.adapter for h in r.backends],
                "run_id": r.run_id,
                "status": r.status,
                "chain": r.chain,
            }
            for r in self._records
        ]

    def clear(self) -> None:
        """Reset the ledger (used by tests for isolation)."""
        self._records.clear()

    def all(self) -> list[TraceRecord]:
        """Return every recorded trace (insertion order)."""
        return list(self._records)


# The process-wide singleton the capability funnel + tools emit into.
LEDGER = TraceLedger()


# ──────────────────────────────────────────────────────────────────────────────
# Recording + OTEL span emission
# ──────────────────────────────────────────────────────────────────────────────
def record_capability_trace(
    *,
    story_id: str,
    flow_id: str,
    tool_ids: list[str] | None = None,
    backends: list[BackendHop] | None = None,
    run_id: str = "",
    status: str = "completed",
    detail: dict[str, Any] | None = None,
    ledger: TraceLedger | None = None,
) -> TraceRecord:
    """Record one story→flow→tool→backend→run trace into the ledger (the §8.5 emission point).

    Called by the capability funnel (and any flow that wants explicit lineage) after an invocation.
    ``backends`` is the list of :class:`BackendHop`s aggregated from the registry resolution-trace
    (``ResolvedBackend.port/adapter_name/reason``). Returns the recorded :class:`TraceRecord`.
    """
    rec = TraceRecord(
        trace_id=uuid.uuid4().hex[:16],
        story_id=story_id,
        flow_id=flow_id,
        tool_ids=list(tool_ids or []),
        backends=list(backends or []),
        run_id=run_id,
        status=status,
        detail=dict(detail or {}),
    )
    return (ledger or LEDGER).record(rec)


def as_span_attributes(rec: TraceRecord) -> dict[str, Any]:
    """Flatten a :class:`TraceRecord` into OTEL span attributes (doc 05 §8.1 span fields).

    Produces the ``amlas.trace.*`` attribute set a ``telemetry/trace.py`` ``TraceSpan`` carries:
    ``story_id`` / ``flow_id`` / ``tool_ids`` / ``run_id`` plus the per-hop ``port→adapter(+reason)``
    + AWS-resource-ARN fields — so the trace span itself encodes the full lineage (the Phase-8 join
    becomes a span query). Values are span-safe scalars / comma-joined strings.
    """
    attrs: dict[str, Any] = {
        "amlas.trace.story_id": rec.story_id,
        "amlas.trace.flow_id": rec.flow_id,
        "amlas.trace.tool_ids": ",".join(rec.tool_ids),
        "amlas.trace.run_id": rec.run_id,
        "amlas.trace.status": rec.status,
        "amlas.trace.ports": ",".join(h.port for h in rec.backends),
        "amlas.trace.adapters": ",".join(h.adapter for h in rec.backends),
    }
    for h in rec.backends:
        attrs[f"amlas.backend.{h.port}.adapter"] = h.adapter
        if h.reason:
            attrs[f"amlas.backend.{h.port}.reason"] = h.reason
        if h.resource_arn:
            attrs[f"amlas.backend.{h.port}.arn"] = h.resource_arn
    return attrs
