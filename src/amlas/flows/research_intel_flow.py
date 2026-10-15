"""``research_intel_flow`` — the Stage-0 research/intelligence sequence (roadmap M6).

The invariant Stage-0 sequence (doc 06 §5.3 ``research_intel_flow``): **scan external + indexed
research → triage by reproducibility → index the corpus → generate a digest → emit a backlog**.
All CHEAP_READ (read/reasoning only; no spend), so the public entrypoint runs without an approval
prompt. Composes the Stage-0 tools (``tools/research_neutral.py``) over the RetrievalBackend port.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..tools import research_neutral as R
from ..tools import retrieval_neutral as RAG
from ._util import raw

__all__ = ["ResearchIntelFlowResult", "run_research_intel_flow"]


@dataclass
class ResearchIntelFlowResult:
    """The invariant output contract of one ``research_intel_flow`` run."""

    status: str  # "completed" | "failed"
    query: str = ""
    findings: list[dict[str, Any]] = field(default_factory=list)
    triaged: list[dict[str, Any]] = field(default_factory=list)
    backlog: list[dict[str, Any]] = field(default_factory=list)
    index: dict[str, Any] = field(default_factory=dict)
    digest: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)


def run_research_intel_flow(
    *,
    query: str,
    index_name: str = "research",
    max_results: int = 5,
    index_corpus: bool = True,
    documents: list[dict[str, Any]] | None = None,
    theme: str = "",
) -> ResearchIntelFlowResult:
    """Run the Stage-0 research-intelligence sequence; return findings + triage + digest + backlog.

    Fixed steps (CHEAP_READ; RetrievalBackend local offline ↔ Bedrock KB managed):
      1. **index** the research corpus (T-RAG-01) so the scan can retrieve against it;
      2. **scan** external + indexed research for ``query`` (T-PR-01..04, evolves web_retrieve);
      3. **triage** the findings by reproducibility (T-PR-05);
      4. **digest** the triaged findings (T-PR-11) the research_curator persists.
    """
    # ── 1. index the research corpus (so the scan retrieves against it) ─────────
    index: dict[str, Any] = {}
    if index_corpus:
        index = raw(RAG.index_research_corpus)(index_name=index_name, documents=documents,
                                               include_memory_bank=documents is None)

    # ── 2. scan external + indexed research ─────────────────────────────────────
    scan = raw(R.scan_research)(query=query, max_results=max_results, index_name=index_name)
    findings = scan.get("findings", [])

    # ── 3. triage by reproducibility ────────────────────────────────────────────
    triage = raw(R.triage_reproducibility)(findings=findings)

    # ── 4. generate the digest ──────────────────────────────────────────────────
    digest = raw(R.generate_research_digest)(findings=triage.get("triaged", findings), theme=theme or query)

    return ResearchIntelFlowResult(
        status="completed", query=query, findings=findings,
        triaged=triage.get("triaged", []), backlog=triage.get("backlog", []),
        index=index, digest=digest,
        detail={"sources": scan.get("sources", []), "high_count": triage.get("high_count", 0)},
    )
