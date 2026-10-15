"""Backend-neutral RAG / retrieval tools — T-RAG-01..06 (roadmap M6; ADR-017).

The doc 05 §4.4 EVOLVE pattern for the RetrievalBackend: each tool resolves the
``RetrievalBackend`` PORT via the ``BackendRegistry`` and delegates to the bound adapter
(``LocalRetrievalBackend`` TF-IDF offline, ``BedrockKnowledgeBasesBackend`` managed). NEW
M6 tools, ADDITIVE.

Tools (doc 06 §RetrievalBackend, T-RAG-01..06):
  * ``index_research_corpus`` (T-RAG-01) — chunk and index supplied research
    documents into a named index.
  * ``chunk_documents`` (T-RAG-02) — split documents into overlapping chunks.
  * ``query_research_corpus`` (T-RAG-06) — retrieve the top-k relevant chunks for a query.
  * ``build_retrieval_index`` (T-RAG-05) — build an index from pre-chunked content.
  * ``build_metadata_filters`` (T-RAG-04) — assemble a metadata-filter dict for narrowed
    retrieval (e.g. ``{"stage": "0", "kind": "arxiv"}``).
  * ``tune_retrieval`` (T-RAG-03) — recommend chunk-size/overlap/top-k settings (reasoning).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from strands import tool

from ..backends.ports import RetrievalBackend
from ..backends.registry import REGISTRY, ResolvedBackend
from ..errors import tool_error_boundary

__all__ = [
    "build_metadata_filters",
    "build_retrieval_index",
    "chunk_documents",
    "index_research_corpus",
    "load_research_corpus_docs",
    "query_research_corpus",
    "tune_retrieval",
]


def _retrieval(steering: Any = None) -> ResolvedBackend:
    """Resolve the RetrievalBackend (local TF-IDF offline, Bedrock KB managed)."""
    return REGISTRY.resolve(RetrievalBackend, steering=steering)


def _repo_root() -> Path:
    """Repo root = parents[3] of ``src/amlas/tools/retrieval_neutral.py``."""
    return Path(__file__).resolve().parents[3]


def load_research_corpus_docs(*, extra_paths: list[str] | None = None, max_chars: int = 20000) -> list[dict[str, Any]]:
    """Load caller-supplied research documents from ``extra_paths``.

    Returns a list of ``{text, source, metadata}`` docs the RAG tools index. Pure filesystem
    read (offline-safe); each doc's text is truncated to ``max_chars`` to keep the index light.
    """
    docs: list[dict[str, Any]] = []
    for extra in (extra_paths or []):
        ep = Path(extra)
        if ep.is_file():
            try:
                text = ep.read_text(encoding="utf-8", errors="ignore")[:max_chars]
                docs.append({"text": text, "source": str(ep), "metadata": {"kind": "extra"}})
            except OSError:
                pass
    return docs


@tool
@tool_error_boundary(source="tools.retrieval_neutral")
def chunk_documents(text: str, chunk_size: int = 512, overlap: int = 64,
                    source: str = "", metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Split ``text`` into overlapping retrieval chunks via :class:`RetrievalBackend` (T-RAG-02)."""
    try:
        resolved = _retrieval()
        chunks = resolved.adapter.chunk(text=text, chunk_size=chunk_size, overlap=overlap,
                                        source=source, metadata=metadata)
        return {
            "chunks": [{"chunk_id": c.chunk_id, "text": c.text, "source": c.source,
                        "metadata": c.metadata} for c in chunks],
            "chunk_count": len(chunks), "adapter": resolved.adapter_name,
        }
    except Exception as e:
        return {"success": False, "error": f"chunk_documents failed: {e}", "chunks": [], "chunk_count": 0}


@tool
@tool_error_boundary(source="tools.retrieval_neutral")
def build_retrieval_index(index_name: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a retrieval index ``index_name`` from pre-chunked content via :class:`RetrievalBackend` (T-RAG-05)."""
    try:
        from ..backends.types import Chunk

        resolved = _retrieval()
        chunk_objs = [
            Chunk(chunk_id=c.get("chunk_id", f"c{i}"), text=str(c.get("text", "")),
                  source=str(c.get("source", "")), metadata=c.get("metadata") or {})
            for i, c in enumerate(chunks or [])
        ]
        ref = resolved.adapter.build_index(name=index_name, chunks=chunk_objs)
        return {"index_name": ref.name, "backend": ref.backend, "doc_count": ref.doc_count,
                "chunk_count": ref.chunk_count, "adapter": resolved.adapter_name}
    except Exception as e:
        return {"success": False, "error": f"build_retrieval_index failed: {e}", "index_name": index_name}


@tool
@tool_error_boundary(source="tools.retrieval_neutral")
def index_research_corpus(index_name: str = "research", documents: list[dict[str, Any]] | None = None,
                          chunk_size: int = 512, overlap: int = 64,
                          include_memory_bank: bool = True) -> dict[str, Any]:
    """Chunk + index a research corpus into ``index_name`` via :class:`RetrievalBackend` (T-RAG-01).

    When ``documents`` is omitted and ``include_memory_bank`` is set, caller-supplied
    paths are loaded through :func:`load_research_corpus_docs`.
    """
    try:
        docs = list(documents or [])
        if include_memory_bank and not docs:
            docs = load_research_corpus_docs()
        resolved = _retrieval()
        ref = resolved.adapter.index_corpus(name=index_name, documents=docs,
                                            chunk_size=chunk_size, overlap=overlap)
        return {"index_name": ref.name, "backend": ref.backend, "doc_count": ref.doc_count,
                "chunk_count": ref.chunk_count, "adapter": resolved.adapter_name}
    except Exception as e:
        return {"success": False, "error": f"index_research_corpus failed: {e}", "index_name": index_name,
                "doc_count": 0, "chunk_count": 0}


@tool
@tool_error_boundary(source="tools.retrieval_neutral")
def query_research_corpus(index_name: str, query: str, top_k: int = 5,
                          filters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Retrieve the ``top_k`` relevant chunks for ``query`` via :class:`RetrievalBackend` (T-RAG-06)."""
    try:
        resolved = _retrieval()
        hits = resolved.adapter.query(name=index_name, query=query, top_k=top_k, filters=filters)
        return {
            "query": query, "index_name": index_name,
            "results": [{"text": h.chunk.text, "source": h.chunk.source, "score": h.score,
                         "rank": h.rank, "metadata": h.chunk.metadata} for h in hits],
            "result_count": len(hits), "adapter": resolved.adapter_name,
        }
    except Exception as e:
        return {"success": False, "error": f"query_research_corpus failed: {e}", "results": [], "result_count": 0}


@tool
@tool_error_boundary(source="tools.retrieval_neutral")
def build_metadata_filters(stage: str | None = None, kind: str | None = None,
                           extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble a metadata-filter dict for narrowed retrieval (T-RAG-04).

    Pure reasoning: builds the ``{key: value}`` filter the query side applies (e.g. restrict to
    a lifecycle stage or a source kind). Returns ``{"filters": {...}}``.
    """
    filters: dict[str, Any] = {}
    if stage is not None:
        filters["stage"] = str(stage)
    if kind is not None:
        filters["kind"] = str(kind)
    if extra:
        filters.update(extra)
    return {"filters": filters, "filter_count": len(filters)}


@tool
@tool_error_boundary(source="tools.retrieval_neutral")
def tune_retrieval(corpus_size_docs: int = 10, avg_doc_chars: int = 4000,
                   precision_priority: bool = False) -> dict[str, Any]:
    """Recommend chunk-size / overlap / top-k retrieval settings (T-RAG-03, pure reasoning).

    Heuristic: smaller chunks + higher top-k favor precision; larger chunks favor recall/context.
    Returns ``{chunk_size, overlap, top_k, rationale}``.
    """
    if precision_priority:
        chunk_size, overlap, top_k = 256, 32, 8
        rationale = "precision-priority: smaller chunks + higher top-k for tighter relevance"
    elif avg_doc_chars > 8000:
        chunk_size, overlap, top_k = 768, 96, 5
        rationale = "large documents: bigger chunks preserve context, moderate top-k"
    else:
        chunk_size, overlap, top_k = 512, 64, 5
        rationale = "balanced default: 512-word chunks, 64 overlap, top-5"
    return {"chunk_size": chunk_size, "overlap": overlap, "top_k": top_k,
            "rationale": rationale, "corpus_size_docs": corpus_size_docs}
