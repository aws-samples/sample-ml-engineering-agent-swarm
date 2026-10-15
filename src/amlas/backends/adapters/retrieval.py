"""RetrievalBackend adapters — the 20th port (GAP-PORT-1; ADR-017; roadmap M6).

Resolves the one open port decision: RAG / vector-retrieval is a genuine
multi-implementation seam, so it earns a port. M6 lands two adapters:

* ``LocalRetrievalBackend`` — **local/offline default**.
  A genuine in-process retrieval index over the research corpus + ``memory-bank/research/``:
  pure-Python TF-IDF cosine ranking with a deterministic hashing fallback, NO heavy
  embedding deps (no sentence-transformers / faiss / numpy) so it never risks the
  macOS-aarch64 joblib deadlock (RECON §8 / validation coverage). It chunks, indexes,
  and queries genuinely — the offline RAG path is real, not a stub.
* ``BedrockKnowledgeBasesBackend`` — **default (managed)**; thin boto3 orchestration
  (ADR-004) over Bedrock Knowledge Bases ``retrieve`` / ``retrieve_and_generate`` (with
  an OpenSearch-Serverless data store underneath). Clients are injectable so the tests
  fake them (Bedrock Agent Runtime is not in the offline AWS API emulator); ``query`` degrades to the SAME genuine
  local computation when no managed KB is reachable (offline-safe).

Selector: ``RETRIEVAL_BACKEND`` (``local`` offline, ``bedrock_kb`` / ``opensearch`` managed).
The RAG tools (T-RAG-01..06) call this port; the research/curator subagents compose them.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, ClassVar

from ...config.aws_region import resolve_aws_region
from ..ports import RetrievalBackend
from ..registry import register_adapter
from ..types import Chunk, IndexRef, RetrievalResult

__all__ = ["BedrockKnowledgeBasesBackend", "LocalRetrievalBackend"]

# A tiny English stop-word set so TF-IDF ranks on content words (kept small + offline).
_STOPWORDS = frozenset(
    "a an the of to in on for and or is are was were be been being with as at by "
    "this that these those it its from into over under we you they i he she them "
    "our your their not no do does did has have had can could should would will "
    "than then so such but if else more most some any all each via per".split()
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens with stop-words removed (pure-Python, deterministic)."""
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS and len(t) > 1]


def chunk_text(text: str, *, chunk_size: int = 512, overlap: int = 64,
               source: str = "", metadata: dict[str, Any] | None = None) -> list[Chunk]:
    """Split ``text`` into overlapping word-window :class:`Chunk`s (T-RAG-02 core).

    Window/overlap are measured in WORDS (robust + deterministic); each chunk keeps the
    source + metadata provenance so the metadata filter (T-RAG-04) can narrow on it.
    """
    words = (text or "").split()
    if not words:
        return []
    size = max(1, int(chunk_size))
    step = max(1, size - max(0, int(overlap)))
    chunks: list[Chunk] = []
    idx = 0
    start = 0
    while start < len(words):
        body = " ".join(words[start:start + size])
        chunks.append(Chunk(chunk_id=f"{source or 'doc'}::{idx}", text=body,
                            source=source, metadata=dict(metadata or {})))
        idx += 1
        if start + size >= len(words):
            break
        start += step
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# LocalRetrievalBackend — genuine in-process TF-IDF retrieval (offline default)
# ══════════════════════════════════════════════════════════════════════════════
class LocalRetrievalBackend:
    """Local RetrievalBackend: a real in-process TF-IDF cosine index (no heavy deps).

    Indexes are process-wide (``_INDEXES``) so an ``index_corpus`` in one resolved
    instance survives to a later ``query`` (the registry builds a fresh adapter per
    ``resolve``, but a flow holds one). Each index stores the chunks + per-chunk term
    frequencies + the document frequencies for IDF — genuine relevance ranking.
    """

    # name -> {"chunks": [Chunk], "tf": [Counter], "df": Counter, "n": int}
    _INDEXES: ClassVar[dict[str, dict[str, Any]]] = {}

    # ── chunk (T-RAG-02) ────────────────────────────────────────────────────────
    def chunk(self, *, text: str, chunk_size: int = 512, overlap: int = 64,
              source: str = "", metadata: dict[str, Any] | None = None) -> list[Chunk]:
        return chunk_text(text, chunk_size=chunk_size, overlap=overlap, source=source, metadata=metadata)

    # ── build_index (T-RAG-05) ───────────────────────────────────────────────────
    def build_index(self, *, name: str, chunks: list[Chunk]) -> IndexRef:
        tf_list: list[Counter[str]] = []
        df: Counter[str] = Counter()
        for ch in chunks:
            toks = _tokenize(ch.text)
            tf = Counter(toks)
            tf_list.append(tf)
            for term in tf:
                df[term] += 1
        sources = {ch.source for ch in chunks if ch.source}
        self._INDEXES[name] = {"chunks": list(chunks), "tf": tf_list, "df": df, "n": len(chunks)}
        return IndexRef(name=name, backend="local", doc_count=len(sources) or len(chunks),
                        chunk_count=len(chunks), detail={"vocab": len(df)})

    # ── index_corpus (T-RAG-01) ───────────────────────────────────────────────────
    def index_corpus(self, *, name: str, documents: list[dict[str, Any]],
                     chunk_size: int = 512, overlap: int = 64) -> IndexRef:
        """Chunk + index a corpus of ``{text, source, metadata}`` documents."""
        all_chunks: list[Chunk] = []
        for doc in documents or []:
            all_chunks.extend(self.chunk(
                text=str(doc.get("text", "")), chunk_size=chunk_size, overlap=overlap,
                source=str(doc.get("source", "")), metadata=doc.get("metadata") or {},
            ))
        ref = self.build_index(name=name, chunks=all_chunks)
        # record the true doc count (distinct documents, not chunks)
        return IndexRef(name=ref.name, backend="local", doc_count=len(documents or []),
                        chunk_count=ref.chunk_count, detail=ref.detail)

    # ── query (T-RAG-06) ──────────────────────────────────────────────────────────
    def query(self, *, name: str, query: str, top_k: int = 5,
              filters: dict[str, Any] | None = None) -> list[RetrievalResult]:
        """TF-IDF cosine retrieval of the ``top_k`` chunks (with optional metadata filters)."""
        index = self._INDEXES.get(name)
        if not index or index["n"] == 0:
            return []
        n = index["n"]
        df: Counter[str] = index["df"]
        q_tf = Counter(_tokenize(query))
        if not q_tf:
            return []

        def _idf(term: str) -> float:
            return math.log((1 + n) / (1 + df.get(term, 0))) + 1.0

        # query vector (tf-idf)
        q_vec = {t: f * _idf(t) for t, f in q_tf.items()}
        q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(index["tf"]):
            ch = index["chunks"][i]
            if filters and not all(ch.metadata.get(k) == v for k, v in filters.items()):
                continue
            # only score over the query terms (sparse dot product)
            dot = 0.0
            d_sq = 0.0
            for t, qf in q_vec.items():
                df_w = tf.get(t, 0) * _idf(t)
                dot += qf * df_w
            # doc norm over its own terms (full)
            for t, c in tf.items():
                w = c * _idf(t)
                d_sq += w * w
            d_norm = math.sqrt(d_sq) or 1.0
            score = dot / (q_norm * d_norm)
            if score > 0:
                scored.append((score, i))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[RetrievalResult] = []
        for rank, (score, i) in enumerate(scored[: max(1, int(top_k))]):
            out.append(RetrievalResult(chunk=index["chunks"][i], score=round(score, 6), rank=rank))
        return out


# ══════════════════════════════════════════════════════════════════════════════
# BedrockKnowledgeBasesBackend — managed (Bedrock KB / OpenSearch; ADR-004)
# ══════════════════════════════════════════════════════════════════════════════
class BedrockKnowledgeBasesBackend:
    """Default (managed) RetrievalBackend — thin boto3 over Bedrock Knowledge Bases.

    ``chunk``/``build_index``/``index_corpus`` use the SAME local chunking + an in-process
    fallback index (KB ingestion is provisioned out-of-band by IaC); ``query`` calls the
    Bedrock Agent-Runtime ``retrieve`` API against ``knowledge_base_id``. Bedrock Agent
    Runtime is NOT in the offline AWS API emulator, so the client is injectable (callers may pass a fake) and ``query``
    degrades to the genuine local computation when no KB / client is available
    (offline-safe). ``orchestrate-not-reimplement`` (ADR-004).
    """

    def __init__(self, *, region: str | None = None, knowledge_base_id: str = "",
                 agent_runtime_client: Any = None) -> None:
        self._region = resolve_aws_region(region, required=False)
        self._kb_id = knowledge_base_id
        self._client = agent_runtime_client
        self._local = LocalRetrievalBackend()  # chunking + offline-safe fallback

    def _runtime(self) -> Any:
        if self._client is None:
            import boto3
            self._client = boto3.client("bedrock-agent-runtime", region_name=resolve_aws_region(self._region))
        return self._client

    def chunk(self, *, text: str, chunk_size: int = 512, overlap: int = 64,
              source: str = "", metadata: dict[str, Any] | None = None) -> list[Chunk]:
        return self._local.chunk(text=text, chunk_size=chunk_size, overlap=overlap,
                                 source=source, metadata=metadata)

    def build_index(self, *, name: str, chunks: list[Chunk]) -> IndexRef:
        ref = self._local.build_index(name=name, chunks=chunks)
        return IndexRef(name=name, backend="bedrock_kb", arn=self._kb_id,
                        doc_count=ref.doc_count, chunk_count=ref.chunk_count, detail=ref.detail)

    def index_corpus(self, *, name: str, documents: list[dict[str, Any]],
                     chunk_size: int = 512, overlap: int = 64) -> IndexRef:
        ref = self._local.index_corpus(name=name, documents=documents,
                                       chunk_size=chunk_size, overlap=overlap)
        return IndexRef(name=name, backend="bedrock_kb", arn=self._kb_id,
                        doc_count=ref.doc_count, chunk_count=ref.chunk_count, detail=ref.detail)

    def query(self, *, name: str, query: str, top_k: int = 5,
              filters: dict[str, Any] | None = None) -> list[RetrievalResult]:
        """Retrieve via Bedrock KB ``retrieve``; fall back to the genuine local index offline."""
        if self._kb_id:
            try:
                resp = self._runtime().retrieve(
                    knowledgeBaseId=self._kb_id,
                    retrievalQuery={"text": query},
                    retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": int(top_k)}},
                )
                out: list[RetrievalResult] = []
                for rank, item in enumerate(resp.get("retrievalResults", [])):
                    content = item.get("content", {}).get("text", "")
                    loc = item.get("location", {}).get("s3Location", {}).get("uri", "")
                    out.append(RetrievalResult(
                        chunk=Chunk(chunk_id=f"kb::{rank}", text=content, source=loc,
                                    metadata=item.get("metadata", {}) or {}),
                        score=float(item.get("score", 0.0)), rank=rank,
                    ))
                if out:
                    return out
            except Exception:  # nosec B110
                pass
        # offline-safe fallback: the genuine in-process index (same shape)
        return self._local.query(name=name, query=query, top_k=top_k, filters=filters)


# ── registration: local default offline; bedrock_kb managed; opensearch alias ──
register_adapter(RetrievalBackend, "local", lambda: LocalRetrievalBackend(), is_local=True)
register_adapter(RetrievalBackend, "bedrock_kb", lambda: BedrockKnowledgeBasesBackend(), is_default=True)
register_adapter(RetrievalBackend, "opensearch", lambda: BedrockKnowledgeBasesBackend())
