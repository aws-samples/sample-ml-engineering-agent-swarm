"""Phase C2 — ``SessionLedger``: the session-wide append-only context store/cache.

This is the ``src/amlas/tui/context/session_ledger.py`` module named in the design
([`docs/tasks/phase-C-agent-context-design.md` §C2/§D](../../../../docs/tasks/phase-C-agent-context-design.md)).
It LAYERS ON TOP of the latest-value :class:`~amlas.tui.context.bus.ContextBus` — it
does NOT overload it (§D.1):

* the **ContextBus** is a *latest-value snapshot* ("what is CURRENT"): a second ``/run``
  overwrites the first's ``run``; its 38 schema tests + "exactly one event / immutable
  snapshot" invariants are LOCKED;
* the **SessionLedger** is an append-only *timeline* ("what has HAPPENED this session"):
  user turns, tool results, slash commands, run/dataset/spec bus transitions, artifacts
  and mode switches — the full session context the TUI agent packages into its own
  AgentCore/Strands request.

Design constraints honored here:

* **Textual-free / dataclasses only** (mirrors :mod:`.models`) — directly verifiable headless
  like a bare ``ChatEngine()``.
* **``RLock``-guarded** (mirrors :class:`~amlas.tui.context.bus.ContextBus`) — the UI
  thread appends on turn boundaries while the bus subscriber may fire on a job thread.
* **Accumulates in ALL runtime modes INCLUDING Offline** — it is fed from the
  DETERMINISTIC paths (a bus subscription + explicit ``record_*`` calls from the engine's
  fold seams), never from Bedrock, so the crux offline requirement is met by construction.
* **Bounded** (§D.3): ``MAX_ENTRIES = 500`` (FIFO prune), ``SUMMARY_MAX = 200``-char
  summaries, consecutive ``(kind, summary)`` dedup, and a bounded ``render_context`` /
  ``snapshot_dict`` slice for the prompt / tool.
* **Persistence** (§D.4): an atomic JSON cache at ``output/sessions/<session_id>/ledger.json``
  (temp + ``os.replace``), best-effort (never breaks a chat turn), turn-boundary debounced;
  load-on-launch is gated behind ``AMLAS_TUI_RESUME_SESSION`` (default OFF — the App owns
  that gate, this module just provides :meth:`SessionLedger.load`).

The **mode-switch handoff** needs NO extra machinery: because the ledger is
mode-independent, the existing ``set_runtime_mode`` gate flip means the next
``_build_context`` already folds the entire offline ledger into the first Bedrock prompt
(§D.6). The App only adds ONE ``record("mode", ...)`` note line.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

logger = logging.getLogger(__name__)

# The kinds of timeline events the ledger records (design §D.2).
LedgerKind = Literal[
    "turn", "tool", "command", "run", "dataset", "spec", "artifact", "mode"
]

__all__ = [
    "LedgerEntry",
    "SessionLedger",
    "SessionState",
]


@dataclass
class LedgerEntry:
    """One immutable event in the session timeline (all fields JSON-serializable)."""

    kind: LedgerKind
    ts: str                              # datetime.now(UTC).isoformat()
    summary: str = ""                    # bounded one-line human summary (≤ SUMMARY_MAX)
    detail: dict[str, Any] = field(default_factory=dict)  # bounded structured payload
    source: str = ""                     # "chat" | "specs_tab" | "agent" | tool | "mode_switch"


@dataclass
class SessionState:
    """The append-only timeline record (session id + started_at + ordered entries)."""

    session_id: str = ""                 # e.g. "session-YYYYMMDD-HHMMSS"
    started_at: str = ""
    entries: list[LedgerEntry] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Render the whole timeline as nested plain data (JSON-serializable)."""
        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "entries": [asdict(e) for e in self.entries],
        }


class SessionLedger:
    """Append-only session timeline that accumulates in ALL runtime modes (design §D.2).

    Fed by (a) a ContextBus subscription (:meth:`on_bus_event` — state transitions) and
    (b) explicit ``record_*`` calls from the engine's fold seams. Bounded + dedup'd +
    optionally persisted to ``output/sessions/<session_id>/ledger.json``.
    """

    MAX_ENTRIES: int = 500               # hard cap; oldest pruned first (§D.3, FIFO)
    SUMMARY_MAX: int = 200               # matches _summarize's 200-char cap (§D.3)
    # The bus events worth remembering as a timeline entry (a latest-value bus write of
    # spec/dataset/run is a real session milestone; job/agents/processes/selection are
    # high-frequency UI churn we do NOT spam the timeline with).
    _RECORDED_BUS_EVENTS: frozenset[str] = frozenset({"run", "dataset", "spec"})

    def __init__(
        self, session_id: str | None = None, *, persist_dir: str | None = None
    ) -> None:
        self._lock = threading.RLock()
        self._session_id = session_id or (
            "session-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        )
        self._started_at = datetime.now(UTC).isoformat()
        self._entries: list[LedgerEntry] = []
        # When set, ``persist`` writes ``<persist_dir>/<session_id>/ledger.json`` and the
        # turn-boundary ``record_turn``/``record_command`` auto-persist (debounced I/O).
        # When ``None`` the ledger is IN-MEMORY only and never touches disk.
        self._persist_dir = persist_dir

    # ─── identity / reads ──────────────────────────────────────────────────────

    @property
    def session_id(self) -> str:
        return self._session_id

    def entries(self) -> list[LedgerEntry]:
        """Return a shallow copy of the ordered entries (immutable read)."""
        with self._lock:
            return list(self._entries)

    def state(self) -> SessionState:
        """Return a deep copy of the whole timeline (immutable read)."""
        with self._lock:
            return SessionState(
                session_id=self._session_id,
                started_at=self._started_at,
                entries=[
                    LedgerEntry(
                        kind=e.kind,
                        ts=e.ts,
                        summary=e.summary,
                        detail=dict(e.detail),
                        source=e.source,
                    )
                    for e in self._entries
                ],
            )

    def to_dict(self) -> dict[str, Any]:
        """Render the whole ledger as the persisted JSON shape (§D.2)."""
        return self.state().to_dict()

    # ─── append (bounded + dedup'd) ─────────────────────────────────────────────

    def record(
        self,
        kind: LedgerKind,
        *,
        summary: str = "",
        detail: dict[str, Any] | None = None,
        source: str = "",
    ) -> None:
        """Append one bounded entry (dedup'd against the immediately-previous entry).

        Bounds (§D.3): the ``summary`` is capped at :attr:`SUMMARY_MAX`; the ``detail``
        payload keeps only scalar fields (nested list/dict dropped, same rule as
        ``_summarize``); consecutive identical ``(kind, summary)`` collapse to one; and
        the in-memory list is FIFO-pruned to :attr:`MAX_ENTRIES`. Never raises.
        """
        summ = (summary or "")[: self.SUMMARY_MAX]
        payload = self._bound_detail(detail)
        with self._lock:
            # Consecutive-dedup: a repeated identical (kind, summary) is a no-op.
            if self._entries:
                last = self._entries[-1]
                if last.kind == kind and last.summary == summ:
                    return
            self._entries.append(
                LedgerEntry(
                    kind=kind,
                    ts=datetime.now(UTC).isoformat(),
                    summary=summ,
                    detail=payload,
                    source=source,
                )
            )
            # FIFO prune: keep only the most-recent MAX_ENTRIES.
            if len(self._entries) > self.MAX_ENTRIES:
                del self._entries[: len(self._entries) - self.MAX_ENTRIES]

    @classmethod
    def _bound_detail(cls, detail: dict[str, Any] | None) -> dict[str, Any]:
        """Keep only scalar detail fields (drop nested list/dict — the ``_summarize`` rule)."""
        if not detail:
            return {}
        out: dict[str, Any] = {}
        for k, v in detail.items():
            if isinstance(v, (list, dict)):
                continue
            out[k] = v
        return out

    # ─── explicit record_* seams (called by the engine, all modes) ──────────────

    def record_turn(
        self, user_text: str, assistant_text: str, *, tools: list[str] | None = None
    ) -> None:
        """Record one completed chat turn (user prompt → assistant reply). Turn boundary.

        Fires from the engine's ``process_message`` return path in EVERY mode — the
        offline local-fallback brain still produces an assistant message, so the ledger
        fills even with ``bedrock_enabled=False`` (§D.6). Auto-persists (debounced).
        """
        user_snip = (user_text or "").strip().replace("\n", " ")
        assist_snip = (assistant_text or "").strip().replace("\n", " ")
        tool_note = f" ({len(tools)} tools)" if tools else ""
        summary = f"user: {user_snip[:90]} → assistant: {assist_snip[:80]}{tool_note}"
        detail: dict[str, Any] = {}
        if tools:
            detail["tools"] = ",".join(tools)[: self.SUMMARY_MAX]
        self.record("turn", summary=summary, detail=detail, source="chat")
        self._maybe_persist()

    def record_tool(
        self, tool_name: str, result_summary: str, *, source: str = "agent"
    ) -> None:
        """Record one executed tool result (reuses the engine's ≤200-char tool summary)."""
        summ = (result_summary or "").strip()
        summary = f"{tool_name}: {summ}" if summ else str(tool_name)
        self.record("tool", summary=summary, source=source)

    def record_command(self, command: str, result_summary: str) -> None:
        """Record one slash-command outcome (runs offline in every mode). Turn boundary."""
        cmd = str(command or "").lstrip("/")
        summ = (result_summary or "").strip().replace("\n", " ")
        summary = f"/{cmd} → {summ[:160]}" if summ else f"/{cmd}"
        self.record("command", summary=summary, source="chat")
        self._maybe_persist()

    # ─── ContextBus subscriber sink (state transitions) ─────────────────────────

    def on_bus_event(self, event: str, state: Any) -> None:
        """ContextBus subscriber ``(event, snapshot)`` → append a bounded run/dataset/spec entry.

        Only the milestone events (:attr:`_RECORDED_BUS_EVENTS`) become timeline entries;
        high-frequency UI churn (job/agents/processes/selection) is ignored so the ledger
        stays a meaningful history. Swallows all exceptions (the bus ``_emit`` already
        isolates a raising sink, but we double-guard so a bad snapshot never propagates).
        """
        try:
            if event not in self._RECORDED_BUS_EVENTS:
                return
            summary, detail = self._summarize_bus(event, state)
            if summary:
                self.record(event, summary=summary, detail=detail, source="bus")  # type: ignore[arg-type]
        except Exception:  # pragma: no cover — defensive (a raising sink is isolated)
            logger.debug("SessionLedger.on_bus_event failed for %r", event)  # no exc_info: payload may hold user data

    @staticmethod
    def _summarize_bus(event: str, state: Any) -> tuple[str, dict[str, Any]]:
        """Build a bounded one-line summary + scalar detail for a bus transition."""
        if event == "run":
            run = getattr(state, "run", None)
            run_id = getattr(run, "run_id", None)
            if not run_id:
                return "", {}
            best = getattr(run, "best_algorithm", None)
            score = getattr(run, "best_score", None)
            status = getattr(run, "status", None)
            parts = [f"run {run_id}"]
            if best:
                parts.append(f"best={best}")
            if score is not None:
                parts.append(f"score={score}")
            if status:
                parts.append(f"status={status}")
            return " ".join(parts), {"run_id": run_id, "best_algorithm": best, "status": status}
        if event == "dataset":
            ds = getattr(state, "dataset", None)
            path = getattr(ds, "path", None)
            if not path:
                return "", {}
            rows = getattr(ds, "rows", None)
            cols = getattr(ds, "cols", None)
            shape = f" ({rows}×{cols})" if rows is not None and cols is not None else ""
            return f"dataset {path}{shape}", {"path": path, "rows": rows, "cols": cols}
        if event == "spec":
            spec = getattr(state, "spec", None)
            ref = getattr(spec, "path", None) or getattr(spec, "project_id", None)
            if not ref:
                return "", {}
            task = getattr(spec, "task_type", None)
            target = getattr(spec, "target", None)
            extra = []
            if task:
                extra.append(f"task={task}")
            if target:
                extra.append(f"target={target}")
            suffix = f" ({', '.join(extra)})" if extra else ""
            return f"spec {ref}{suffix}", {"path": ref, "task_type": task, "target": target}
        return "", {}

    # ─── render for the prompt fold + the read tool (bounded) ───────────────────

    def render_context(self, *, max_entries: int = 12) -> str:
        """Render the most-recent ``max_entries`` as the ``_build_context`` section body.

        Bounded (§D.3/R3): only the recent slice enters the prompt. Returns ``""`` when
        the ledger is empty so ``_render_session_ledger`` can skip the section entirely.
        """
        with self._lock:
            recent = self._entries[-max_entries:] if max_entries > 0 else list(self._entries)
        if not recent:
            return ""
        lines = [f"- [{e.kind}] {e.summary}" for e in recent if e.summary]
        return "\n".join(lines)

    def snapshot_dict(self, *, max_entries: int = 20) -> dict[str, Any]:
        """Return a bounded dict for the ``get_tui_context`` ``"session"`` key (§D.5)."""
        with self._lock:
            recent = self._entries[-max_entries:] if max_entries > 0 else list(self._entries)
            total = len(self._entries)
            session_id = self._session_id
            started_at = self._started_at
        return {
            "session_id": session_id,
            "started_at": started_at,
            "count": len(recent),
            "total": total,
            "entries": [
                {"kind": e.kind, "ts": e.ts, "summary": e.summary, "source": e.source}
                for e in recent
            ],
        }

    # ─── persistence (atomic JSON, best-effort, turn-boundary debounced) ────────

    def _ledger_path(self) -> str | None:
        """Return the on-disk ledger.json path, or ``None`` when persistence is off."""
        if not self._persist_dir:
            return None
        return os.path.join(self._persist_dir, self._session_id, "ledger.json")

    def _maybe_persist(self) -> None:
        """Persist on a turn boundary when a persist dir is configured (best-effort)."""
        if self._persist_dir:
            self.persist()

    def persist(self) -> None:
        """Atomically write the ledger to ``output/sessions/<session_id>/ledger.json`` (§D.4).

        Atomic write = temp file + ``os.replace`` (the repo's mandated pattern), so a
        reader never sees a partial file. Best-effort: any failure is logged at debug and
        SWALLOWED — a persist failure must NEVER break a chat turn (mirrors ``_sync_bus``).
        """
        final = self._ledger_path()
        if final is None:
            return
        try:
            import json

            data = self.to_dict()
            directory = os.path.dirname(final)
            os.makedirs(directory, exist_ok=True)
            tmp = final + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            os.replace(tmp, final)
        except Exception:
            logger.debug("SessionLedger.persist failed (best-effort)", exc_info=True)

    @classmethod
    def load(
        cls, session_id: str, *, persist_dir: str | None = None
    ) -> SessionLedger:
        """Load a persisted ledger (resume path, §D.4 — gated by the App, default OFF).

        Reconstructs the entries from ``<persist_dir>/<session_id>/ledger.json``. Returns
        a FRESH ledger (empty) on any read/parse failure — never raises, so a corrupt
        cache degrades to a clean session rather than blocking launch.
        """
        led = cls(session_id=session_id, persist_dir=persist_dir)
        path = led._ledger_path()
        if not path or not os.path.exists(path):
            return led
        try:
            import json

            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            with led._lock:
                led._started_at = str(data.get("started_at") or led._started_at)
                led._entries = [
                    LedgerEntry(
                        kind=e.get("kind", "turn"),
                        ts=e.get("ts", ""),
                        summary=e.get("summary", ""),
                        detail=dict(e.get("detail") or {}),
                        source=e.get("source", ""),
                    )
                    for e in (data.get("entries") or [])
                    if isinstance(e, dict)
                ]
                # Respect the in-memory bound even on load.
                if len(led._entries) > cls.MAX_ENTRIES:
                    del led._entries[: len(led._entries) - cls.MAX_ENTRIES]
        except Exception:
            logger.debug("SessionLedger.load failed for %r (fresh)", session_id)  # no exc_info: ledger holds user turns
        return led
