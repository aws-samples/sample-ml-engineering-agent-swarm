"""Feature 2 — the ephemeral ``ThoughtTrace`` channel (Textual-free).

A per-turn, bounded, thread-safe ring buffer of transient :class:`ThoughtEvent`s
that feeds the rotating :class:`~amlas.tui.widgets.thought_ticker.ThoughtTicker`
while the agent works. It is the STRUCTURAL non-persistence guarantee of the
"stock ticker" thought trace (master design
``docs/tui-agent-fs-remote-and-thought-trace-design.md`` §3.2a): it is created
per-turn, holds only recent thoughts, and is discarded at turn end.

Crucially this is **NOT** the ``ContextBus``. The bus carries *durable*,
Dashboard-facing session state (``AgentActivity`` folded post-turn — the OPPOSITE
of ephemeral, see design §3.4). Nothing here ever appends to ``ChatEngine.history``,
constructs a ``ChatMessage``, or mounts into the persisted ``#chat-history``
scrollback — so "not persisted / not clogging scrollback" is guaranteed by
construction, not by convention.

Marshalling / raising-sink discipline mirrors ``ContextBus._emit``
([`bus.py:341-355`](../context/bus.py:341)) and ``JobManager._emit``: a sink that
raises is logged and swallowed so one bad subscriber never breaks the turn.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["ThoughtEvent", "ThoughtTrace"]

# Recognised event kinds (a plain string, kept open for forward-compat):
#   "reasoning" | "tool_start" | "tool_end" | "model" | "status"


@dataclass
class ThoughtEvent:
    """One transient thought/step the agent surfaces during a turn.

    ``text`` is a short, human phrase (e.g. ``"running describe_dataset"`` or
    ``"reading s3://bucket/train.csv"``). ``tool`` names the tool when the event is
    tool-scoped. ``ts`` is a monotonic timestamp for ordering/throttling.
    """

    kind: str
    text: str
    tool: str | None = None
    ts: float = field(default_factory=time.monotonic)


class ThoughtTrace:
    """A bounded, thread-safe ring buffer of transient :class:`ThoughtEvent`s.

    One instance lives for exactly ONE agent turn and is then discarded (or
    :meth:`clear`-ed). It publishes each event to an optional ``on_event`` sink
    (the ticker widget's ``push``) using the raising-sink-safe discipline of
    ``ContextBus._emit``. It is Textual-free and importable without any UI deps.
    """

    def __init__(
        self,
        on_event: Callable[[ThoughtEvent], None] | None = None,
        maxlen: int = 50,
    ) -> None:
        self._on_event = on_event
        self._events: deque[ThoughtEvent] = deque(maxlen=max(1, maxlen))
        # A re-entrant lock: ``emit`` appends under the lock, but the sink itself
        # (the ticker's ``push``) must not be able to deadlock if it ever reads back.
        self._lock = threading.RLock()

    def emit(self, event: ThoughtEvent) -> None:
        """Append ``event`` to the ring, then notify the sink. NEVER raises.

        The append happens under the lock; the sink is invoked OUTSIDE the lock so a
        slow/marshalling sink can't hold the buffer. A raising sink is logged and
        swallowed (mirrors [`bus.py:341-355`](../context/bus.py:341)) so a bad
        subscriber never breaks the agent turn.
        """
        with self._lock:
            self._events.append(event)
        sink = self._on_event
        if sink is None:
            return
        try:
            sink(event)
        except Exception as exc:  # never let a sink break the turn
            logger.warning("ThoughtTrace sink raised on %r: %s", event.kind, type(exc).__name__)

    def recent(self, n: int = 3) -> list[ThoughtEvent]:
        """Return up to the last ``n`` events, oldest→newest (a copy)."""
        if n <= 0:
            return []
        with self._lock:
            events = list(self._events)
        return events[-n:]

    def clear(self) -> None:
        """Drop all buffered events (turn-end reset)."""
        with self._lock:
            self._events.clear()


# ─── Trace-source callback (Strands streaming) + coarse fallback ──────────────
#
# ``_make_trace_callback`` builds the ``callback_handler`` the tool backend passes
# into the Strands ``Agent``. Strands invokes the handler with streaming kwargs
# during a turn; the shapes vary across versions, so we map DEFENSIVELY and never
# raise (a broken callback must not break the turn). When the streaming shape is
# unavailable, the coarse ``execute_tool_use`` fallback in the backend emits
# tool_start/tool_end directly — guaranteeing tool-level ticker content offline
# (design §3.2b, Risk "Strands callback shape may differ").


def _clip(text: str, limit: int = 80) -> str:
    """Clip a reasoning/text delta to a short single-line ticker phrase."""
    flat = " ".join(str(text).split())
    return flat[:limit]


def _make_trace_callback(trace: ThoughtTrace) -> Callable[..., None]:
    """Return a Strands ``callback_handler`` that maps streaming events → the trace.

    Mapped shapes (all optional / best-effort, mirroring the Strands streaming
    callback contract; unknown shapes are ignored):

      * ``current_tool_use={"name": ...}``  → ``ThoughtEvent("tool_start", ...)``
      * ``message`` with a ``toolResult`` block → ``ThoughtEvent("tool_end", ...)``
      * ``data=<text delta>`` (reasoning/model text) → throttled
        ``ThoughtEvent("reasoning", <clipped>)``

    The handler is crash-safe: any unexpected payload is swallowed so a callback
    invocation never breaks the agent's event loop.
    """
    # Track the last tool we announced so we don't spam identical tool_start events
    # for every streamed delta of the same tool_use, and so we can pair an end.
    last_tool: dict[str, str | None] = {"name": None}

    def _callback(**kwargs: Any) -> None:
        try:
            # 1. Tool-use start (the most useful signal for the ticker).
            ctu = kwargs.get("current_tool_use")
            if isinstance(ctu, dict):
                name = ctu.get("name") or None
                if name and name != last_tool["name"]:
                    last_tool["name"] = name
                    trace.emit(
                        ThoughtEvent(kind="tool_start", text=f"running {name}", tool=name)
                    )
                return

            # 2. Tool result → tool_end (final-message content block form).
            message = kwargs.get("message")
            content = message.get("content", []) if isinstance(message, dict) else []
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "toolResult" in block:
                        name = last_tool["name"]
                        trace.emit(
                            ThoughtEvent(
                                kind="tool_end",
                                text=f"{name or 'tool'} done",
                                tool=name,
                            )
                        )
                        return

            # 3. Reasoning / model text delta → a throttled reasoning event.
            data = kwargs.get("data")
            if isinstance(data, str) and data.strip():
                trace.emit(ThoughtEvent(kind="reasoning", text=_clip(data)))
        except Exception as exc:  # a callback must never break the event loop
            logger.debug("trace callback swallowed error: %s", type(exc).__name__)

    return _callback
