"""Feature 2 — the ephemeral rotating ``ThoughtTicker`` widget.

A single-line "stock ticker" of the agent's live thoughts, mounted only WHILE a
turn is in flight (id ``#thought-ticker``) and REMOVED before the persisted
assistant bubble is written (master design
``docs/tui-agent-fs-remote-and-thought-trace-design.md`` §3.2c/§3.3). It replaces
the old static ``"🤖 ⏳ Thinking..."`` indicator during a turn.

Non-persistence: the ticker reads only from its own bounded buffer (fed by the
per-turn :class:`~amlas.tui.chat.thought_trace.ThoughtTrace`). It NEVER writes to
``ChatEngine.history``, constructs a ``ChatMessage``, or mounts anything into the
persisted ``#chat-history`` scrollback — so its content cannot leak into
scrollback or history (the T-TT-05 guarantee).

Thread-marshalling mirrors ``AgentActivityPanel._on_context`` /
``app.call_from_thread`` ([`agent_activity.py:63-81`](agent_activity.py:63)): the
callback handler fires ``push`` from the agent's worker thread, so the widget
mutation is marshalled onto the UI thread. Rendered with ``markup=False`` for the
same crash-safety reason as the chat bubbles ([`chat.py:205-224`](../screens/chat.py:205)).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from textual.widgets import Static

from ..chat.thought_trace import ThoughtEvent

logger = logging.getLogger(__name__)

__all__ = ["ThoughtTicker"]

# Rotation cadence (seconds) — a calm "ticker" pace, not a strobe.
_ROTATE_INTERVAL = 1.2
# How many recent thoughts to keep in the rotating window.
_WINDOW = 6
# The generic line shown before any tool/step events arrive (no-tool turns).
_THINKING = "thinking…"


class ThoughtTicker(Static):
    """Ephemeral, rotating single-line ticker of the agent's live thoughts.

    Lifecycle (driven by :class:`~amlas.tui.screens.chat.ChatPane`):
      * mounted at turn start (id ``#thought-ticker``);
      * fed via :meth:`push` (from the ``ThoughtTrace`` sink, marshalled);
      * :meth:`start` begins the rotation timer;
      * :meth:`stop_and_clear` cancels the timer + clears the text, then the pane
        removes the widget BEFORE appending the persisted assistant bubble.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # markup=False → dynamic thought text (tool names, URIs) is literal, never
        # parsed as Rich markup (same crash-safety reason as the chat bubbles).
        kwargs.setdefault("markup", False)
        super().__init__(*args, **kwargs)
        self._events: deque[ThoughtEvent] = deque(maxlen=_WINDOW)
        self._index = 0
        self._timer: Any | None = None

    # ─── lifecycle ────────────────────────────────────────────────────────────

    def on_mount(self) -> None:
        """Render the initial generic thinking line on mount."""
        try:
            self.update(self._render_line())
        except Exception:  # nosec B110
            pass

    def start(self) -> None:
        """Begin the rotation timer (idempotent). Rotates ~every 1.2s."""
        if self._timer is not None:
            return
        try:
            self._timer = self.set_interval(_ROTATE_INTERVAL, self.advance)
        except Exception:
            # Headless / not-yet-mounted: rotation is a cosmetic nicety; skip.
            self._timer = None

    def advance(self) -> None:
        """Advance the rotation index by one and repaint (safe to call directly).

        Exposed as a public method so callers can rotate deterministically without
        waiting on the wall-clock interval.
        """
        self._index += 1
        try:
            self.update(self._render_line())
        except Exception:  # nosec B110
            pass

    def push(self, event: ThoughtEvent) -> None:
        """Add a thought event and repaint — marshalled onto the UI thread.

        The agent's callback handler fires from a worker thread
        (``asyncio.to_thread(chat_with_tools, …)``), so a direct widget mutation
        would be a cross-thread violation. Mirroring
        ``AgentActivityPanel._on_context`` ([`agent_activity.py:63-81`](agent_activity.py:63)),
        we marshal the mutation via ``app.call_from_thread`` when an app is present;
        headless (no app) we apply directly. Never raises.
        """
        try:
            app = self.app
        except Exception:
            app = None

        if app is None:
            self._apply_push(event)
            return
        try:
            app.call_from_thread(self._apply_push, event)
        except Exception:
            # Already on the UI thread (call_from_thread rejects same-thread calls),
            # or the app is tearing down — apply directly as a fallback.
            try:
                self._apply_push(event)
            except Exception:  # nosec B110
                pass

    def _apply_push(self, event: ThoughtEvent) -> None:
        """UI-thread mutation: buffer the event and repaint the current line."""
        self._events.append(event)
        try:
            self.update(self._render_line())
        except Exception:  # nosec B110
            pass

    def stop_and_clear(self) -> None:
        """Cancel the rotation timer and clear the text (pre-removal reset).

        Called by the pane right before the ticker is removed and the persisted
        assistant bubble is appended, so no ticker content survives the turn.
        """
        if self._timer is not None:
            try:
                self._timer.stop()
            except Exception:  # nosec B110
                pass
            self._timer = None
        self._events.clear()
        try:
            self.update("")
        except Exception:  # nosec B110
            pass

    # ─── render ─────────────────────────────────────────────────────────────

    def _render_line(self) -> str:
        """Compose the rotating single-line ticker from the recent thoughts.

        With no events yet, shows the generic thinking line. With events, renders a
        compact ``▐ a ▸ b ▸ c ▸`` strip whose leading entry rotates over time so the
        operator sees movement (a "stock ticker" feel) rather than a static line.
        """
        events = list(self._events)
        if not events:
            return f"🤖 ▐ {_THINKING} ▸"

        phrases = [e.text for e in events][-_WINDOW:]
        # Rotate the starting point so the strip visibly scrolls on each tick.
        offset = self._index % len(phrases)
        rotated = phrases[offset:] + phrases[:offset]
        strip = " ▸ ".join(rotated)
        return f"🤖 ▐ {strip} ▸"
