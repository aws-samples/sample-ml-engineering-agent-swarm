"""Chat screen — conversational interface for the Agentic-ML TUI.

Provides a full chat interface with:
- Scrollable message history
- A TRUE multi-line input box at the bottom (``ChatInput``)
- File attachment support
- Slash commands for quick actions

Multi-line input (this module's :class:`ChatInput`)
───────────────────────────────────────────────────
The input box is a thin :class:`textual.widgets.TextArea` subclass so the user
can type or PASTE multi-line prompts (carriage returns preserved) and submit the
whole block as ONE turn. Submission semantics are chosen to keep the fast
single-line flow that the 100-story suite + e2e harness rely on (they submit via
``pilot.press("enter")``):

  • **Enter**                  → SUBMIT the entire buffer (newlines intact).
  • **Ctrl+J** / **Shift+Enter** → insert a literal newline WITHOUT submitting.

``Ctrl+J`` is the primary newline key because it is a real control character that
every terminal delivers identically (``\\n``); ``Shift+Enter`` also works where the
terminal reports a distinct key. Pasted text flows through ``TextArea._on_paste``
unchanged, so carriage returns in a paste are retained automatically.

``ChatInput`` also exposes a ``value`` read/write alias for ``TextArea.text`` so the
existing harness (``send_chat`` sets ``chat_input.value = text``) keeps working
byte-for-byte — only the widget *type* changed, not the submit contract.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Static, TabPane, TextArea

from ..chat.command_queue import CommandQueue, QueuedCommand
from ..chat.engine import ChatEngine
from ..chat.thought_trace import ThoughtTrace
from ..widgets.thought_ticker import ThoughtTicker

# Keys that insert a newline instead of submitting. ``ctrl+j`` is the canonical,
# always-available control char; ``newline`` is its Textual alias (keys.py);
# ``shift+enter`` is honored where the terminal distinguishes it.
_NEWLINE_KEYS = frozenset({"ctrl+j", "newline", "shift+enter"})


class ChatInput(TextArea):
    """Multi-line chat input: Enter submits, Ctrl+J / Shift+Enter add a newline.

    Subclasses :class:`TextArea` (true multi-line, paste-preserving) and overrides
    the key handler so Enter dispatches a :class:`ChatInput.Submitted` message
    (mirroring the old ``Input.Submitted`` contract) instead of inserting ``\\n``.
    """

    class Submitted(Message):
        """Posted when the user presses Enter — carries the full buffer text.

        Bubbles to the enclosing :class:`ChatPane`, whose
        ``on_chat_input_submitted`` handler forwards it to the engine. The
        Textual handler name is derived from the class path, so this resolves to
        ``on_chat_input_submitted``.
        """

        def __init__(self, chat_input: ChatInput, value: str) -> None:
            super().__init__()
            self.input = chat_input
            self.value = value

        @property
        def control(self) -> ChatInput:
            return self.input

    @property
    def value(self) -> str:
        """Alias for :attr:`TextArea.text` — keeps the old ``Input`` API/harness."""
        return self.text

    @value.setter
    def value(self, new_value: str) -> None:
        self.load_text(new_value or "")

    def clear(self) -> None:  # type: ignore[override]
        """Empty the buffer (parity with the old ``Input.clear()`` call site)."""
        self.load_text("")

    async def _on_key(self, event: events.Key) -> None:
        """Enter submits; Ctrl+J / Shift+Enter insert a newline; else default."""
        key = event.key
        if key == "enter":
            # Submit the whole buffer as one turn (do NOT insert a newline).
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self, self.text))
            return
        if key in _NEWLINE_KEYS:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        # All other keys (printable chars, cursor moves, paste, etc.) keep the
        # native TextArea behavior — including multi-line editing.
        await super()._on_key(event)


class ChatPane(TabPane):
    """Chat interface — conversational control for all platform features."""

    def __init__(self, title: str = "Chat", **kwargs: Any) -> None:
        super().__init__(title, **kwargs)
        self._engine = ChatEngine()
        #  (command queuing): commands submitted while a turn is already
        # in flight are QUEUED here and fired one-after-another instead of the old
        # ``exclusive=True`` worker CANCELLING the running turn. ``_turn_active`` is
        # the single-flight guard the queue drains against.
        self._command_queue = CommandQueue()
        self._turn_active = False

    def compose(self) -> ComposeResult:
        # : the redundant per-tab title (the top tab-nav already labels the
        # active tab) is removed to reclaim vertical space.
        with Vertical(classes="chat-content"):
            with VerticalScroll(id="chat-history", classes="chat-history"):
                yield Static(
                    "🤖 **Welcome to Agentic-ML Chat!**\n\n"
                    "I'm your ML engineering assistant. You can:\n"
                    "  • Chat naturally about ML tasks\n"
                    "  • Use `/commands` for direct actions\n"
                    "  • Attach files for analysis\n\n"
                    "**Runtime mode** — pick it from the selector at the top-left of the "
                    "header:\n"
                    "  • **Offline** — no AWS; local models + a regex chat brain\n"
                    "  • **Local** — pipeline runs locally, chat uses Bedrock\n"
                    "  • **Full AWS** — managed backends + Bedrock chat\n"
                    "Your `/run` pipeline always executes locally; the mode changes "
                    "chat + cloud tabs, not the training path.\n\n"
                    "Multi-line prompts: **Enter** sends · **Ctrl+J** (or **Shift+Enter**) "
                    "adds a new line.\n"
                    "Type `/help` to see all commands.",
                    classes="chat-message assistant-message",
                    id="welcome-msg",
                )
            # Command-queue panel: lists commands queued behind the
            # in-flight turn; each row carries an "⬆" send-immediately control that
            # promotes it to the FRONT (steer vs queue). Hidden while empty so the
            # idle chat is visually unchanged.
            yield Vertical(id="command-queue", classes="command-queue")
            with Horizontal(classes="chat-input-row"):
                yield ChatInput(
                    id="chat-input",
                    classes="chat-input-field",
                    placeholder=(
                        "Ask anything or use /commands…  "
                        "(Enter = send · Ctrl+J / Shift+Enter = new line)"
                    ),
                )
                yield Button("Send", id="send-btn", variant="primary", classes="chat-send-btn")
                yield Button("📎", id="attach-btn", variant="default", classes="chat-attach-btn")

    def on_mount(self) -> None:
        """Focus the input on mount; keep the (empty) queue panel hidden."""
        self.query_one("#chat-input", ChatInput).focus()
        self.query_one("#command-queue", Vertical).display = False

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses (send/attach + per-queued-item promote/remove)."""
        btn_id = event.button.id or ""
        if btn_id == "send-btn":
            self._send_message()
        elif btn_id == "attach-btn":
            self._prompt_attach()
        elif btn_id.startswith("queue-up-"):
            # "Send immediately" (up-arrow): promote this queued command to front.
            self._promote_queued(btn_id[len("queue-up-") :])
        elif btn_id.startswith("queue-del-"):
            self._remove_queued(btn_id[len("queue-del-") :])

    def on_chat_input_submitted(self, event: ChatInput.Submitted) -> None:
        """Handle Enter in the multi-line input (replaces ``on_input_submitted``)."""
        if event.input.id == "chat-input":
            self._send_message()

    def _send_message(self) -> None:
        """Send the current input as a message."""
        input_widget = self.query_one("#chat-input", ChatInput)
        message = input_widget.value.strip()
        if not message:
            return

        # Clear input IMMEDIATELY using clear() for proper reactivity.
        input_widget.clear()
        input_widget.focus()
        self._send_message_text(message)

    def _send_message_text(self, message: str) -> QueuedCommand | None:
        """Start the turn now, or QUEUE it if one is already running.

        Split out from :meth:`_send_message` so the input-clearing UI action is
        separable from the turn kickoff (and so callers can drive a full turn without
        typing into the widget).

        **Command queuing:** if a turn is already in flight
        (``_turn_active``), this command is appended to :attr:`_command_queue` and
        the queue panel is refreshed — it will fire when the current turn (and any
        earlier-queued commands) complete, instead of the pre-#5
        ``run_worker(exclusive=True)`` CANCELLING the running turn. Returns the
        created :class:`QueuedCommand` when queued, or ``None`` when it started
        immediately.
        """
        message = message.strip()
        if not message:
            return None

        # A turn is already running → queue this one (do NOT cancel the in-flight
        # turn). The idle single-submission path below is unchanged.
        if self._turn_active:
            item = self._command_queue.enqueue(message)
            self._render_queue()
            return item

        self._start_turn(message)
        return None

    def _start_turn(self, message: str) -> None:
        """Append the user bubble, mount the ephemeral ticker, and start the worker.

        The ephemeral :class:`ThoughtTicker` REPLACES the old static
        ``"🤖 ⏳ Thinking..."`` indicator during the turn (Feature 2, design §3.3):
        it is mounted here (id ``#thought-ticker``), fed by a per-turn
        :class:`ThoughtTrace`, and REMOVED before the persisted assistant bubble.

        Single-flight is now enforced by :attr:`_turn_active` (set here, cleared in
        :meth:`_process_and_respond`), so the worker no longer needs ``exclusive`` —
        the queue guarantees at most one turn runs at a time and pending commands
        drain in order.
        """
        self._turn_active = True

        # 1. Show user message in history right away.
        self._append_message("user", message)

        # 2. Mount the ephemeral rotating ticker (replaces the static indicator).
        history = self.query_one("#chat-history", VerticalScroll)
        ticker = ThoughtTicker(id="thought-ticker", classes="chat-message thought-ticker")
        history.mount(ticker)
        ticker.start()

        # 3. Scroll to bottom.
        history.scroll_end(animate=False)

        # 4. Build the per-turn ephemeral trace bound to the ticker, then process in
        #    the background (non-blocking). The trace is transient — NEVER persisted.
        trace = self._build_thought_trace()
        self.run_worker(self._process_and_respond(message, trace))

    def _pump_queue(self) -> None:
        """Fire the next queued command, if any and no turn is active."""
        if self._turn_active:
            return
        nxt = self._command_queue.dequeue()
        if nxt is None:
            return
        self._render_queue()
        self._start_turn(nxt.text)

    def _promote_queued(self, item_id: str) -> None:
        """Promote a queued command to the FRONT (up-arrow "send immediately")."""
        if self._command_queue.promote(item_id):
            self._render_queue()

    def _remove_queued(self, item_id: str) -> None:
        """Drop a queued command before it runs."""
        if self._command_queue.remove(item_id):
            self._render_queue()

    def _render_queue(self) -> None:
        """Re-render the command-queue panel from :attr:`_command_queue`.

        Hidden while empty. Each pending command gets a row with an "⬆" promote
        (send-immediately) button, an "✖" remove button, and the command text
        rendered with ``markup=False`` (queued text may contain ``[...]`` tokens —
        same markup-safety discipline as ).
        """
        try:
            container = self.query_one("#command-queue", Vertical)
        except Exception:
            return
        container.remove_children()
        pending = self._command_queue.pending
        if not pending:
            container.display = False
            return
        container.display = True
        rows: list[Any] = [
            Static(
                f"⏳ Queued ({len(pending)}) — ⬆ runs next",
                classes="queue-header",
                markup=False,
            )
        ]
        for item in pending:
            rows.append(
                Horizontal(
                    Button("⬆", id=f"queue-up-{item.id}", classes="queue-up-btn"),
                    Button("✖", id=f"queue-del-{item.id}", classes="queue-del-btn"),
                    Static(item.text, classes="queue-text", markup=False),
                    classes="queue-row",
                )
            )
        container.mount(*rows)

    def _build_thought_trace(self) -> ThoughtTrace:
        """Construct the per-turn ephemeral trace, sinking events into the ticker.

        The ticker's ``push`` is the trace's ``on_event`` sink (marshalled onto the
        UI thread by the ticker itself). If the ticker isn't mounted (defensive), a
        sink-less trace is returned so the turn still runs. This is a distinct method
        so the live ticker test can wrap it to observe the events.
        """
        try:
            ticker = self.query_one("#thought-ticker", ThoughtTicker)
            return ThoughtTrace(on_event=ticker.push)
        except Exception:
            return ThoughtTrace()

    async def _remove_thought_ticker(self) -> None:
        """Stop + REMOVE the ephemeral ticker (and any legacy indicator).

        Called BEFORE the persisted assistant bubble is appended, so no ticker
        content survives into ``#chat-history`` (the T-TT-06 ordering guarantee). We
        ``await`` the removal so the widget is truly gone before ``_append_message``.
        """
        for wid in ("#thought-ticker", "#thinking-indicator"):
            try:
                widget = self.query_one(wid)
            except Exception:  # nosec B112
                continue
            stop = getattr(widget, "stop_and_clear", None)
            if callable(stop):
                try:
                    stop()
                except Exception:  # nosec B110
                    pass
            try:
                await widget.remove()
            except Exception:  # nosec B110
                pass

    async def _process_and_respond(
        self, message: str, thought_trace: ThoughtTrace | None = None
    ) -> None:
        """Background worker — processes message and updates UI.

        The ephemeral thought ticker is fed live during ``process_message`` (via
        ``thought_trace``), then STOPPED + REMOVED before the persisted assistant
        bubble is appended — so its rotating content never reaches ``engine.history``
        or the ``#chat-history`` scrollback (Feature 2 non-persistence guarantee).
        """
        try:
            response = await self._engine.process_message(
                message, thought_trace=thought_trace
            )

            # Remove the ephemeral ticker BEFORE writing the persisted assistant bubble.
            await self._remove_thought_ticker()

            # Add response (the persisted assistant bubble).
            self._append_message("assistant", response.message)

            # If there are suggestions, show them.
            if response.suggestions:
                suggestion_text = "💡 " + " | ".join(f"`{s}`" for s in response.suggestions)
                self._append_message("system", suggestion_text)
        finally:
            # Turn finished (success or error) → drop the single-flight guard and
            # fire the next queued command, if any.
            self._turn_active = False
            self._pump_queue()

    def _append_message(self, role: str, content: str) -> None:
        """Append a message to the chat history display.

        RENDER-SAFETY (TUI-MARKUP-01): the reply ``content`` is DYNAMIC — tool
        output, LLM prose, URIs, dicts, error strings — none of which is authored
        as Rich console markup (styling here is ``**bold**``/emoji/backticks, i.e.
        literal text). Textual's ``Static`` defaults to ``markup=True``, so a
        stray closing-tag-shaped token in the content (e.g. ``[/:]`` from a lake
        URI / error message) makes Rich raise ``MarkupError`` and crashes the
        whole turn's render (observed via story MO-043 ``/lake write``). We render
        the mounted ``Static`` with ``markup=False`` so ALL content — including
        the ``[HH:MM]`` timestamp — is treated as literal text. No intended
        styling is lost because the chat layer never emits Rich ``[tag]`` markup.
        """
        prefixes = {"user": "👤", "assistant": "🤖", "system": "ℹ️"}
        prefix = prefixes.get(role, "•")
        css_class = f"chat-message {role}-message"

        timestamp = datetime.now(UTC).strftime("%H:%M")
        display_text = f"{prefix} [{timestamp}] {content}"

        history = self.query_one("#chat-history", VerticalScroll)
        new_msg = Static(display_text, classes=css_class, markup=False)
        history.mount(new_msg)

        # Scroll to bottom
        history.scroll_end(animate=False)

    def _prompt_attach(self) -> None:
        """Prompt user for file attachment path."""
        input_widget = self.query_one("#chat-input", ChatInput)
        input_widget.value = "/attach "
        input_widget.focus()
