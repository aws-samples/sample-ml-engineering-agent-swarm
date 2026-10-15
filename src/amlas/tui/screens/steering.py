"""Steering pane — the domain-steering "dial" (master plan §10, §3.1).

Lets a lab tech build a ``SteeringProfile`` interactively: forbid/pin methods,
mechanism hints, max-LLM-fraction / novelty-budget, acceptance thresholds, and
free-text domain notes. The Save action constructs a real ``SteeringProfile`` and
stores it on the app as the **session profile** that the chat ``/impute``,
``/synth`` and ``/dqscan`` commands honor (HARD constraints enforced in the tools).
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Static, TextArea

from ...data_eng.steering import SteeringProfile


def _split_csv(value: str) -> list[str]:
    """Split a comma/space separated input into a clean list (drops empties)."""
    parts = [p.strip() for p in value.replace(",", " ").split()]
    return [p for p in parts if p]


class SteeringPane(Static):
    """Domain-steering "dial" screen — builds + saves a session SteeringProfile."""

    def compose(self) -> ComposeResult:
        # : redundant per-tab title removed (tab-nav already labels the tab).
        with Vertical(classes="steering-content"):
            with Horizontal(classes="steering-hard"):
                with Vertical(classes="steering-col"):
                    yield Static("Hard constraints", classes="section-title")
                    yield Input(
                        placeholder="forbid methods (e.g. mean, knn)", id="steering-forbid"
                    )
                    yield Input(
                        placeholder="pins col_or_dtype=method (e.g. datetime=time)",
                        id="steering-pin",
                    )
                    yield Input(
                        placeholder="max_llm_fraction (0..1, default 0.25)", id="steering-maxllm"
                    )
                    yield Input(
                        placeholder="novelty_budget (int, default 2)", id="steering-novelty"
                    )
                with Vertical(classes="steering-col"):
                    yield Static("Soft preferences + notes", classes="section-title")
                    yield Input(
                        placeholder="preferred methods (e.g. missforest, knn)",
                        id="steering-preferred",
                    )
                    yield TextArea(id="steering-domain-notes")
            yield Static("Domain notes feed the agent prompt; pins/forbids feed the tools.", classes="dim")
            with Horizontal(classes="steering-actions"):
                yield Button("💾 Save Steering Profile", id="steering-save", variant="primary")
                yield Button("🧹 Clear", id="steering-clear", variant="default")
            yield Static("", id="steering-status")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "steering-save":
            self._save_profile()
        elif event.button.id == "steering-clear":
            self._clear_profile()

    def _save_profile(self) -> None:
        """Build a SteeringProfile from the form and store it as the session dial."""
        forbid = _split_csv(self.query_one("#steering-forbid", Input).value)
        preferred = _split_csv(self.query_one("#steering-preferred", Input).value)
        notes = self.query_one("#steering-domain-notes", TextArea).text

        pin_methods: dict[str, str] = {}
        pin_raw = self.query_one("#steering-pin", Input).value
        for token in _split_csv(pin_raw):
            if "=" in token:
                k, v = token.split("=", 1)
                pin_methods[k.strip()] = v.strip()

        kwargs: dict[str, Any] = {
            "forbid_methods": forbid,
            "preferred_methods": preferred,
            "pin_methods": pin_methods,
            "domain_notes": notes,
        }
        maxllm = self.query_one("#steering-maxllm", Input).value.strip()
        if maxllm:
            try:
                kwargs["max_llm_fraction"] = float(maxllm)
            except ValueError:
                pass
        novelty = self.query_one("#steering-novelty", Input).value.strip()
        if novelty:
            try:
                kwargs["novelty_budget"] = int(novelty)
            except ValueError:
                pass

        profile = SteeringProfile(**kwargs)
        # Store on the app so the dial is discoverable as the session profile.
        self.app.session_steering = profile  # type: ignore[attr-defined]
        # WIRING FIX (06-tui-e2e-hardening, bug #1): also push the profile into the
        # CHAT engine's ``steering_profile`` — the attribute the chat ``/impute``,
        # ``/synth`` and ``/dqscan`` tools actually read
        # (``ChatEngine._steering_profile_dict`` → ``self.steering_profile``). The
        # screen previously wrote ONLY ``app.session_steering``, an attribute no
        # tool consulted, so a dial built on this screen silently never reached the
        # tools (manual doc §3 "Steering screen → chat tools wiring" gap). Bridging
        # both attributes makes the Steering screen a real, honored control path.
        self._apply_to_chat_engine(profile)
        status = self.query_one("#steering-status", Static)
        status.update(
            f"✓ Steering saved — forbid={profile.forbid_methods or '∅'}, "
            f"pins={profile.pin_methods or '∅'}, novelty={profile.novelty_budget}"
        )
        self.notify("Steering profile saved for this session", severity="information")

    def _chat_engine(self) -> object | None:
        """Return the live Chat-pane ``ChatEngine`` (or ``None`` if not mounted).

        Best-effort: the Steering screen and the Chat pane share one app, so we
        locate the ChatPane and return its private engine. Never raises — if the
        pane isn't present (e.g. a bare-pane offline validation) we simply skip the bridge.
        """
        try:
            from .chat import ChatPane

            return self.app.query_one(ChatPane)._engine
        except Exception:
            return None

    def _apply_to_chat_engine(self, profile: SteeringProfile | None) -> None:
        """Set/clear ``engine.steering_profile`` so chat tools honor this screen.

        Phase 3 (design §E.3): ALSO route the profile through the shared
        ``ContextBus`` (``set_steering``) so the dial has a single cross-tab home
        (unifying ``session_steering`` + ``engine.steering_profile``). Additive: the
        two legacy mirrors are kept so every existing caller stays green; the bus write
        is null-safe (no-op when there is no ``app.context_bus``).
        """
        engine = self._chat_engine()
        if engine is not None and hasattr(engine, "steering_profile"):
            engine.steering_profile = profile
        try:
            bus = getattr(self.app, "context_bus", None)
        except Exception:
            bus = None
        if bus is not None:
            try:
                bus.set_steering(profile)
            except Exception:  # nosec B110
                pass

    def _clear_profile(self) -> None:
        """Clear the session steering profile (back to the no-dial path)."""
        self.app.session_steering = None  # type: ignore[attr-defined]
        # Mirror the clear into the chat engine so a previously-saved dial is
        # actually removed from the tools too (the other half of the wiring fix).
        self._apply_to_chat_engine(None)
        for input_id in ("steering-forbid", "steering-pin", "steering-maxllm", "steering-novelty", "steering-preferred"):
            self.query_one(f"#{input_id}", Input).value = ""
        self.query_one("#steering-domain-notes", TextArea).text = ""
        self.query_one("#steering-status", Static).update("Steering cleared.")
