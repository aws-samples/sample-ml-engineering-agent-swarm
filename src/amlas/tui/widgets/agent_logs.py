"""``AgentLogsPane``: the dashboard's right-hand seven-agent logs pane.

The dashboard's right-hand pane is a dedicated
agent-focused surface for the seven agentcore/strands ML-automation leads — a
complete roster (each lead is always represented, even when idle) with a per-agent
SUMMARY line (phase · last tool · last model · status) plus a distinct
errors/warnings section calling out any agent that is ``failed`` or reporting a
warning status.

**Data source:** ``state.agents`` on the shared ``ContextBus`` — the same
:class:`~amlas.tui.context.models.AgentActivity` records the ``AgentActivityPanel``
renders, and populated by the Dashboard's ``TraceTailReader`` poller + the engine's
chat-agent result seam. The canonical roster is
:data:`amlas.control_plane.token_cost.ML_AUTOMATION_AGENTS` (the seven leads), so the
pane can NEVER drift from the real agent identities.

**Discipline (mirrors** :class:`~amlas.tui.widgets.agent_activity.AgentActivityPanel`
**):** a ``set_agent_activity`` may fire from the ``@work(thread=True)`` trace poller,
so the repaint marshals via ``app.call_from_thread`` with a direct-call fallback;
every ``query_one`` is guarded so a headless / pre-mount subscriber never crashes.
Content is ``markup=False`` and escaped for render safety.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.widgets import Static

from ...control_plane.token_cost import ML_AUTOMATION_AGENTS
from ..context import AppContextState, ContextEvent
from ..markup import escape_markup

__all__ = ["AgentLogsPane"]

# Statuses that mark an agent's log line as an error / warning (surfaced distinctly).
_ERROR_STATUSES = frozenset({"failed", "error"})
_WARNING_STATUSES = frozenset({"warning", "warn", "degraded", "stalled"})


class AgentLogsPane(Static):
    """Right-hand pane: the 7 ML-automation leads' logs — summaries + errors/warnings."""

    def compose(self) -> ComposeResult:
        # markup=False keeps dynamic agent and tool tokens literal.
        yield Static(self._render_text(), id="agent-logs-content", markup=False)

    # ─── ContextBus wiring (mirrors AgentActivityPanel) ─────────────────────────

    def on_mount(self) -> None:
        """Subscribe to the bus and prime the roster once. Null-safe (no bus → no-op)."""
        bus = self._bus()
        if bus is None:
            return
        try:
            bus.subscribe(self._on_context)
            self._apply_agents(bus.state())
        except Exception:  # nosec B110
            pass

    def _bus(self) -> Any | None:
        try:
            return getattr(self.app, "context_bus", None)
        except Exception:
            return None

    def _on_context(self, event: ContextEvent, state: AppContextState) -> None:
        """Repaint on the ``agents`` event, marshalled to the UI thread. Never raises."""
        if event != "agents":
            return
        try:
            app = self.app
        except Exception:
            app = None
        if app is None:
            self._apply_agents(state)
            return
        try:
            app.call_from_thread(self._apply_agents, state)
        except Exception:
            try:
                self._apply_agents(state)
            except Exception:  # nosec B110
                pass

    # ─── Render ─────────────────────────────────────────────────────────────────

    def _apply_agents(self, state: AppContextState) -> None:
        """Repaint the roster + logs from ``state.agents`` (guarded pre-mount)."""
        self._state_agents = dict(state.agents or {})
        try:
            content = self.query_one("#agent-logs-content", Static)
        except Exception:
            return
        try:
            content.update(self._render_text())
        except Exception:  # nosec B110
            pass

    def _render_text(self) -> str:
        """Compose the pane body: roster summaries + an errors/warnings section.

        A pure string (directly verifiable headless). Always lists ALL seven canonical
        leads so the pane is a complete roster, folding any live ``AgentActivity``
        record for a lead into its summary line.
        """
        agents: dict[str, Any] = getattr(self, "_state_agents", {}) or {}
        header = ["🕵️ Agent Logs", "━━━━━━━━━━━━━━━━━━━━━━━━━━"]

        lines: list[str] = list(header)
        lines.append("Agents (summary): Agent | Phase | Last tool | Model | Status")

        errors: list[str] = []
        warnings: list[str] = []

        for name in ML_AUTOMATION_AGENTS:
            act = agents.get(name)
            if act is None:
                lines.append(f"  {name} | — | — | — | idle")
                continue
            phase = escape_markup(str(getattr(act, "phase", None) or "—"))
            last_tool = escape_markup(str(getattr(act, "last_tool", None) or "—"))
            model = escape_markup(str(getattr(act, "last_model", None) or "—"))
            status = str(getattr(act, "status", None) or "idle")
            status_disp = escape_markup(status)
            lines.append(
                f"  {name} | {phase} | {last_tool} | {model} | {status_disp}"
            )
            low = status.lower()
            if low in _ERROR_STATUSES:
                errors.append(f"  ⚠️ {name}: {status_disp} (last tool: {last_tool})")
            elif low in _WARNING_STATUSES:
                warnings.append(f"  ⚠️ {name}: {status_disp} (last tool: {last_tool})")

        # Also fold any NON-lead agent that reported activity (e.g. tui_chat_agent) so
        # nothing is dropped, but only in the summary tail (the roster is the 7 leads).
        extra = [n for n in sorted(agents) if n not in ML_AUTOMATION_AGENTS]
        if extra:
            lines.append("")
            lines.append("Other agents:")
            for name in extra:
                act = agents[name]
                status = str(getattr(act, "status", None) or "idle")
                last_tool = escape_markup(str(getattr(act, "last_tool", None) or "—"))
                lines.append(f"  {escape_markup(name)} | {escape_markup(status)} | {last_tool}")
                low = status.lower()
                if low in _ERROR_STATUSES:
                    errors.append(f"  ⚠️ {escape_markup(name)}: {escape_markup(status)}")
                elif low in _WARNING_STATUSES:
                    warnings.append(f"  ⚠️ {escape_markup(name)}: {escape_markup(status)}")

        lines.append("")
        lines.append("Errors / Warnings:")
        if errors or warnings:
            lines.extend(errors)
            lines.extend(warnings)
        else:
            lines.append("  ✅ none")

        return "\n".join(lines)
