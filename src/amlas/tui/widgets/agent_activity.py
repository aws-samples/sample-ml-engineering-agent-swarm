"""Phase 4 — ``AgentActivityPanel``: WHAT the agents are doing (§E.8 / GO-3/4/5).

This is the ``src/amlas/tui/widgets/agent_activity.py`` widget named in the design
(``docs/tui-context-integration-design.md`` §E.8 / §F Phase 4). It gives the operator
full transparency into the AgentCore swarm and the local chat agent — closing GO-3
(no agent-activity visibility), GO-4 (local chat-agent tool trace never aggregated),
and GO-5 (cloud dispatch run events not surfaced).

**Data source:** ``state.agents`` on the shared ``ContextBus`` — a dict of
``agent_name → `` :class:`~amlas.tui.context.models.AgentActivity`, populated by
``ContextBus.set_agent_activity`` (from the Dashboard's ``TraceTailReader`` poller
and/or the engine's chat-agent result seam). The panel SUBSCRIBES to the ``agents``
event and repaints a live table:

    ``Agent | Phase | Last tool | Model | AWS services | Status``

e.g. ``research_lead | R1_FRAMING | search_algorithms | Opus 4.8 | Bedrock | running``.

**Discipline:** the same ``call_from_thread`` marshalling as the Phase-3 panes — a
``set_agent_activity`` may fire from the ``@work(thread=True)`` trace poller, so the
repaint lands on the UI thread. Everything is additive and null-safe: with no bus
(headless / pre-mount) it shows an empty-table placeholder and never crashes.
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.widgets import Static

from ..context import AppContextState, ContextEvent

__all__ = ["AgentActivityPanel"]

# Statuses that qualify an agent for the "Roster (signed-on + running)" grouping
# (Phase 5 §2e / D-5). Module-level constant — an active-work status set.
_ROSTER_STATUSES = frozenset({"running", "submitted"})


class AgentActivityPanel(Static):
    """Live table of per-agent activity sourced from ``state.agents`` (§E.8)."""

    def compose(self) -> ComposeResult:
        yield Static(self._placeholder(), id="agent-activity-content")

    # ─── ContextBus wiring (design §E.8, mirrors the Phase-3 panes) ─────────────

    def on_mount(self) -> None:
        """Subscribe to the bus and prime the table once. Null-safe (no bus → no-op)."""
        bus = self._bus()
        if bus is None:
            return
        try:
            bus.subscribe(self._on_context)
            self._apply_agents(bus.state())
        except Exception:  # nosec B110
            pass

    def _bus(self) -> Any | None:
        """Return the App-owned ``ContextBus`` or ``None`` (headless / pre-mount)."""
        try:
            return getattr(self.app, "context_bus", None)
        except Exception:
            return None

    def _on_context(self, event: ContextEvent, state: AppContextState) -> None:
        """Bus subscriber — repaint on the ``agents`` event, marshalled to the UI thread.

        Only ``agents`` events matter to this panel. Never raises (a bad refresh must
        not break the mutation that triggered it).
        """
        if event != "agents":
            return
        try:
            app = self.app
        except Exception:
            return
        try:
            app.call_from_thread(self._apply_agents, state)
        except Exception:
            try:
                self._apply_agents(state)
            except Exception:  # nosec B110
                pass

    # ─── Render ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _placeholder() -> str:
        return "🤖 Agent Activity\n━━━━━━━━━━━━━━━━━━━━━━━━━━\n  No active agents"

    def _apply_agents(self, state: AppContextState) -> None:
        """Repaint the activity table from ``state.agents`` (guarded pre-mount)."""
        agents = state.agents or {}
        try:
            content = self.query_one("#agent-activity-content", Static)
        except Exception:
            return

        if not agents:
            try:
                content.update(self._placeholder())
            except Exception:  # nosec B110
                pass
            return

        # Phase 5 §2e (D-5): render TWO groupings — a "Roster" of signed-on agents that
        # are actively working (``signed_on and status in {running,submitted}``) and a
        # "Recent" section of ALL agents ordered by ``updated_at`` (last-seen freshness).
        # The columns are additive (``Last seen`` appended); every previously-rendered
        # substring (agent name/phase/tool/model/AWS/status) stays present so the locked
        # ``test_agent_activity_panel_renders_bus_record`` regression holds.
        def _row(act: Any) -> str:
            phase = act.phase or "—"
            last_tool = act.last_tool or "—"
            model = act.last_model or "—"
            services = ", ".join(act.aws_services) if act.aws_services else "—"
            status = act.status or "idle"
            last_seen = act.updated_at or "—"
            return (
                f"{act.agent_name} | {phase} | {last_tool} | {model} | "
                f"{services} | {status} | {last_seen}"
            )

        lines = ["🤖 Agent Activity", "━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        header = "Agent | Phase | Last tool | Model | AWS | Status | Last seen"

        roster = [
            agents[n]
            for n in sorted(agents)
            if getattr(agents[n], "signed_on", True) and (agents[n].status or "idle") in _ROSTER_STATUSES
        ]
        lines.append("Roster (signed-on + running):")
        lines.append(header)
        if roster:
            for act in roster:
                lines.append(_row(act))
        else:
            lines.append("  (no signed-on agents running)")

        # Recent: ALL agents by last-seen (most-recent ``updated_at`` first).
        lines.append("")
        lines.append("Recent (by last seen):")
        lines.append(header)
        for name in sorted(agents, key=lambda n: (agents[n].updated_at or ""), reverse=True):
            lines.append(_row(agents[name]))

        try:
            content.update("\n".join(lines))
        except Exception:  # nosec B110
            pass
