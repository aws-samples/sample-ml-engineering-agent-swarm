"""TUI capability surface — the TUI's thin view over the M7 funnel (ADR-010; roadmap M7).

A small, framework-light surface object the TUI (a pane / screen) renders from. It does NOT
rewrite the Textual app — it is a pure-Python adapter that reads the shared flow registry and
invokes the SAME one-capability funnel (``control_plane.capability.invoke_capability``) the
Chat / CLI / Python-API use, so the TUI exposes capabilities identically (ADR-010) with no
flow logic duplicated (ADR-013: the surface is a renderer).

Kept import-cheap + Textual-free so it is directly verifiable WITHOUT launching the TUI (the Textual
app hangs under the macOS-aarch64 joblib deadlock); a Textual pane simply calls
:meth:`CapabilitySurface.rows` to populate a table and :meth:`CapabilitySurface.invoke` /
:meth:`CapabilitySurface.preview` on a key-press.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["CapabilityRow", "CapabilitySurface"]


@dataclass(frozen=True)
class CapabilityRow:
    """One render-ready capability row for a TUI table/pane (registry-derived)."""

    capability_id: str
    description: str
    milestone: str
    gated: bool
    runnable: bool

    @property
    def gate_label(self) -> str:
        """Short gate badge for the table cell."""
        return "🔒 gated" if self.gated else "📖 read"

    @property
    def run_label(self) -> str:
        """Short runnable badge for the table cell."""
        return "ok" if self.runnable else "—"


class CapabilitySurface:
    """The TUI's thin capability adapter (ADR-010 — one capability, surfaces are renderers).

    Wraps the shared capability funnel so a Textual pane can: list registered capabilities
    (:meth:`rows`), preview one in Plan Mode (:meth:`preview`), and invoke one
    (:meth:`invoke`) — all delegating to ``control_plane.capability`` with ``surface="tui"``.
    No flow logic lives here; the funnel handles resolution + the approval gate identically
    to every other surface.
    """

    surface_name = "tui"

    def rows(self) -> list[CapabilityRow]:
        """Return render-ready capability rows from the shared registry (for the pane table)."""
        from ..control_plane.capability import list_capabilities

        return [
            CapabilityRow(
                capability_id=c["capability_id"],
                description=c["description"],
                milestone=str(c["milestone"]),
                gated=bool(c["gated"]),
                runnable=bool(c["runnable"]),
            )
            for c in list_capabilities()
        ]

    def preview(self, capability_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Plan-Mode preview a capability for the pane (read-only; provisions nothing)."""
        from ..control_plane.capability import invoke_capability

        return invoke_capability(
            capability_id=capability_id, params=params, surface=self.surface_name, plan_only=True
        ).to_dict()

    def invoke(self, capability_id: str, params: dict[str, Any] | None = None,
               policy: Any | None = None) -> dict[str, Any]:
        """Invoke a capability from the TUI through the shared funnel (gated flows route to approval)."""
        from ..control_plane.capability import invoke_capability

        return invoke_capability(
            capability_id=capability_id, params=params, policy=policy, surface=self.surface_name
        ).to_dict()
