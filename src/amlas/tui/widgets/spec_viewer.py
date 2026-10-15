"""Spec viewer widget — formatted display of ML spec details."""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.widgets import Static


class SpecViewerWidget(Static):
    """Displays a formatted view of an ML spec."""

    def __init__(self, spec: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._spec = spec or {}

    def compose(self) -> ComposeResult:
        yield Static(self._format_spec(), id="spec-viewer-content")

    def update_spec(self, spec: dict[str, Any]) -> None:
        """Update displayed spec data."""
        self._spec = spec
        content = self.query_one("#spec-viewer-content", Static)
        content.update(self._format_spec())

    def _format_spec(self) -> str:
        """Format spec dict as readable text."""
        if not self._spec:
            return "No spec loaded. Select a spec from the list."

        lines = ["━━━ ML Spec ━━━", ""]
        for key, value in self._spec.items():
            if isinstance(value, list):
                lines.append(f"  {key}:")
                for item in value:
                    lines.append(f"    - {item}")
            else:
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)
