"""Pipeline progress widget — stage-by-stage status indicators."""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static


class PipelineStagesWidget(Static):
    """Displays pipeline stages with status indicators.

    Status: ○ pending, ⟳ running, ✓ completed, ✗ failed
    """

    STAGES: ClassVar[list[str]] = ["Research", "Plan", "Training", "Evaluate", "Leaderboard", "Decision"]

    def compose(self) -> ComposeResult:
        with Horizontal(classes="stages-row"):
            for i, stage in enumerate(self.STAGES):
                yield Static(f"○ {stage}", id=f"stage-{i}", classes="stage-indicator pending")

    def set_stage_running(self, index: int) -> None:
        """Mark a stage as running."""
        if 0 <= index < len(self.STAGES):
            indicator = self.query_one(f"#stage-{index}", Static)
            indicator.update(f"⟳ {self.STAGES[index]}")
            indicator.remove_class("pending", "complete", "failed")
            indicator.add_class("running")

    def set_stage_complete(self, index: int) -> None:
        """Mark a stage as complete."""
        if 0 <= index < len(self.STAGES):
            indicator = self.query_one(f"#stage-{index}", Static)
            indicator.update(f"✓ {self.STAGES[index]}")
            indicator.remove_class("pending", "running", "failed")
            indicator.add_class("complete")

    def set_stage_failed(self, index: int) -> None:
        """Mark a stage as failed."""
        if 0 <= index < len(self.STAGES):
            indicator = self.query_one(f"#stage-{index}", Static)
            indicator.update(f"✗ {self.STAGES[index]}")
            indicator.remove_class("pending", "running", "complete")
            indicator.add_class("failed")

    def reset_all(self) -> None:
        """Reset all stages to pending."""
        for i, stage in enumerate(self.STAGES):
            indicator = self.query_one(f"#stage-{i}", Static)
            indicator.update(f"○ {stage}")
            indicator.remove_class("running", "complete", "failed")
            indicator.add_class("pending")
