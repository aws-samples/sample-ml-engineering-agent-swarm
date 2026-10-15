"""Chat message and response data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class ChatMessage:
    """A single chat message in the conversation history."""

    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    attachments: list[str] = field(default_factory=list)
    action_taken: str = ""  # e.g., "Started pipeline run"
    # Hop-1 C-1 (assessment §D.2): a COMPACT per-turn tool-result summary (one short
    # "tool: k=v, k=v" line per executed tool, NOT a raw result dump). Populated on the
    # assistant turn that used tools so ``_build_agent_history`` can replay what was
    # already computed/looked up to the next turn ("what was that F1 again?"). Empty by
    # default — a plain (no-tool) turn carries no summary, so history replay is unchanged.
    # NB: this is sourced from the structured ``ToolExecution.result_summary`` trace, NOT
    # from the ephemeral ``ThoughtTrace`` ticker (which is never persisted — T-TT-05).
    tool_summary: str = ""

    @property
    def display_prefix(self) -> str:
        """Return the display prefix icon for the message role."""
        prefixes = {
            "user": "👤",
            "assistant": "🤖",
            "system": "ℹ️",
        }
        return prefixes.get(self.role, "•")


@dataclass
class ChatResponse:
    """Response from the chat engine after processing a message."""

    message: str
    action: str = ""  # What was done (e.g., "spec_generated", "pipeline_started")
    data: dict[str, Any] = field(default_factory=dict)  # Structured data
    suggestions: list[str] = field(default_factory=list)  # Follow-up suggestions
