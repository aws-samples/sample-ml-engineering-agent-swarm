"""Agentic-ML Terminal UI — full-screen interactive interface using Textual."""

# Lazy import to avoid requiring textual at module-load time for testing
# Import AgenticMLApp only when needed (e.g., when running the TUI)


def __getattr__(name: str) -> object:
    """Lazy import of TUI components requiring textual."""
    if name == "AgenticMLApp":
        from .app import AgenticMLApp

        return AgenticMLApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AgenticMLApp"]
