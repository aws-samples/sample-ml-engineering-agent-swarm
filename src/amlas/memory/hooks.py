"""AgentCore Memory HookProvider — injects last-k turns into agent system prompt.

Port of `amazon-bedrock-agentcore-samples/01-tutorials/04-AgentCore-memory/
01-short-term-memory/01-single-agent/with-strands-agent/personal-agent.ipynb`.

Session scoping for Amlas:
    actor_id   = spec.project_id   (persists across runs for a project)
    session_id = orchestrator.run_id  (unique per run; ≥33 chars for A2A)
"""

from __future__ import annotations

from typing import Any, Protocol

from strands.hooks import (
    AgentInitializedEvent,
    HookProvider,
    HookRegistry,
    MessageAddedEvent,
)


class MemoryClientProto(Protocol):
    """Minimal surface of `bedrock_agentcore.memory.MemoryClient` we actually use.

    Enables dependency-injection for isolated validation — we pass a FakeMemoryClient that records
    calls instead of hitting real AgentCore Memory.
    """

    def get_last_k_turns(
        self, *, memory_id: str, actor_id: str, session_id: str, k: int
    ) -> list[dict[str, Any]]: ...

    def create_event(
        self,
        *,
        memory_id: str,
        actor_id: str,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


class MemoryHookProvider(HookProvider):
    """HookProvider that pulls history on init and persists new messages on turn boundaries."""

    def __init__(
        self,
        *,
        client: MemoryClientProto,
        memory_id: str,
        actor_id: str,
        session_id: str,
        k: int = 5,
    ):
        self.client = client
        self.memory_id = memory_id
        self.actor_id = actor_id
        self.session_id = session_id
        self.k = k
        self._messages_seen = 0

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(AgentInitializedEvent, self.on_agent_initialized)
        registry.add_callback(MessageAddedEvent, self.on_message_added)

    def on_agent_initialized(self, event: AgentInitializedEvent) -> None:
        """Inject last-k turns into the agent's conversation history."""
        try:
            turns = self.client.get_last_k_turns(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                k=self.k,
            )
        except Exception:
            # Memory read is best-effort — do not fail agent construction on a
            # transient error or when the Memory resource is unreachable
            # (mirrors on_message_added's best-effort write pattern).
            return
        if not turns:
            return
        # Append turns as messages on the agent (Strands Agent has .messages list)
        agent = getattr(event, "agent", None)
        if agent is None or not hasattr(agent, "messages"):
            return
        for turn in turns:
            agent.messages.append(turn)

    def on_message_added(self, event: MessageAddedEvent) -> None:
        """Persist new messages to AgentCore Memory after each turn."""
        agent = getattr(event, "agent", None)
        if agent is None or not hasattr(agent, "messages"):
            return
        new = agent.messages[self._messages_seen :]
        if not new:
            return
        try:
            self.client.create_event(
                memory_id=self.memory_id,
                actor_id=self.actor_id,
                session_id=self.session_id,
                messages=new,
            )
            self._messages_seen = len(agent.messages)
        except Exception:  # nosec B110
            # Memory write is best-effort — do not fail the agent loop on a transient error.
            pass
