"""A2A-compatible AgentCore Runtime entry point for the Fine-Tune sub-agent.

Port of the canonical pattern from `sample-strands-agent-with-agentcore/
agent-blueprint/agentcore-runtime-a2a-stack/research-agent/src/main.py`:
    - MetadataAwareExecutor streams tool events as A2A artifacts
    - Per-model agent cache (model_id → Agent instance)
    - A2A session IDs MUST be ≥33 chars — we enforce at entry.

Deploy as a second AgentCore Runtime (`amlas_ml_finetune`). The main Runtime
invokes this one via A2A for long GPU fine-tunes.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from ...subagents.fine_tune import build_fine_tune_agent

# Universal error handling: the AgentCore runtime is a distinct process
# entrypoint, so initialize the centralized error surface (real per-session JSONL file +
# the CloudWatch mirror in managed modes) and install the process-level uncaught-exception
# funnel at module import. Best-effort — a telemetry init failure must never block the
# runtime from serving invocations.
try:
    from ..._runtime_guard import install_error_excepthook
    from ...telemetry.error_log import init_error_log

    init_error_log()
    install_error_excepthook()
except Exception:  # nosec B110
    pass

app = BedrockAgentCoreApp()

# Per-model agent cache (populated lazily on first invoke per model_id)
_AGENT_CACHE: dict[str, Any] = {}


A2A_MIN_SESSION_ID_LEN = 33


@app.entrypoint
async def fine_tune_invocation(payload: dict[str, Any]) -> AsyncGenerator[dict[str, Any], None]:
    """A2A entry point — receives fine-tune requests, delegates to the fine_tune sub-agent.

    Payload contract:
        { "prompt": "<fine-tune request>",
          "model_id": "<optional — overrides default>",
          "session_id": "<≥33 chars>" }
    """
    session_id = payload.get("session_id", "")
    if len(session_id) < A2A_MIN_SESSION_ID_LEN:
        yield {
            "event": "error",
            "error": f"A2A session_id must be ≥{A2A_MIN_SESSION_ID_LEN} chars; got {len(session_id)}",
        }
        return

    model_id = payload.get("model_id", "default")
    agent = _AGENT_CACHE.get(model_id)
    if agent is None:
        agent = build_fine_tune_agent(model=None)  # production path — real Bedrock
        _AGENT_CACHE[model_id] = agent

    prompt = payload.get("prompt", "")
    if not prompt:
        yield {"event": "error", "error": "payload must include 'prompt'"}
        return

    yield {"event": "a2a_invocation_started", "session_id": session_id}
    stream = agent.stream_async(prompt)
    async for ev in stream:
        yield ev
    yield {"event": "a2a_invocation_complete"}


if __name__ == "__main__":  # pragma: no cover
    app.run()
