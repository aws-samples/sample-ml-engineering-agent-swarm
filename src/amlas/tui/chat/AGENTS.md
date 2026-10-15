# src/amlas/tui/chat

## Purpose

Own conversational turns, deterministic command handling, tool binding, dispatch and visible thought/tool traces.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Free-form chat uses the tool-bound agent; deterministic slash commands keep their own handler paths.
- Model/output/history configuration is owned by the shared configuration accessors; preserve the visible degraded-mode behavior.
- Honor existing scoped execution/spend authority in the initial request while
  retaining actual approval/resource gates. Reuse selected deployment references;
  report concrete execution handles or blockers rather than a staged-work claim.
- Optional caller policy is snapshotted per turn and transported through SDK
  invocation_state. It is never inserted into model input or stored globally.
  Missing/invalid authority retains the capability funnel's safe default.
- `agent_tools.py:tool_display_name` — Best-effort tool name (Strands ``@tool`` sets ``tool_name``/``__name__``).
- `agent_tools.py:get_safe_llm_tools` — Return the curated list of real ``@tool`` callables bound to the LLM agent.
- `command_queue.py:QueuedCommand` — One queued chat submission: an immutable (id, text) pair.
- `command_queue.py:CommandQueue` — FIFO queue of pending chat submissions with front-promotion.
- Owned files: `__init__.py`, `agent_tools.py`, `command_queue.py`, `commands.py`, `engine.py`, `flow_dispatch.py`, `intent_agent.py`, `models.py`, `prompt_cache.py`, `self_context.py`, `thought_trace.py`, `tool_binding_backend.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
