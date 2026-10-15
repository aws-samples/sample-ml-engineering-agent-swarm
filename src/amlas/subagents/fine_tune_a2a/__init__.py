"""Fine-Tune A2A sub-runtime — separate AgentCore Runtime for long GPU fine-tunes.

Phase J wires the real A2AServer + MetadataAwareExecutor pattern from
`sample-strands-agent-with-agentcore/agent-blueprint/agentcore-runtime-a2a-stack/
research-agent/src/main.py`. v1 ships the scaffold (module + a test that the
entrypoint decorator is registered). Real deploy is in Phase K onward.
"""
