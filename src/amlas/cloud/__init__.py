"""Phase 14b cloud package — dispatch + runtime-side staging helpers.

This package is deliberately **boto3-only** (NO `strands` / `bedrock_agentcore`
imports at module top level). That keeps every module here directly verifiable on a
plain dev machine with real AWS, and lets the AgentCore Runtime entrypoint
(`app.py`, which DOES import `bedrock_agentcore`) call into it as a thin shim.

Modules:
    staging   — N1+N2: resolve a spec's golden_dataset to concrete train/test
                files on the container filesystem + materialize an aligned
                y_true.npy for the Arbiter.
    dispatch  — B1/B4: NL → spec → stage to session S3 → build app.py payload →
                InvokeAgentRuntime → stream/parse the response.
    secrets   — B3 mechanism: resolve TAVILY_API_KEY from Secrets Manager into
                the process env at runtime startup (graceful no-op if absent).
"""

from __future__ import annotations
