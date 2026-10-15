"""Agentic-ML Configuration System.

This package provides two config layers:
1. Runtime config (model IDs, Bedrock settings, budgets) — from `runtime.py`
2. Settings config (AWS profile, S3 bucket, dirs, YAML files) — from `settings.py`

Backward compatibility: `from amlas.config import Config, load_config` still works.
"""

from __future__ import annotations

# --- Health check ---
from .health import AWSHealthStatus, check_aws_health

# --- Backward-compatible re-exports from runtime.py ---
from .runtime import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_REVIEWER_MODEL,
    GPT_6_1_SOL,
    HAIKU_4_5,
    OPUS_4_7,
    OPUS_4_7_CONTEXT_1M_BETA,
    SONNET_4_6,
    SONNET_5_5,
    BedrockConfig,
    Budgets,
    Config,
    ModelConfig,
    ModelRole,
    load_config,
)

# --- New settings system ---
from .settings import AWSConfig, LakeConfig, LocalConfig, MLOpsConfig, Settings

__all__ = [
    "DEFAULT_AGENT_MODEL",
    "DEFAULT_REVIEWER_MODEL",
    "GPT_6_1_SOL",
    "HAIKU_4_5",
    "OPUS_4_7",
    "OPUS_4_7_CONTEXT_1M_BETA",
    "SONNET_4_6",
    "SONNET_5_5",
    "AWSConfig",
    # Health check
    "AWSHealthStatus",
    "BedrockConfig",
    "Budgets",
    # Runtime (backward compat)
    "Config",
    "LakeConfig",
    "LocalConfig",
    "MLOpsConfig",
    "ModelConfig",
    "ModelRole",
    # New settings
    "Settings",
    "check_aws_health",
    "load_config",
]
