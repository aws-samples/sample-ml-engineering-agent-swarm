"""Shared helpers for the MLOps tools (client seam, region, name sanitizers).

Kept import-light: only ``boto3`` is imported **lazily inside** ``make_*_client``
so ``tools.mlops`` imports cleanly without the ``mlops`` extra (matching the
``tools.imputation`` / ``tools.synthetic`` convention). Every tool accepts an
injectable client (``sm_client`` / ``fs_runtime_client``) so callers can pass a
real-AWS-backed boto3 client in the selected region. Every MLOps
surface — Pipelines, Registry, Clarify, Model Monitor, Feature Store — is
exercised against REAL AWS (heavy provisioning gated ``expensive``).

SEC-01: the region default is sourced from the single canonical
``config.runtime.DEFAULT_REGION`` rather than a re-declared ``"us-west-2"``.
"""

from __future__ import annotations

import re
from typing import Any

from ...config.aws_region import resolve_aws_client_region, resolve_aws_region

# SEC-01 / CFG-01: one region source of truth.
from ...config.runtime import DEFAULT_REGION

# Reuse the SM-01 GPU image/instance selector instead of duplicating it
# (master plan §4 "Reuses existing tools/sagemaker.py").
from ..sagemaker import _select_image_and_instance

__all__ = [
    "DEFAULT_REGION",
    "MISSING_BOTO3",
    "make_fs_runtime_client",
    "make_sm_client",
    "sanitize_name",
    "training_image_for",
]

MISSING_BOTO3 = {
    "success": False,
    "error": "MLOps dependencies are unavailable; rerun scripts/setup.sh",
}

# SageMaker resource names: [a-zA-Z0-9](-*[a-zA-Z0-9]){0,62} — no underscores.
_NAME_CLEAN = re.compile(r"[^a-zA-Z0-9-]+")


def sanitize_name(raw: str, *, max_len: int = 63) -> str:
    """Coerce an arbitrary string into a valid SageMaker resource name.

    SageMaker names must match ``[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}`` (hyphens,
    no underscores — the same constraint the Bedrock job-name rule enforces).
    Underscores → hyphens, illegal chars dropped, collapsed, length-capped, and
    guaranteed to start with an alphanumeric.
    """
    s = str(raw).strip().replace("_", "-")
    s = _NAME_CLEAN.sub("-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    if not s:
        s = "amlas"
    if not s[0].isalnum():
        s = f"w{s}"
    return s[:max_len].rstrip("-")


def make_sm_client(region: str | None = None, client: Any | None = None) -> Any:
    """Return an injected SageMaker client, else build a default boto3 one.

    Callers may pass ``client=`` (a real-AWS-backed boto3 client); prod leaves it
    ``None`` and we build ``boto3.client("sagemaker", ...)`` lazily.
    """
    if client is not None:
        resolve_aws_client_region(client, region, required=False)
        return client
    import boto3  # local import keeps module import cheap + boto-optional

    return boto3.client("sagemaker", region_name=resolve_aws_region(region))


def make_fs_runtime_client(region: str | None = None, client: Any | None = None) -> Any:
    """Return an injected Feature Store *runtime* client, else a default one.

    The online put/get record APIs live on the separate
    ``sagemaker-featurestore-runtime`` service, not ``sagemaker``.
    """
    if client is not None:
        return client
    import boto3

    return boto3.client("sagemaker-featurestore-runtime", region_name=resolve_aws_region(region))


def training_image_for(training_type: str, config: dict[str, Any] | None = None) -> tuple[str, str]:
    """Pick (image_uri, instance_type) via the SM-01 selector (GPU for fine-tune/RL)."""
    selected = dict(config or {})
    return _select_image_and_instance(str(training_type or "train"), selected, region=selected.get("region"))
