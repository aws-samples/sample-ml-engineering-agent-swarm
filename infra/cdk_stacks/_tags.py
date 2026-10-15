"""Shared cost-allocation tagging helper — WA-12 (Cost · expenditure awareness).

The audit found NO cost-allocation tagging strategy on the data/compute stacks, blocking
chargeback. ``apply_cost_tags`` stamps a consistent ``project`` / ``environment`` / ``owner``
tag set (plus ``managed-by=cdk``) on a stack so every resource that supports tags inherits
them. Values are env-configurable with safe OSS defaults — NO account/org-specific value is
hardcoded (the audit §6 cost-tag finding + SEC-01).

Env overrides:
    * ``AMLAS_COST_PROJECT``  (default ``amlas``)
    * ``AMLAS_ENV``           (default ``dev``) — also drives the WA-21 prod RETAIN flip
    * ``AMLAS_COST_OWNER``    (default ``amlas-ml-platform``)
"""

from __future__ import annotations

import os

from aws_cdk import Stack, Tags

from amlas.config._env_compat import env_with_compat


def cost_tag_values() -> dict[str, str]:
    """Resolve the cost-allocation tag values from env (safe defaults; no org-specific value)."""
    return {
        "project": env_with_compat("AMLAS_COST_PROJECT", "amlas"),
        "environment": env_with_compat("AMLAS_ENV", "dev"),
        "owner": env_with_compat("AMLAS_COST_OWNER", "amlas-ml-platform"),
        "managed-by": "cdk",
    }


def apply_cost_tags(stack: Stack) -> dict[str, str]:
    """Stamp project/environment/owner cost-allocation tags on every taggable resource (WA-12).

    Returns the applied tag dict (for outputs/tests). Idempotent — calling once per stack in
    ``__init__`` propagates to all child constructs via the CDK Tags aspect.
    """
    tags = cost_tag_values()
    for key, value in tags.items():
        Tags.of(stack).add(key, value)
    return tags


def deployment_tags(prefix: str) -> dict[str, str]:
    """Native ownership tags for resources that do not expose a CDK TagManager."""
    tags = {"amlas:deployment-prefix": prefix}
    deployment_id = os.environ.get("AMLAS_DEPLOYMENT_ID", "").strip()
    if deployment_id:
        tags["amlas:deployment-id"] = deployment_id
    return tags
