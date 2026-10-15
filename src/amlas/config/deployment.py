"""Dependency-free deployment naming shared by setup, infrastructure, and clients.

Selectors are read when called, so importing a module before loading a runtime
overlay never binds discovery to the legacy deployment. Execute this file directly
for JSON validation before installing project dependencies.
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from collections.abc import Mapping

DEFAULT_DEPLOYMENT_PREFIX = "amlas"
_DEPLOYMENT_PREFIX = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_SSM_PREFIX = re.compile(r"/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*")


def resolve_deployment_prefix(*, env: Mapping[str, str] | None = None) -> str:
    """Resolve a lowercase resource prefix of at most 24 characters."""
    source = os.environ if env is None else env
    prefix = (source.get("AMLAS_DEPLOYMENT_PREFIX") or DEFAULT_DEPLOYMENT_PREFIX).strip()
    return validate_deployment_prefix(prefix)


def validate_deployment_prefix(prefix: str) -> str:
    """Reject an invalid resource prefix before constructing cloud names."""
    if len(prefix) > 24 or not _DEPLOYMENT_PREFIX.fullmatch(prefix):
        raise ValueError(
            "deployment prefix must start with a lowercase letter, contain only "
            "lowercase letters, digits and single separating hyphens, and be at most 24 characters"
        )
    return prefix


def resolve_ssm_prefix(*, env: Mapping[str, str] | None = None) -> str:
    """Resolve the SSM path namespace, defaulting to the deployment prefix."""
    source = os.environ if env is None else env
    configured = (source.get("AMLAS_SSM_PREFIX") or "").strip()
    prefix = configured.rstrip("/") if configured else "/" + resolve_deployment_prefix(env=source)
    if not _SSM_PREFIX.fullmatch(prefix) or len(prefix) > 1011:
        raise ValueError(
            "SSM prefix must be an absolute path with non-empty alphanumeric, dot, underscore or hyphen segments"
        )
    return prefix


def deployment_stack_name(logical_id: str, prefix: str) -> str:
    """Preserve legacy stack names and isolate explicitly selected deployments."""
    validate_deployment_prefix(prefix)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9-]*", logical_id):
        raise ValueError("logical stack id must be a CloudFormation-compatible name")
    return logical_id if prefix == DEFAULT_DEPLOYMENT_PREFIX else f"{prefix}-{logical_id}"


def namespace_parameter_name(path: str, *, env: Mapping[str, str] | None = None) -> str:
    """Remap registry paths under /amlas while preserving explicitly unrelated paths."""
    if path == "/amlas" or path.startswith("/amlas/"):
        return resolve_ssm_prefix(env=env) + path[len("/amlas") :]
    return path


def deployment_job_tags(runtime_session_id: str) -> list[dict[str, str]]:
    """Bind an owned managed submit atomically; legacy unowned SDK callers stay unowned."""
    deployment_id = os.environ.get("AMLAS_DEPLOYMENT_ID", "").strip()
    if not deployment_id:
        return []
    try:
        uuid.UUID(deployment_id)
    except ValueError as error:
        raise ValueError("deployment ownership id must be a UUID before managed job submission") from error
    if not runtime_session_id or len(runtime_session_id) > 256:
        raise ValueError("owned managed job requires a non-empty runtime session correlation")
    return [
        {"Key": "amlas:deployment-id", "Value": deployment_id},
        {"Key": "amlas:deployment-prefix", "Value": resolve_deployment_prefix()},
        {"Key": "amlas:runtime-session-id", "Value": runtime_session_id},
    ]


if __name__ == "__main__":
    try:
        print(
            json.dumps({"deployment_prefix": resolve_deployment_prefix(), "ssm_prefix": resolve_ssm_prefix()})
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2) from error
