"""Guarded AWS control-plane tools — the agents configure AWS themselves (plan §A2/§A3).

**The gap these close.** The 7 AgentCore leads orchestrate the ML lifecycle (ADR-0002) but
could only *consume* a hand-provisioned account: the sole IAM calls in the package were two
read-only ``simulate_principal_policy`` probes in
[`dispatch_preflight.py`](dispatch_preflight.py), and there were no ``create_role`` /
``put_role_policy`` / ``create_bucket`` / quota call sites anywhere. So an agent could
*diagnose* a missing ``iam:PassRole`` and never fix it — the shift-left loop dead-ended at a
prose ``remediation`` string that nothing consumed. These tools consume it.

**Discipline every tool here follows** (deliberately uniform, so the guarantees are
auditable in one place):

1. **Policy first.** Every mutation calls
   :func:`~amlas.control_plane.aws_resource_policy.authorize` BEFORE constructing a boto3
   client. On denial the tool returns ``{"success": False, "denied": True, ...}`` and no
   client is ever built — the deny path cannot touch AWS.
2. **Plan by default.** ``plan_only=True`` is the default on every mutator, so an agent must
   explicitly ask to apply. A plan result carries the exact one-liner (and, for IAM, the
   rendered policy document) the operator would approve.
3. **Idempotent ``ensure_*``.** Re-running is a successful no-op reporting
   ``already_present=True`` — safe for an agent retry loop.
4. **Never raises.** Wrapped in :func:`~amlas.errors.tool_error_boundary` so a botocore
   failure becomes the standard error dict.

Compute placement (ADR-0002) is unaffected: these are *control-plane* configuration calls
(create a prefix, a database, a role, a grant, a quota request), not compute. All compute
still offloads to SageMaker/Glue/Lambda/Fargate.
"""

from __future__ import annotations

import json
import re
from typing import Any

from strands import tool

from ..control_plane.aws_resource_policy import (
    ALLOWED_ACTIONS,
    Decision,
    ResourceControlPolicy,
    ResourceRequest,
    authorize,
    lint_policy_document,
    resolve_control_region,
)
from ..errors import tool_error_boundary

__all__ = [
    "describe_aws_readiness",
    "ensure_glue_database",
    "ensure_iam_service_role",
    "ensure_s3_prefix",
    "grant_role_policy",
    "remediate_dispatch_preflight",
    "request_quota_increase",
]

# A well-formed AWS service principal, e.g. ``sagemaker.amazonaws.com``. Anything else
# (notably ``*``) is refused before the policy check — a wildcard trust policy would make
# the created role assumable by the world.
_SERVICE_PRINCIPAL_RE = re.compile(r"^[a-z0-9.\-]+\.amazonaws\.com$")

# The zero-byte marker object that materializes an S3 "directory" prefix. S3 has no real
# directories; a keep-marker is the conventional way to make a prefix exist and listable.
_PREFIX_MARKER = ".amlas-keep"


def _denied(decision: Decision) -> dict[str, Any]:
    """Render a policy denial as the standard tool result (no AWS was touched)."""
    return {
        "success": False,
        "denied": True,
        "reason": decision.reason,
        "plan": decision.plan,
        "findings": list(decision.findings),
        "planned": False,
        "applied": False,
    }


def _planned(decision: Decision, **extra: Any) -> dict[str, Any]:
    """Render an authorized-but-not-applied result (the ``plan_only`` default)."""
    return {
        "success": True,
        "denied": False,
        "planned": True,
        "applied": False,
        "already_present": False,
        "reason": decision.reason,
        "plan": decision.plan,
        "findings": [],
        **extra,
    }


def _applied(decision: Decision, *, already_present: bool = False, **extra: Any) -> dict[str, Any]:
    """Render a successfully-applied (or already-satisfied) mutation result."""
    return {
        "success": True,
        "denied": False,
        "planned": False,
        "applied": not already_present,
        "already_present": already_present,
        "reason": decision.reason,
        "plan": decision.plan,
        "findings": [],
        **extra,
    }


def _region(region: str | None) -> str:
    """Resolve the effective region via the SHARED resolver the policy's pin also uses.

    Delegating (rather than reading a module-level default here) is what keeps a tool's
    chosen region and the policy's pinned region identical — otherwise every mutation is
    denied by its own guard.
    """
    return resolve_control_region(region)


# ─── S3 lake prefixes ─────────────────────────────────────────────────────────


@tool
@tool_error_boundary("tools.aws_control")
def ensure_s3_prefix(
    bucket: str,
    prefix: str,
    *,
    region: str | None = None,
    plan_only: bool = True,
) -> dict[str, Any]:
    """Ensure an S3 lake prefix exists (idempotent), so a job can write there.

    Materializes the prefix with a zero-byte keep-marker — S3 has no real directories, and
    several managed services (Glue crawlers, SageMaker output paths) behave better against
    a prefix that already lists.

    Args:
        bucket: The lake bucket; must carry the platform resource prefix.
        prefix: The key prefix to ensure, e.g. ``"silver/bogie/"``.
        region: Target region (defaults to the configured region).
        plan_only: When True (default) report the plan without writing anything.

    Returns:
        Standard control result: ``success`` / ``denied`` / ``planned`` / ``applied`` /
        ``already_present`` / ``reason`` / ``plan``.
    """
    reg = _region(region)
    key_prefix = (prefix or "").strip().lstrip("/")
    if not key_prefix.endswith("/"):
        key_prefix = f"{key_prefix}/"

    decision = authorize(
        ResourceRequest(
            service="s3",
            action="PutObject",
            resource_name=f"{bucket}/{key_prefix}",
            region=reg,
            detail="ensure lake prefix exists",
        )
    )
    if not decision.allowed:
        return _denied(decision)
    if plan_only:
        return _planned(decision, bucket=bucket, prefix=key_prefix)

    import boto3

    s3 = boto3.client("s3", region_name=reg)
    existing = s3.list_objects_v2(Bucket=bucket, Prefix=key_prefix, MaxKeys=1)
    if existing.get("KeyCount", 0) > 0:
        return _applied(decision, already_present=True, bucket=bucket, prefix=key_prefix)

    s3.put_object(Bucket=bucket, Key=f"{key_prefix}{_PREFIX_MARKER}", Body=b"")
    return _applied(decision, bucket=bucket, prefix=key_prefix)


# ─── Glue catalog ─────────────────────────────────────────────────────────────


@tool
@tool_error_boundary("tools.aws_control")
def ensure_glue_database(
    database: str,
    *,
    description: str = "",
    region: str | None = None,
    plan_only: bool = True,
) -> dict[str, Any]:
    """Ensure a Glue Data Catalog database exists (idempotent).

    The catalog entry a crawler / Athena query / lake table registration needs before the
    data-engineering lead can register a dataset version.

    Args:
        database: Database name; must carry the platform resource prefix.
        description: Optional catalog description.
        region: Target region (defaults to the configured region).
        plan_only: When True (default) report the plan without creating anything.

    Returns:
        Standard control result.
    """
    reg = _region(region)
    decision = authorize(
        ResourceRequest(
            service="glue",
            action="CreateDatabase",
            resource_name=database,
            region=reg,
            detail="ensure catalog database exists",
        )
    )
    if not decision.allowed:
        return _denied(decision)
    if plan_only:
        return _planned(decision, database=database)

    import boto3
    from botocore.exceptions import ClientError

    glue = boto3.client("glue", region_name=reg)
    try:
        glue.get_database(Name=database)
        return _applied(decision, already_present=True, database=database)
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "EntityNotFoundException":
            raise

    payload: dict[str, Any] = {"Name": database}
    if description:
        payload["Description"] = description
    glue.create_database(DatabaseInput=payload)
    return _applied(decision, database=database)


# ─── IAM: roles + scoped grants ────────────────────────────────────────────────


@tool
@tool_error_boundary("tools.aws_control")
def ensure_iam_service_role(
    role_name: str,
    service_principal: str,
    *,
    description: str = "",
    region: str | None = None,
    plan_only: bool = True,
) -> dict[str, Any]:
    """Ensure an IAM service role exists that ``service_principal`` may assume (idempotent).

    This is the role a dispatched job runs as (e.g. the SageMaker training/processing
    execution role). The trust policy is built here — never accepted from the caller — so
    it is always a single-service ``sts:AssumeRole``, never a wildcard principal.

    Args:
        role_name: Role name; must carry the platform resource prefix.
        service_principal: The AWS service that may assume it, e.g.
            ``"sagemaker.amazonaws.com"``. A wildcard or malformed principal is refused.
        description: Optional role description.
        region: Region used for the IAM client (IAM is global; this only affects endpoint).
        plan_only: When True (default) report the plan without creating the role.

    Returns:
        Standard control result plus ``role_arn`` when known.
    """
    reg = _region(region)
    principal = (service_principal or "").strip().lower()
    if not _SERVICE_PRINCIPAL_RE.match(principal):
        return {
            "success": False,
            "denied": True,
            "planned": False,
            "applied": False,
            "reason": (
                f"service_principal {service_principal!r} is not a valid AWS service "
                "principal (expected e.g. 'sagemaker.amazonaws.com'); a wildcard trust "
                "policy would make the role world-assumable"
            ),
            "plan": f"iam:CreateRole {role_name}",
            "findings": [],
        }

    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": principal},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    decision = authorize(
        ResourceRequest(
            service="iam",
            action="CreateRole",
            resource_name=role_name,
            region=reg,
            detail=f"assumable by {principal}",
        )
    )
    if not decision.allowed:
        return _denied(decision)
    if plan_only:
        return _planned(decision, role_name=role_name, trust_policy=trust)

    import boto3
    from botocore.exceptions import ClientError

    iam = boto3.client("iam", region_name=reg)
    try:
        existing = iam.get_role(RoleName=role_name)["Role"]
        return _applied(
            decision,
            already_present=True,
            role_name=role_name,
            role_arn=existing.get("Arn", ""),
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "NoSuchEntity":
            raise

    created = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust),
        Description=description or f"amlas agent-provisioned role for {principal}",
    )["Role"]
    return _applied(decision, role_name=role_name, role_arn=created.get("Arn", ""))


@tool
@tool_error_boundary("tools.aws_control")
def grant_role_policy(
    role_name: str,
    policy_name: str,
    actions: list[str],
    resources: list[str],
    *,
    region: str | None = None,
    plan_only: bool = True,
) -> dict[str, Any]:
    """Attach a SCOPED inline policy to a role (the shift-left permission grant).

    The policy document is CONSTRUCTED here from ``actions`` + ``resources`` and then run
    through :func:`~amlas.control_plane.aws_resource_policy.lint_policy_document`, so a
    wildcard action, an unscoped ``Resource: "*"``, or any re-granting IAM action is
    refused before AWS is touched. This is the single call that lets an agent fix the
    ``iam:PassRole`` / lake-read gaps ``dispatch_preflight`` reports.

    Args:
        role_name: Target role; must carry the platform resource prefix.
        policy_name: Inline policy name; must carry the platform resource prefix.
        actions: Exact API actions, e.g. ``["s3:GetObject"]``. Wildcards are refused.
        resources: Exact ARNs the actions apply to.
        region: Region used for the IAM client.
        plan_only: When True (default) return the rendered document without attaching it.

    Returns:
        Standard control result plus ``policy_document`` (the document that was/would be
        attached) and ``findings`` when the linter refused it.
    """
    reg = _region(region)
    document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [str(a) for a in (actions or [])],
                "Resource": [str(r) for r in (resources or [])],
            }
        ],
    }

    if not (actions and resources):
        return {
            "success": False,
            "denied": True,
            "planned": False,
            "applied": False,
            "reason": "both actions and resources are required — an empty grant is refused",
            "plan": f"iam:PutRolePolicy {policy_name} on {role_name}",
            "findings": [],
        }

    decision = authorize(
        ResourceRequest(
            service="iam",
            action="PutRolePolicy",
            resource_name=role_name,
            region=reg,
            policy_document=document,
            detail=f"inline policy {policy_name}",
        )
    )
    if not decision.allowed:
        result = _denied(decision)
        # Surface the lint findings even when the denial came from another check.
        result["findings"] = decision.findings or lint_policy_document(document)
        result["policy_document"] = document
        return result
    if plan_only:
        return _planned(
            decision, role_name=role_name, policy_name=policy_name, policy_document=document
        )

    import boto3

    iam = boto3.client("iam", region_name=reg)
    iam.put_role_policy(
        RoleName=role_name, PolicyName=policy_name, PolicyDocument=json.dumps(document)
    )
    return _applied(
        decision, role_name=role_name, policy_name=policy_name, policy_document=document
    )


# ─── Service quotas ───────────────────────────────────────────────────────────


@tool
@tool_error_boundary("tools.aws_control")
def request_quota_increase(
    service_code: str,
    quota_code: str,
    desired_value: float,
    *,
    region: str | None = None,
    plan_only: bool = True,
) -> dict[str, Any]:
    """Request a service-quota increase (the classic autonomy blocker).

    Concurrent Code-Interpreter sessions, SageMaker instance caps, and endpoint limits all
    surface as ``ServiceQuotaExceededException`` mid-run. This lets the agent raise the
    ceiling instead of failing the whole experiment.

    Args:
        service_code: AWS service code, e.g. ``"sagemaker"``.
        quota_code: The quota code, e.g. ``"L-1234ABCD"``.
        desired_value: The requested new value; must be > 0.
        region: Target region.
        plan_only: When True (default) report the plan without filing the request.

    Returns:
        Standard control result plus ``request_id`` when a request was filed.
    """
    reg = _region(region)
    if not desired_value or float(desired_value) <= 0:
        return {
            "success": False,
            "denied": True,
            "planned": False,
            "applied": False,
            "reason": f"desired_value must be > 0 (got {desired_value!r})",
            "plan": f"servicequotas:RequestServiceQuotaIncrease {quota_code}",
            "findings": [],
        }

    # Quota codes are AWS-assigned (``L-…``) and cannot carry our platform prefix, so the
    # authorized resource is the amlas-scoped quota REQUEST, with the codes in the detail.
    decision = authorize(
        ResourceRequest(
            service="servicequotas",
            action="RequestServiceQuotaIncrease",
            resource_name=f"amlas-quota-{service_code}",
            region=reg,
            detail=f"{quota_code} → {desired_value}",
        )
    )
    if not decision.allowed:
        return _denied(decision)
    if plan_only:
        return _planned(
            decision,
            service_code=service_code,
            quota_code=quota_code,
            desired_value=float(desired_value),
        )

    import boto3

    quotas = boto3.client("service-quotas", region_name=reg)
    response = quotas.request_service_quota_increase(
        ServiceCode=service_code, QuotaCode=quota_code, DesiredValue=float(desired_value)
    )
    request_id = (
        response.get("RequestedQuota", {}).get("Id", "") if isinstance(response, dict) else ""
    )
    return _applied(
        decision,
        service_code=service_code,
        quota_code=quota_code,
        desired_value=float(desired_value),
        request_id=request_id,
    )


# ─── Readiness (read-only) ────────────────────────────────────────────────────


@tool
@tool_error_boundary("tools.aws_control")
def describe_aws_readiness(*, region: str | None = None) -> dict[str, Any]:
    """Report control policy and selected deployment configuration (read-only).

    Lets the agent (and the operator) see the active guard — the resource prefix, the
    pinned regions, service/action allowlist and configured resource references before
    proposing a change. References resolve through the normal environment/selected SSM
    namespace; they do not prove resource existence, ownership or usable permissions.

    Args:
        region: Selected region for configuration discovery.

    Returns:
        Policy fields plus ``deployment`` references and unresolved configuration keys.
    """
    from ..config import externalized
    from ..config.deployment import resolve_deployment_prefix, resolve_ssm_prefix

    policy = ResourceControlPolicy.from_env()
    reg = _region(region)
    resources = {
        name: externalized.resolve(name, region=reg)
        for name in (
            "lake_bucket", "glue_database", "session_bucket",
            "sagemaker_execution_role_arn", "agentcore_runtime_arn", "model_package_group",
        )
    }
    return {
        "success": True,
        "resource_prefix": policy.resource_prefix,
        "allowed_regions": list(policy.allowed_regions),
        "allowed_services": sorted(policy.allowed_actions),
        "allowed_actions": {
            svc: sorted(acts) for svc, acts in sorted(policy.allowed_actions.items())
        },
        "deployment": {
            "prefix": resolve_deployment_prefix(),
            "ssm_prefix": resolve_ssm_prefix(),
            "region": reg,
            "resources": resources,
            "unresolved": [name for name, value in resources.items() if not value],
            "verification": "configuration_only",
        },
        "note": (
            "Policy permits only the listed mutations. Use selected references instead "
            "of guessing resources. Unresolved means missing or unverifiable configuration; "
            "configured references still require native existence/permission checks. "
            "Do not create replacement identities or resources to conceal a discovery gap."
        ),
    }


# ─── A3: close the dispatch-preflight loop ───────────────────────────────────

# Which control tool fixes which ``dispatch_preflight`` check. The preflight already emits
# a per-check ``remediation`` string; this table turns each one into an actionable plan.
_PREFLIGHT_FIX: dict[str, str] = {
    "session_bucket": "ensure_s3_prefix",
    "pass_role": "grant_role_policy",
    "kms_jobs_table": "grant_role_policy",
    "image": "manual",  # an ECR image must be built+pushed; not an IAM/resource fix.
}


@tool
@tool_error_boundary("tools.aws_control")
def remediate_dispatch_preflight(
    preflight: dict[str, Any],
    *,
    region: str | None = None,
    plan_only: bool = True,
) -> dict[str, Any]:
    """Turn a failed :func:`~amlas.tools.dispatch_preflight.dispatch_preflight` into fixes.

    Closes the shift-left loop the audit found open: ``dispatch_preflight`` returns
    ``{"status": "fail", "remediation": "<prose>"}`` per check and **nothing consumed
    it**, so a blocked dispatch required a human CDK deploy. This maps each failed check
    to the control tool that fixes it and returns an ordered remediation plan.

    Only ``status == "fail"`` checks produce a remediation (``ok`` / ``unknown`` are left
    alone — an ``unknown`` means we could not determine the state, and guessing a mutation
    there would be unsafe).

    Args:
        preflight: The dict returned by ``dispatch_preflight``.
        region: Target region for the proposed fixes.
        plan_only: When True (default) only plan. Applying is intentionally left to the
            individual control tools so each mutation gets its own approval decision.

    Returns:
        ``{"success": True, "planned": bool, "remediations": [{check, tool, detail,
        remediation, plan}], "ready": bool}``.
    """
    reg = _region(region)
    checks = (preflight or {}).get("checks")
    checks = checks if isinstance(checks, dict) else {}

    remediations: list[dict[str, Any]] = []
    for name, check in checks.items():
        if not isinstance(check, dict):
            continue
        if str(check.get("status", "")).lower() != "fail":
            continue
        fix_tool = _PREFLIGHT_FIX.get(name, "manual")
        remediations.append(
            {
                "check": name,
                "tool": fix_tool,
                "detail": str(check.get("detail", "")),
                "remediation": str(check.get("remediation", "")),
                "plan": (
                    f"{name}: call {fix_tool} in {reg} — "
                    f"{check.get('remediation') or check.get('detail') or 'see check detail'}"
                ),
            }
        )

    return {
        "success": True,
        "planned": plan_only,
        "applied": False,
        "ready": bool((preflight or {}).get("ready")),
        "remediations": remediations,
        "summary": (
            "preflight already ready — nothing to remediate"
            if not remediations
            else f"{len(remediations)} failed check(s) have an actionable fix"
        ),
    }


# Re-exported for callers that want the allowlist without importing the control-plane
# module directly (e.g. a prompt builder describing the agents' AWS reach).
AGENT_ALLOWED_AWS_ACTIONS = ALLOWED_ACTIONS
