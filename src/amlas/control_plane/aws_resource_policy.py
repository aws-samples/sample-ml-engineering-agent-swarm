"""The AWS resource-control policy chokepoint (universal-ml-control-plane plan §A1).

**Why this module exists.** The 7 AgentCore leads orchestrate the ML lifecycle
(ADR-0002) but until now could only *consume* a hand-provisioned AWS account: the only
IAM calls in the package were two read-only ``simulate_principal_policy`` probes in
[`tools/dispatch_preflight.py`](../tools/dispatch_preflight.py), so an agent could
*diagnose* a missing ``iam:PassRole`` and never fix it. Shifting resource + permission
configuration LEFT to the agents means they need real mutating AWS control calls — which
means those calls need one deterministic, testable guard in front of them.

This is that guard. It is **pure Python** (no boto3, no network, no credentials), so the
entire authorization surface is directly verifiable offline. Every mutating control tool in
[`tools/aws_control.py`](../tools/aws_control.py) calls :func:`authorize` BEFORE it
constructs a client, and refuses to proceed on a denial.

Five independent checks, all of which must pass (deny-by-default at every one):

1. **Service allowlist** — only the ML-platform services in :data:`ALLOWED_ACTIONS`.
2. **Per-service action allowlist** — an allowed service does not imply an allowed verb
   (``s3:PutObject`` yes, ``s3:DeleteBucket`` never).
3. **Resource-name prefix** — every resource the agents create or touch must start with
   the configured prefix (default ``amlas-``), so an agent can never reach a resource the
   platform does not own. Wildcards and ``..`` traversal are rejected outright.
4. **Region pin** — the request region must be in the configured allowlist.
5. **IAM policy-document lint** — for grant actions, the attached document is linted for
   privilege escalation (:func:`lint_policy_document`).

The returned :class:`Decision` always carries a human ``plan`` line so a ``plan_only``
tool call can show the operator exactly what WOULD happen without doing it.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from ..config.aws_region import resolve_aws_region
from ..config.deployment import resolve_deployment_prefix

__all__ = [
    "ALLOWED_ACTIONS",
    "DEFAULT_RESOURCE_PREFIX",
    "RESOURCE_WILDCARD_EXEMPT_ACTIONS",
    "Decision",
    "ResourceControlPolicy",
    "ResourceRequest",
    "authorize",
    "lint_policy_document",
    "resolve_control_region",
]

# The resource-name prefix every agent-created/agent-touched AWS resource must carry.
# Overridable via ``AMLAS_RESOURCE_PREFIX`` so an adopter can namespace their account.
DEFAULT_RESOURCE_PREFIX = "amlas-"



def resolve_control_region(explicit: str | None = None) -> str:
    """Resolve the same selected region used by every AWS control tool."""
    return resolve_aws_region(explicit)

# ─── The allowlist: service → the exact control actions agents may perform ─────
#
# Deliberately narrow and additive. An action absent here is DENIED, so widening the
# agents' reach is an explicit, reviewable edit to this table (and its tests) — never an
# accident. Names are the AWS API action names (matched case-insensitively).
ALLOWED_ACTIONS: dict[str, frozenset[str]] = {
    # Lake storage: write/read objects and inspect layout. NEVER bucket deletion.
    "s3": frozenset(
        {
            "PutObject",
            "GetObject",
            "ListBucket",
            "HeadBucket",
            "CreateBucket",
            "PutBucketEncryption",
            "PutBucketVersioning",
            "PutBucketPublicAccessBlock",
        }
    ),
    # Catalog: databases/tables/crawlers for the medallion lake.
    "glue": frozenset(
        {
            "CreateDatabase",
            "GetDatabase",
            "CreateTable",
            "GetTable",
            "UpdateTable",
            "CreateCrawler",
            "StartCrawler",
            "GetCrawler",
        }
    ),
    # Permissions: create the job roles and scope grants. NEVER user/group/key actions.
    "iam": frozenset(
        {
            "CreateRole",
            "GetRole",
            "PutRolePolicy",
            "GetRolePolicy",
            "AttachRolePolicy",
            "ListAttachedRolePolicies",
            "SimulatePrincipalPolicy",
            "PassRole",
        }
    ),
    # The compute the leads dispatch to (ADR-0002 data plane) AND the full SageMaker
    # lifecycle they orchestrate — pipelines, experiments, endpoints, monitoring schedules,
    # AutoML, labeling. This is the "SageMaker Studio, but agentic" surface: every verb an
    # ML engineer would click, MINUS every deletion (retiring production capacity stays an
    # operator action, enforced by ``test_no_service_grants_a_delete_or_terminate_verb``).
    "sagemaker": frozenset(
        {
            # Jobs
            "CreateTrainingJob",
            "CreateProcessingJob",
            "CreateTransformJob",
            "CreateHyperParameterTuningJob",
            "CreateAutoMLJob",
            "CreateLabelingJob",
            # Models + registry
            "CreateModel",
            "CreateModelPackage",
            "CreateModelPackageGroup",
            "CreateModelCard",
            "UpdateModelPackage",
            # Pipelines (the DAG layer inside SageMaker)
            "CreatePipeline",
            "UpdatePipeline",
            "StartPipelineExecution",
            "DescribePipeline",
            "DescribePipelineExecution",
            "ListPipelineExecutions",
            # Experiments / trials (the tracking surface Studio shows)
            "CreateExperiment",
            "CreateTrial",
            "CreateTrialComponent",
            "Search",
            # Serving
            "CreateEndpoint",
            "CreateEndpointConfig",
            "UpdateEndpoint",
            "UpdateEndpointWeightsAndCapacities",
            "InvokeEndpoint",
            "DescribeEndpoint",
            # Feature store + monitoring
            "CreateFeatureGroup",
            "CreateMonitoringSchedule",
            "UpdateMonitoringSchedule",
            "DescribeMonitoringSchedule",
            # Reads
            "DescribeTrainingJob",
            "DescribeProcessingJob",
            "ListTrainingJobs",
            "ListModelPackages",
            "ListEndpoints",
        }
    ),
    # Step Functions — cross-service workflow orchestration ABOVE individual jobs (the
    # boto3 service name is ``stepfunctions``; the IAM/API prefix is ``states``, which is
    # what a policy document and this table key on).
    "states": frozenset(
        {
            "CreateStateMachine",
            "UpdateStateMachine",
            "StartExecution",
            "DescribeExecution",
            "DescribeStateMachine",
            "ListExecutions",
            "SendTaskSuccess",
            "SendTaskFailure",
        }
    ),
    # Lake Formation — table/column-level governance for the medallion lake. GRANT is
    # allowed (an agent may open access it needs); REVOKE is not — silently removing
    # someone else's access is exactly the blast radius the guard exists to prevent.
    "lakeformation": frozenset(
        {
            "GrantPermissions",
            "ListPermissions",
            "RegisterResource",
            "DescribeResource",
            "GetDataAccess",
        }
    ),
    # EMR Serverless — Spark-scale ETL for lake work too big for Glue.
    "emr-serverless": frozenset(
        {
            "CreateApplication",
            "StartApplication",
            "StartJobRun",
            "GetJobRun",
            "GetApplication",
            "ListJobRuns",
        }
    ),
    # AWS Batch — containerized batch compute (the ADR-0002 "doesn't fit Lambda/SageMaker"
    # tier), e.g. a long backfill or a bespoke evaluation sweep.
    "batch": frozenset(
        {
            "SubmitJob",
            "DescribeJobs",
            "RegisterJobDefinition",
            "CreateComputeEnvironment",
            "CreateJobQueue",
            "DescribeJobQueues",
        }
    ),
    # ECS / Fargate — containerized tasks (same tier as Batch, task-shaped).
    "ecs": frozenset(
        {
            "RunTask",
            "DescribeTasks",
            "RegisterTaskDefinition",
            "CreateCluster",
            "DescribeClusters",
            "ListTasks",
        }
    ),
    # Kinesis — streaming ingestion for online features and live monitoring signals.
    "kinesis": frozenset(
        {
            "CreateStream",
            "PutRecord",
            "PutRecords",
            "DescribeStream",
            "GetRecords",
            "GetShardIterator",
            "ListStreams",
        }
    ),
    # ECR — container images for custom training/serving containers.
    "ecr": frozenset(
        {
            "CreateRepository",
            "GetAuthorizationToken",
            "PutImage",
            "BatchGetImage",
            "DescribeRepositories",
            "DescribeImages",
        }
    ),
    # KMS — encryption for lake objects + model artifacts. Grants and data-key operations
    # only: key POLICY edits and key deletion are operator-only (a key policy is the last
    # line of defence on the artifacts themselves).
    "kms": frozenset(
        {
            "CreateGrant",
            "ListGrants",
            "GenerateDataKey",
            "Encrypt",
            "Decrypt",
            "DescribeKey",
        }
    ),
    # Quota headroom — the classic autonomy blocker (concurrent sessions, instance caps).
    "servicequotas": frozenset(
        {"GetServiceQuota", "ListServiceQuotas", "RequestServiceQuotaIncrease"}
    ),
    # Scheduling recurring work (research scans, monitor ticks, retrain triggers).
    "events": frozenset({"PutRule", "PutTargets", "DescribeRule", "ListRules"}),
    # Observability the monitoring lead configures for itself.
    "cloudwatch": frozenset({"PutMetricData", "PutMetricAlarm", "DescribeAlarms", "PutDashboard"}),
    "logs": frozenset({"CreateLogGroup", "CreateLogStream", "PutLogEvents", "DescribeLogGroups"}),
    # Query surface over the lake.
    "athena": frozenset({"StartQueryExecution", "GetQueryExecution", "GetQueryResults"}),
}

# Actions that genuinely do not support resource-level scoping, so ``Resource: "*"`` in a
# policy document is correct rather than sloppy. Mirrors the documented exemption already
# taken in ``infra/cdk_stacks/backend_iam_stack.py`` (``sagemaker:ListTrainingJobs``).
RESOURCE_WILDCARD_EXEMPT_ACTIONS: frozenset[str] = frozenset(
    {
        "sagemaker:listtrainingjobs",
        "sagemaker:listprocessingjobs",
        "sagemaker:listmodels",
        "sagemaker:listendpoints",
        "sagemaker:listmodelpackages",
        "sagemaker:listpipelineexecutions",
        "sagemaker:search",
        "servicequotas:listservicequotas",
        "cloudwatch:putmetricdata",
        "ecr:getauthorizationtoken",
        "logs:describeloggroups",
        "athena:listworkgroups",
        "states:listexecutions",
        "batch:describejobqueues",
        "ecs:listtasks",
        "kinesis:liststreams",
        "emr-serverless:listjobruns",
        "lakeformation:listpermissions",
    }
)

# Any resource string containing one of these is rejected before prefix checking: a
# wildcard would let a prefixed name still fan out, and ``..`` is path traversal.
_FORBIDDEN_RESOURCE_TOKENS = ("*", "?", "..")

# IAM action patterns that grant the power to re-grant (privilege escalation).
_ESCALATION_ACTION_RE = re.compile(
    r"^(iam|sts|organizations|kms):", re.IGNORECASE
)
_ESCALATION_ALLOWED = frozenset({"iam:passrole", "sts:assumerole", "iam:getrole"})


@dataclass(frozen=True)
class ResourceControlPolicy:
    """The configured guard (prefix + region pin + allowlist override).

    Args:
        resource_prefix: Mandatory prefix on every resource name the agents touch.
        allowed_regions: Regions requests may target; empty tuple means "any region".
        allowed_actions: Service → action allowlist; defaults to :data:`ALLOWED_ACTIONS`.
    """

    resource_prefix: str = DEFAULT_RESOURCE_PREFIX
    # Derived from :func:`resolve_control_region` (not a hard-coded constant) so a
    # directly-constructed policy pins the SAME region the control tools will call with.
    allowed_regions: tuple[str, ...] = field(default_factory=lambda: tuple(filter(None, (resolve_aws_region(required=False),))))
    allowed_actions: dict[str, frozenset[str]] = field(default_factory=lambda: ALLOWED_ACTIONS)

    @classmethod
    def from_env(cls) -> ResourceControlPolicy:
        """Build the policy from the environment (safe defaults when unset).

        ``AMLAS_RESOURCE_PREFIX`` sets the prefix; otherwise the selected deployment
        prefix plus ``-`` is used (legacy ``amlas`` remains ``amlas-``).
        ``AWS_REGION`` / ``AWS_DEFAULT_REGION`` pin the region. A blank/whitespace env value
        falls back to the default rather than disabling the check.
        """
        prefix = (
            (os.environ.get("AMLAS_RESOURCE_PREFIX") or "").strip()
            or f"{resolve_deployment_prefix()}-"
        )
        return cls(resource_prefix=prefix, allowed_regions=(resolve_control_region(),))


@dataclass(frozen=True)
class ResourceRequest:
    """One proposed AWS control-plane mutation, as the calling tool describes it.

    Args:
        service: AWS service name as boto3 spells it (``"s3"``, ``"iam"``, ``"glue"``).
        action: AWS API action name (``"PutObject"``, ``"CreateRole"``).
        resource_name: The resource this touches — bucket/prefix, role name, database
            name. Must carry the policy's ``resource_prefix``.
        region: Target region.
        policy_document: For IAM grant actions, the policy document to be attached; it
            is linted by :func:`lint_policy_document`.
        detail: Optional human note carried into the plan line.
    """

    service: str
    action: str
    resource_name: str
    region: str = ""
    policy_document: dict[str, Any] | None = None
    detail: str = ""


@dataclass(frozen=True)
class Decision:
    """The guard's verdict for one :class:`ResourceRequest`.

    ``allowed`` is the only thing a caller may branch on. ``plan`` is always populated so
    a ``plan_only`` tool call can render what WOULD happen; ``findings`` carries the IAM
    lint findings when a policy document was rejected.
    """

    allowed: bool
    reason: str
    plan: str
    findings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Render as plain data for a tool result / audit log."""
        return {
            "allowed": self.allowed,
            "reason": self.reason,
            "plan": self.plan,
            "findings": list(self.findings),
        }


def _statements(document: Any) -> list[dict[str, Any]]:
    """Return the statement list from a policy document, tolerating any garbage input."""
    if not isinstance(document, dict):
        return []
    raw = document.get("Statement")
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        return [s for s in raw if isinstance(s, dict)]
    return []


def _as_list(value: Any) -> list[str]:
    """Coerce a policy field (str | list | anything) to a list of strings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return []


def lint_policy_document(document: Any) -> list[str]:
    """Return privilege-escalation findings for an IAM policy document (empty ⇒ clean).

    Never raises — malformed/garbage input simply yields findings or an empty list, so a
    linting failure can never crash a tool call. Rejects:

    * ``Action: "*"`` or a service-wide ``"<svc>:*"`` wildcard.
    * ``Resource: "*"`` unless every action in that statement is in
      :data:`RESOURCE_WILDCARD_EXEMPT_ACTIONS` (actions that cannot be resource-scoped).
    * ``NotAction`` / ``NotResource`` (allowlist inversion — deny-by-default's enemy).
    * IAM/STS/Organizations/KMS actions that would let the grantee re-grant, other than
      the narrowly-needed ``iam:PassRole`` / ``sts:AssumeRole`` / ``iam:GetRole``.
    """
    findings: list[str] = []
    for stmt in _statements(document):
        if str(stmt.get("Effect", "Allow")).lower() != "allow":
            continue  # Deny statements only ever narrow access.

        for inverted in ("NotAction", "NotResource"):
            if stmt.get(inverted):
                findings.append(
                    f"{inverted} is not permitted — it inverts the allowlist and defeats "
                    "deny-by-default; enumerate the exact actions/resources instead."
                )

        actions = [a.strip() for a in _as_list(stmt.get("Action")) if a.strip()]
        resources = [r.strip() for r in _as_list(stmt.get("Resource")) if r.strip()]

        for action in actions:
            low = action.lower()
            if low == "*" or low.endswith(":*"):
                findings.append(
                    f"Action {action!r} is a wildcard — enumerate the exact API actions."
                )
                continue
            if _ESCALATION_ACTION_RE.match(low) and low not in _ESCALATION_ALLOWED:
                findings.append(
                    f"Action {action!r} can re-grant permissions (privilege escalation) — "
                    "only iam:PassRole / sts:AssumeRole / iam:GetRole are permitted."
                )

        if any(r == "*" for r in resources):
            exempt = bool(actions) and all(
                a.lower() in RESOURCE_WILDCARD_EXEMPT_ACTIONS for a in actions
            )
            if not exempt:
                findings.append(
                    "Resource '*' is not permitted for "
                    f"{actions or ['<no action>']} — scope the Resource to an ARN "
                    "(only actions that cannot be resource-scoped may use '*')."
                )
    return findings


def _plan_line(request: ResourceRequest) -> str:
    """Build the human one-liner describing what this request would do."""
    line = f"{request.service}:{request.action} on {request.resource_name} in {request.region}"
    if request.detail:
        line = f"{line} — {request.detail}"
    return line


def authorize(
    request: ResourceRequest, *, policy: ResourceControlPolicy | None = None
) -> Decision:
    """Authorize one AWS control-plane mutation (deny-by-default at every check).

    Runs the five checks in cheapest-first order (service → action → resource → region →
    IAM lint) and returns on the FIRST failure, so ``reason`` names the single thing that
    blocked the request. A caller must treat ``allowed=False`` as a hard stop.

    Args:
        request: The proposed mutation.
        policy: The guard configuration; defaults to :meth:`ResourceControlPolicy.from_env`.

    Returns:
        A :class:`Decision`; ``plan`` is populated either way.
    """
    policy = policy or ResourceControlPolicy.from_env()
    plan = _plan_line(request)

    service = (request.service or "").strip().lower()
    action = (request.action or "").strip()
    resource = (request.resource_name or "").strip()

    # 1. Service allowlist.
    allowed_for_service = policy.allowed_actions.get(service)
    if not allowed_for_service:
        return Decision(
            allowed=False,
            reason=(
                f"service {service!r} is not in the allowlist "
                f"({', '.join(sorted(policy.allowed_actions))})"
            ),
            plan=plan,
        )

    # 2. Per-service action allowlist (case-insensitive).
    if action.lower() not in {a.lower() for a in allowed_for_service}:
        return Decision(
            allowed=False,
            reason=f"action {service}:{action} is not an allowed action for {service}",
            plan=plan,
        )

    # 3. Resource name: non-empty, no wildcard/traversal, carries the mandatory prefix.
    if not resource:
        return Decision(allowed=False, reason="resource_name is required", plan=plan)
    for token in _FORBIDDEN_RESOURCE_TOKENS:
        if token in resource:
            return Decision(
                allowed=False,
                reason=f"resource {resource!r} contains the forbidden token {token!r}",
                plan=plan,
            )
    if not resource.startswith(policy.resource_prefix):
        return Decision(
            allowed=False,
            reason=(
                f"resource {resource!r} must start with the platform prefix "
                f"{policy.resource_prefix!r} — agents may only touch resources amlas owns"
            ),
            plan=plan,
        )

    # 4. Region pin (empty allowlist ⇒ any region).
    region = (request.region or "").strip()
    if policy.allowed_regions and region not in policy.allowed_regions:
        return Decision(
            allowed=False,
            reason=(
                f"region {region!r} is outside the pinned regions "
                f"({', '.join(policy.allowed_regions)})"
            ),
            plan=plan,
        )

    # 5. IAM policy-document lint (only when a document is attached).
    if request.policy_document is not None:
        findings = lint_policy_document(request.policy_document)
        if findings:
            return Decision(
                allowed=False,
                reason=f"policy document rejected by the linter ({len(findings)} finding(s))",
                plan=plan,
                findings=findings,
            )

    return Decision(allowed=True, reason=f"permitted: {plan}", plan=plan)
