"""``dispatch_preflight`` — the shift-left capability probe for off-box dispatch (ADR-0002 Phase 6).

FIX-THE-FRAMEWORK-NOT-THE-JOB: rather than let the swarm dispatch a SageMaker job straight into an
IAM/KMS/bucket/image misconfiguration (the FIX-31…34b class of blockers), this tool validates the
END-TO-END DISPATCH CAPABILITY *first* and returns a STRUCTURED, ACTIONABLE report. The agent calls
it before ``submit_tabular_training`` (or the tool can call it internally) and, on a gap, gets a
precise remediation — which IaC statement/role/grant is missing — instead of a post-hoc failure.

Every check is READ-ONLY (describe / simulate-principal-policy / head) — no resource is created or
mutated. Clients are injectable so this is directly verifiable offline with fakes. Absent creds / probe
errors degrade to ``status="unknown"`` for that check (never a false "ok").

Checks:
  * session_bucket   — resolvable (config/env) and writable (head-bucket / list).
  * exec_role        — the runtime exec-role exists and can be introspected.
  * pass_role        — exec-role has ``iam:PassRole`` on the SageMaker job role (simulate).
  * kms_jobs_table   — exec-role can ``kms:GenerateDataKey`` on the jobs-table CMK via dynamodb.
  * image            — the managed training image exists in ECR (describe-images).

The aggregate ``ready`` is True only when no check is a hard FAIL (``unknown`` does not block —
it is surfaced but the agent may proceed, since a missing probe permission is not a missing
capability).
"""

from __future__ import annotations

import os
from typing import Any

from strands import tool

from amlas.config._env_compat import env_with_compat

from ..config.aws_region import resolve_aws_region
from ..errors import tool_error_boundary


def _resolve_region() -> str:
    return resolve_aws_region()


def _resolve_session_bucket() -> str:
    return (
        env_with_compat("AMLAS_SESSION_BUCKET")
        or os.environ.get("SESSION_BUCKET")
        or ""
    )


def _ok(detail: str, **extra: Any) -> dict[str, Any]:
    return {"status": "ok", "detail": detail, **extra}


def _fail(detail: str, remediation: str, **extra: Any) -> dict[str, Any]:
    return {"status": "fail", "detail": detail, "remediation": remediation, **extra}


def _unknown(detail: str, **extra: Any) -> dict[str, Any]:
    return {"status": "unknown", "detail": detail, **extra}


def check_session_bucket(bucket: str, *, region: str, s3_client: Any = None) -> dict[str, Any]:
    if not bucket:
        return _fail(
            "no session bucket resolvable",
            "set AMLAS_SESSION_BUCKET in the runtime env (deploy: deploy_agentcore_runtime resolves it from the "
            "AsyncTrainingStack export).",
        )
    try:
        import boto3

        s3 = s3_client or boto3.client("s3", region_name=resolve_aws_region(region))
        s3.head_bucket(Bucket=bucket)
        return _ok(f"session bucket {bucket} reachable", bucket=bucket)
    except Exception as exc:
        return _unknown(f"could not head-bucket {bucket}: {type(exc).__name__}", bucket=bucket)


def check_pass_role(
    *, exec_role_arn: str, job_role_arn: str, region: str, iam_client: Any = None
) -> dict[str, Any]:
    if not exec_role_arn or not job_role_arn:
        return _unknown("exec/job role ARN unresolved — cannot simulate PassRole")
    try:
        import boto3

        iam = iam_client or boto3.client("iam", region_name=resolve_aws_region(region))
        resp = iam.simulate_principal_policy(
            PolicySourceArn=exec_role_arn,
            ActionNames=["iam:PassRole"],
            ResourceArns=[job_role_arn],
            ContextEntries=[
                {
                    "ContextKeyName": "iam:PassedToService",
                    "ContextKeyValues": ["sagemaker.amazonaws.com"],
                    "ContextKeyType": "string",
                }
            ],
        )
        results = resp.get("EvaluationResults", [])
        decision = results[0].get("EvalDecision") if results else "unknown"
        if decision == "allowed":
            return _ok(f"exec-role may PassRole {job_role_arn}")
        return _fail(
            f"iam:PassRole on {job_role_arn} is {decision}",
            "grant the runtime exec-role iam:PassRole on the SageMaker job role scoped to that "
            "exact role (deploy: deploy_agentcore_runtime derives the PassRole grant FROM the passed-role env, "
            "FIX-33). Ensure SAGEMAKER_EXECUTION_ROLE_ARN == the granted role.",
        )
    except Exception as exc:
        return _unknown(f"PassRole simulate failed: {type(exc).__name__}")


def check_kms_jobs_table(
    *, exec_role_arn: str, region: str, jobs_table: str, iam_client: Any = None, ddb_client: Any = None
) -> dict[str, Any]:
    """Resolve the jobs-table CMK, then simulate kms:GenerateDataKey on it via dynamodb."""
    try:
        import boto3

        ddb = ddb_client or boto3.client("dynamodb", region_name=resolve_aws_region(region))
        desc = ddb.describe_table(TableName=jobs_table)
        cmk = (desc.get("Table", {}).get("SSEDescription", {}) or {}).get("KMSMasterKeyArn", "")
    except Exception as exc:
        return _unknown(f"could not resolve jobs-table CMK: {type(exc).__name__}")
    if not cmk:
        return _ok(f"jobs table {jobs_table} uses SSE-S3/AWS-owned key — no caller KMS grant needed")
    if not exec_role_arn:
        return _unknown("exec-role ARN unresolved — cannot simulate KMS on the CMK")
    try:
        import boto3

        iam = iam_client or boto3.client("iam", region_name=resolve_aws_region(region))
        resp = iam.simulate_principal_policy(
            PolicySourceArn=exec_role_arn,
            ActionNames=["kms:GenerateDataKey", "kms:Decrypt"],
            ResourceArns=[cmk],
            ContextEntries=[
                {
                    "ContextKeyName": "kms:ViaService",
                    "ContextKeyValues": [f"dynamodb.{region}.amazonaws.com"],
                    "ContextKeyType": "string",
                }
            ],
        )
        decisions = {r["EvalActionName"]: r["EvalDecision"] for r in resp.get("EvaluationResults", [])}
        if all(d == "allowed" for d in decisions.values()) and decisions:
            return _ok("exec-role may use the jobs-table CMK via dynamodb", cmk=cmk)
        return _fail(
            f"KMS on jobs-table CMK not fully allowed: {decisions}",
            "grant the exec-role kms:Decrypt/GenerateDataKey on the jobs-table CMK scoped to "
            "kms:ViaService dynamodb.<region>.amazonaws.com (deploy: deploy_agentcore_runtime SseKmsForDdbJobs, "
            "FIX-31).",
            cmk=cmk,
        )
    except Exception as exc:
        return _unknown(f"KMS simulate failed: {type(exc).__name__}", cmk=cmk)


def check_staged_test_alignment(
    *, test_path: str = "", y_true_path: str = ""
) -> dict[str, Any]:
    """Is the test split about to be dispatched still 1:1 with the staged ``y_true``?

    This is the LAST moment the mismatch is cheap. ``y_true`` is materialised once at stage time
    and never regenerated, so an agent step that rewrites ``test.csv`` — which the FE step does,
    in place — desynchronises the pair. Nothing downstream can tell: the training job faithfully
    predicts every row of the file it receives, so the failure surfaces minutes later at grading as
    a shape mismatch attributed to the entrypoint.

    Paths come from the caller, else from the env hints the staging step publishes. Unresolvable
    paths give ``unknown``, never ``ok`` — "could not check" must not read as "checked and fine".
    """
    test_path = test_path or os.environ.get("AMLAS_STAGED_TEST_PATH", "")
    y_true_path = y_true_path or os.environ.get("AMLAS_STAGED_Y_TRUE_PATH", "")
    if not (test_path and y_true_path):
        return _unknown(
            "staged test/y_true paths unknown — set AMLAS_STAGED_TEST_PATH and "
            "AMLAS_STAGED_Y_TRUE_PATH (the entrypoint publishes them after staging), or pass them"
        )

    from ..cloud.staging import verify_staged_test_split

    res = verify_staged_test_split(test_path, y_true_path)
    if res["ok"] is None:
        return _unknown(res["detail"], test_path=test_path, y_true_path=y_true_path)
    if res["ok"]:
        return _ok(res["detail"], n_test_rows=res["n_test_rows"], n_y_true=res["n_y_true"])
    return _fail(
        res["detail"],
        "Do NOT dispatch. Either restore the staged test split, or write engineered features to "
        "NEW paths and dispatch those — the staged pair must stay 1:1 with y_true.",
        n_test_rows=res["n_test_rows"],
        n_y_true=res["n_y_true"],
    )


def check_image(image_uri: str, *, region: str, ecr_client: Any = None) -> dict[str, Any]:
    if not image_uri:
        return _unknown("no image URI supplied")
    # image_uri: <acct>.dkr.ecr.<region>.amazonaws.com/<repo>:<tag>
    try:
        registry = image_uri.split(".dkr.ecr.")[0]
        rest = image_uri.split(".amazonaws.com/", 1)[1]
        repo, _, tag = rest.partition(":")
    except (IndexError, ValueError):
        return _unknown(f"could not parse image URI {image_uri}")
    try:
        import boto3

        ecr = ecr_client or boto3.client("ecr", region_name=resolve_aws_region(region))
        ecr.describe_images(
            registryId=registry, repositoryName=repo, imageIds=[{"imageTag": tag or "latest"}]
        )
        return _ok(f"training image present: {image_uri}", image=image_uri)
    except Exception as exc:
        return _unknown(f"could not describe image {image_uri}: {type(exc).__name__}", image=image_uri)


@tool
@tool_error_boundary(source="tools.dispatch_preflight")
def dispatch_preflight(
    job_role_arn: str = "",
    image_uri: str = "",
    jobs_table: str | None = None,
    exec_role_arn: str = "",
) -> dict[str, Any]:
    """Probe the end-to-end SageMaker DISPATCH capability BEFORE dispatching (ADR-0002 Phase 6).

    Read-only. Returns ``{ready: bool, checks: {name: {status, detail, [remediation]}}, summary}``
    so the agent can (a) proceed with confidence when ready, or (b) surface the EXACT infra gap +
    remediation instead of dispatching into a failure or falling back to an in-sandbox fit. A
    ``fail`` in any check sets ``ready=False``; ``unknown`` (a missing PROBE permission, not a
    missing capability) is surfaced but does not block.

    Args:
        job_role_arn: the SageMaker execution/job role the training job will run as (the role the
            runtime PASSES). Defaults to the ``SAGEMAKER_EXECUTION_ROLE_ARN`` env.
        image_uri: the managed training image to verify exists (optional).
        jobs_table: the DynamoDB job-state table (default ``amlas-training-jobs``).
        exec_role_arn: the runtime exec-role doing the dispatch. Defaults to the
            ``AMLAS_RUNTIME_EXEC_ROLE_ARN`` env (best-effort).
    """
    from .sagemaker import resolve_jobs_table

    jobs_table = resolve_jobs_table(jobs_table)
    region = _resolve_region()
    bucket = _resolve_session_bucket()
    job_role_arn = job_role_arn or os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN") or ""
    exec_role_arn = exec_role_arn or os.environ.get("AMLAS_RUNTIME_EXEC_ROLE_ARN") or ""

    checks = {
        "session_bucket": check_session_bucket(bucket, region=region),
        "pass_role": check_pass_role(
            exec_role_arn=exec_role_arn, job_role_arn=job_role_arn, region=region
        ),
        "kms_jobs_table": check_kms_jobs_table(
            exec_role_arn=exec_role_arn, region=region, jobs_table=jobs_table
        ),
        # A data-contract check among the infra ones on purpose: this is the last point before the
        # job is submitted where a rewritten test split is still cheap to catch.
        "staged_test_alignment": check_staged_test_alignment(),
    }
    if image_uri:
        checks["image"] = check_image(image_uri, region=region)

    fails = [name for name, c in checks.items() if c["status"] == "fail"]
    ready = not fails
    if ready:
        summary = "dispatch capability OK — safe to submit the SageMaker job."
    else:
        summary = "DISPATCH BLOCKED: " + "; ".join(
            f"{n}: {checks[n]['remediation']}" for n in fails
        )
    return {"ready": ready, "checks": checks, "summary": summary, "region": region}
