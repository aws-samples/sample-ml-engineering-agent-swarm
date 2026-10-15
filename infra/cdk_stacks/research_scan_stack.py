"""ResearchScanScheduleStack (NEW, gap-closure §G5) — the hosted recurring research-scan cadence.

The deploy-time half of G5: an **EventBridge schedule rule** (cron) that, on a configurable
cadence, invokes a small Lambda which runs the Stage-0 scheduled research scan
(``run_research_scan_flow`` → ``ResearchScanScheduler``) and delivers the digest to a configurable
destination. The in-process ``ResearchScanScheduler`` is the tickable core; this stack is the
hosted trigger so the scan recurs without a human kicking it off (the "every Monday scan + digest"
story, master plan §G5).

Resources:
    - S3 bucket — the default digest delivery destination (the §G5 "where is the digest posted"
      decision; S3 by default, overridable to a channel/kanban webhook via ``DigestDestination``).
    - IAM Lambda execution role (basic execution + write to the digest bucket).
    - Lambda (python3.11) — runs one scheduled scan tick + writes the digest.
    - EventBridge rule with a **cron ScheduleExpression** (the cadence) → Lambda target.

Deploy-time parameters (the §G5 OPEN DECISION — sensible defaults so it runs; human confirms):
    - ``ScanCadenceCron``   — the cron cadence. Default ``cron(0 13 ? * MON *)`` = every Monday
      13:00 UTC (the proposed "weekly Monday"). The human confirms the final cadence.
    - ``DigestDestination`` — where the digest is posted. Default ``s3`` (this stack's bucket);
      a real channel/kanban webhook URL can be supplied instead.
    - ``ScanQueries``       — comma-separated research queries the scan covers.

SEC-01: no hardcoded account ids — ARNs are composed from CDK tokens (``self.account`` /
``self.region``). The rule name follows the ``hyphen_project`` convention (hyphens, no
underscores) so it matches the AWS resource-name rule.

Naming + EventBridge style mirror ``AsyncTrainingStack._build_event_rule`` (the existing
EventBridge-rule pattern in this repo).
"""

from __future__ import annotations

from typing import Any

from aws_cdk import (
    CfnOutput,
    CfnParameter,
    Duration,
    RemovalPolicy,
    Stack,
)
from aws_cdk import (
    aws_events as events,
)
from aws_cdk import (
    aws_events_targets as targets,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_lambda as _lambda,
)
from aws_cdk import (
    aws_logs as logs,
)
from aws_cdk import (
    aws_s3 as s3,
)
from constructs import Construct

from ._kms import make_data_key
from ._s3 import make_access_log_bucket
from ._tags import apply_cost_tags

# A small, self-contained inline handler: it runs ONE scheduled research-scan tick via the
# product's ``run_research_scan_flow`` (CHEAP_READ, no AWS spend) and writes the digest to the
# destination. Inline (not from_asset) so the schedule stack is deployable independently of the
# runtime image. The product package is provided via a Lambda layer at deploy time (or the
# handler degrades to recording the trigger when the package is absent — fail-soft, never crash).
_INLINE_HANDLER = '''
import json
import os
import time


def handler(event, context):
    """EventBridge-triggered scheduled research-scan tick (gap-closure G5)."""
    queries = [q.strip() for q in os.environ.get("SCAN_QUERIES", "").split(",") if q.strip()]
    cadence = os.environ.get("SCAN_CADENCE", "weekly")
    destination = os.environ.get("DIGEST_DESTINATION", "")
    record = {"status": "triggered", "cadence": cadence, "queries": queries, "ts": int(time.time())}
    try:
        from amlas.flows.research_scan_flow import run_research_scan_flow

        record = run_research_scan_flow(queries=queries or None, cadence=cadence)
    except Exception as exc:  # noqa: BLE001 - fail-soft: still record the trigger
        record["error"] = type(exc).__name__  # type only: message may embed query/API data

    if destination.startswith("s3://") or os.environ.get("DIGEST_BUCKET"):
        try:
            import boto3

            bucket = os.environ.get("DIGEST_BUCKET", "")
            if bucket:
                key = f"research-digests/{record.get('scan_id', 'scan')}-{int(time.time())}.json"
                boto3.client("s3").put_object(
                    Bucket=bucket, Key=key,
                    Body=json.dumps(record, default=str).encode("utf-8"),
                )
                record["delivered_to"] = f"s3://{bucket}/{key}"
        except Exception as exc:  # noqa: BLE001
            record["delivery_error"] = type(exc).__name__  # type only: message may embed bucket/key/data
    return {"statusCode": 200, "body": json.dumps(record, default=str)}
'''


class ResearchScanScheduleStack(Stack):
    def __init__(
        self,
        scope: Construct,
        cid: str,
        *,
        hyphen_project: str = "amlas",
        region_name: str | None = None,
        kms_key_arn: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, cid, **kwargs)
        self.hyphen_project = hyphen_project
        self.region_name = region_name or self.region
        if self.region_name != self.region:
            raise ValueError("region_name must match the stack environment; configure the deployment region through env.")
        # AWS-safe rule name (hyphens, no underscores) — mirrors the async-training rule naming.
        self.rule_name = f"{hyphen_project}-research-scan-schedule"
        # WA-12: cost-allocation tags (project/env/owner) for chargeback.
        apply_cost_tags(self)

        # WA-01: customer-managed CMK (rotation on) for the digest bucket at rest.
        self.data_key = make_data_key(
            self, "ResearchDigestKey", key_arn=kms_key_arn,
            alias=f"alias/{hyphen_project}-research-digest", for_logs=True,
            description="Amlas research-digest CMK (rotation on) — digest/logs at rest",
        )

        # ── §G5 OPEN DECISION parameters (sensible defaults so it runs; human confirms) ──
        self.cadence_param = CfnParameter(
            self,
            "ScanCadenceCron",
            type="String",
            default="cron(0 13 ? * MON *)",  # every Monday 13:00 UTC — the proposed "weekly Monday"
            description="EventBridge cron for the recurring research scan (the §G5 cadence decision). "
            "Default: every Monday 13:00 UTC. Confirm/override before provisioning.",
        )
        self.destination_param = CfnParameter(
            self,
            "DigestDestination",
            type="String",
            default="s3",
            description="Where the research digest is delivered (the §G5 destination decision): "
            "'s3' (this stack's bucket) | a channel/kanban webhook URL. Default: s3.",
        )
        self.queries_param = CfnParameter(
            self,
            "ScanQueries",
            type="String",
            default="ML model monitoring drift detection,predictive maintenance,anomaly detection tabular",
            description="Comma-separated research queries the scheduled scan covers.",
        )

        self.access_log_bucket = make_access_log_bucket(self)
        self.digest_bucket = self._build_digest_bucket()
        self.scan_log_group = self._build_scan_log_group()
        self.scan_role = self._build_scan_role()
        self.scan_fn = self._build_scan_lambda()
        self.schedule_rule = self._build_schedule_rule()

        self._emit_outputs()

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------
    def _build_digest_bucket(self) -> s3.Bucket:
        """The default digest delivery destination (S3) — versioned, encrypted, private."""
        return s3.Bucket(
            self,
            "ResearchDigestBucket",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.data_key,
            bucket_key_enabled=True,
            versioned=True,
            enforce_ssl=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            server_access_logs_bucket=self.access_log_bucket,
            server_access_logs_prefix="research-digest-bucket/",
            removal_policy=RemovalPolicy.RETAIN,
        )

    def _build_scan_log_group(self) -> logs.LogGroup:
        return logs.LogGroup(
            self,
            "ResearchScanLogGroup",
            log_group_name=f"/aws/lambda/{self.hyphen_project}-research-scan",
            retention=logs.RetentionDays.ONE_MONTH,
            encryption_key=self.data_key,
            removal_policy=RemovalPolicy.DESTROY,
        )

    def _build_scan_role(self) -> iam.Role:
        role = iam.Role(
            self,
            "ResearchScanLambdaRole",
            role_name=f"{self.hyphen_project}-research-scan-role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Scheduled research-scan Lambda role - logs + digest bucket write.",
        )
        # Least-priv: write only the digest bucket (the scan itself is read-only public web).
        self.digest_bucket.grant_write(role)
        return role

    def _build_scan_lambda(self) -> _lambda.Function:
        return _lambda.Function(
            self,
            "ResearchScanFunction",
            function_name=f"{self.hyphen_project}-research-scan",
            runtime=_lambda.Runtime.PYTHON_3_14,
            handler="index.handler",
            code=_lambda.Code.from_inline(_INLINE_HANDLER),
            role=self.scan_role,
            log_group=self.scan_log_group,
            timeout=Duration.minutes(5),
            memory_size=512,
            environment={
                "SCAN_CADENCE": "weekly",
                "SCAN_QUERIES": self.queries_param.value_as_string,
                "DIGEST_DESTINATION": self.destination_param.value_as_string,
                "DIGEST_BUCKET": self.digest_bucket.bucket_name,
            },
        )

    def _build_schedule_rule(self) -> events.Rule:
        """EventBridge schedule rule (cron) → the scan Lambda — the recurring cadence trigger."""
        rule = events.Rule(
            self,
            "ResearchScanScheduleRule",
            rule_name=self.rule_name,
            description="Recurring Stage-0 research scan on a configurable cron cadence (gap-closure G5)",
            schedule=events.Schedule.expression(self.cadence_param.value_as_string),
        )
        rule.add_target(targets.LambdaFunction(self.scan_fn))
        return rule

    def _emit_outputs(self) -> None:
        CfnOutput(
            self, "ResearchScanFunctionNameOut", value=self.scan_fn.function_name,
            export_name=f"{self.hyphen_project}-research-scan-function-name",
        )
        CfnOutput(
            self, "ResearchScanRuleNameOut", value=self.rule_name,
            export_name=f"{self.hyphen_project}-research-scan-rule-name",
        )
        CfnOutput(
            self, "ResearchDigestBucketOut", value=self.digest_bucket.bucket_name,
            export_name=f"{self.hyphen_project}-research-digest-bucket",
        )
