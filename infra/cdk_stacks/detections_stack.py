"""DetectionsStack (NEW, R-011 / AMLAS-SEC-012) — CloudTrail-based detections as IaC.

Codifies the previously narrative-only reliance on account-level CloudTrail into
this service's IaC, satisfying the **"CloudTrail-based Detections"** BSC line-item.

Resources:
    - aws_s3.Bucket — encrypted, SSL-enforced, public-access-blocked trail log store.
    - aws_cloudtrail.Trail — multi-region management-event trail delivering to a
      CloudWatch Logs group (so metric filters can fire alarms in near-real-time).
    - >= 3 metric-filter + alarm pairs on security-relevant events:
        1. Secrets Manager GetSecretValue anomalies (credential-access monitoring).
        2. IAM mutating changes (Put/Attach/Create/Delete policy + role/user).
        3. SageMaker CreateTrainingJob (unexpected training-job creation).
      (A 4th alarm covers AgentCore UpdateAgentRuntime config drift.)
    - aws_sns.Topic — alarm notification target (subscriptions wired out-of-band).

SEC-01: NO hardcoded account IDs — bucket name is account/region-tokenized.
"""

from __future__ import annotations

from typing import Any

from aws_cdk import (
    CfnOutput,
    RemovalPolicy,
    Stack,
)
from aws_cdk import (
    aws_cloudtrail as cloudtrail,
)
from aws_cdk import (
    aws_cloudwatch as cloudwatch,
)
from aws_cdk import (
    aws_cloudwatch_actions as cloudwatch_actions,
)
from aws_cdk import (
    aws_logs as logs,
)
from aws_cdk import (
    aws_s3 as s3,
)
from aws_cdk import (
    aws_sns as sns,
)
from constructs import Construct

from ._kms import make_data_key
from ._s3 import make_access_log_bucket
from ._tags import apply_cost_tags

# (metric_name, filter_pattern, alarm_description) for each detection.
# Patterns use CloudWatch Logs filter syntax over the CloudTrail JSON event.
_DETECTIONS: list[tuple[str, str, str]] = [
    (
        "SecretsManagerGetSecretValue",
        '{ ($.eventSource = "secretsmanager.amazonaws.com") && ($.eventName = "GetSecretValue") }',
        "Secrets Manager GetSecretValue access — credential-access monitoring (R-011).",
    ),
    (
        "IamMutatingChange",
        (  # nosemgrep
            '{ ($.eventSource = "iam.amazonaws.com") && '
            '(($.eventName = "PutRolePolicy") || ($.eventName = "AttachRolePolicy") || '
            '($.eventName = "CreatePolicy") || ($.eventName = "DeleteRolePolicy") || '
            '($.eventName = "CreateRole") || ($.eventName = "DeleteRole") || '
            '($.eventName = "CreateUser") || ($.eventName = "CreateAccessKey")) }'
        ),
        "IAM mutating change — privilege-escalation / persistence monitoring (R-011).",
    ),
    (
        "SageMakerCreateTrainingJob",
        '{ ($.eventSource = "sagemaker.amazonaws.com") && ($.eventName = "CreateTrainingJob") }',
        "SageMaker CreateTrainingJob — unexpected training-job creation monitoring (R-011).",
    ),
    (
        "AgentCoreUpdateRuntime",
        (  # nosemgrep
            '{ ($.eventSource = "bedrock-agentcore-control.amazonaws.com") && '
            '($.eventName = "UpdateAgentRuntime") }'
        ),
        "AgentCore UpdateAgentRuntime — runtime config-drift monitoring (R-011).",
    ),
]

_METRIC_NAMESPACE = "Amlas/Security"


class DetectionsStack(Stack):
    def __init__(
        self,
        scope: Construct,
        cid: str,
        *,
        hyphen_project: str = "amlas",
        kms_key_arn: str = "",
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, cid, **kwargs)
        self.hyphen_project = hyphen_project
        # WA-12: cost-allocation tags (project/env/owner) for chargeback.
        apply_cost_tags(self)

        # WA-01: CMK (rotation on) for the CloudTrail log store, the trail log group,
        # and the alarm SNS topic. ``for_logs=True`` grants the Logs service principal.
        self.data_key = make_data_key(
            self,
            "DetectionsDataKey",
            key_arn=kms_key_arn,
            alias=f"alias/{hyphen_project}-detections",
            for_logs=True,
            description="Amlas detections CMK (rotation on) — trail/logs/SNS at rest",
        )

        self.access_log_bucket = make_access_log_bucket(self)
        self.trail_bucket = self._build_trail_bucket()
        self.trail_log_group = self._build_trail_log_group()
        self.trail = self._build_trail()
        self.alarm_topic = self._build_alarm_topic()
        self.alarms = self._build_detection_alarms()

        self._emit_outputs()

    def _build_trail_bucket(self) -> s3.Bucket:
        return s3.Bucket(
            self,
            "TrailBucket",
            bucket_name=f"{self.hyphen_project}-cloudtrail-{self.account}-{self.region}",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.data_key,
            bucket_key_enabled=True,
            enforce_ssl=True,
            versioned=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            server_access_logs_bucket=self.access_log_bucket,
            server_access_logs_prefix="cloudtrail-bucket/",
            removal_policy=RemovalPolicy.RETAIN,
        )

    def _build_trail_log_group(self) -> logs.LogGroup:
        return logs.LogGroup(
            self,
            "TrailLogGroup",
            log_group_name=f"/aws/cloudtrail/{self.hyphen_project}-security",
            retention=logs.RetentionDays.THREE_MONTHS,
            encryption_key=self.data_key,  # WA-01: CMK-encrypt the trail log group
            removal_policy=RemovalPolicy.DESTROY,
        )

    def _build_trail(self) -> cloudtrail.Trail:
        return cloudtrail.Trail(
            self,
            "SecurityTrail",
            trail_name=f"{self.hyphen_project}-security-trail",
            bucket=self.trail_bucket,
            cloud_watch_log_group=self.trail_log_group,
            send_to_cloud_watch_logs=True,
            is_multi_region_trail=True,
            include_global_service_events=True,
            enable_file_validation=True,
        )

    def _build_alarm_topic(self) -> sns.Topic:
        return sns.Topic(
            self,
            "SecurityAlarmTopic",
            topic_name=f"{self.hyphen_project}-security-alarms",
            display_name="Amlas agents — security detection alarms",
            master_key=self.data_key,  # WA-01: CMK-encrypt the alarm topic
        )

    def _build_detection_alarms(self) -> list[cloudwatch.Alarm]:
        alarms: list[cloudwatch.Alarm] = []
        for metric_name, pattern, description in _DETECTIONS:
            mf = logs.MetricFilter(
                self,
                f"MetricFilter-{metric_name}",
                log_group=self.trail_log_group,
                metric_namespace=(
                    _METRIC_NAMESPACE if self.hyphen_project == "amlas" else f"{self.hyphen_project}/Security"
                ),
                metric_name=metric_name,
                filter_pattern=logs.FilterPattern.literal(pattern),
                metric_value="1",
                default_value=0,
            )
            metric = mf.metric(statistic="Sum")
            alarm = cloudwatch.Alarm(
                self,
                f"Alarm-{metric_name}",
                alarm_name=f"{self.hyphen_project}-{metric_name}",
                alarm_description=description,
                metric=metric,
                threshold=1,
                evaluation_periods=1,
                comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
                treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
            )
            alarm.add_alarm_action(cloudwatch_actions.SnsAction(self.alarm_topic))
            alarms.append(alarm)
        return alarms

    def _emit_outputs(self) -> None:
        CfnOutput(
            self,
            "SecurityTrailName",
            value=self.trail.trail_arn,
            export_name=f"{self.hyphen_project}-security-trail-arn",
        )
        CfnOutput(
            self,
            "SecurityAlarmCount",
            value=str(len(self.alarms)),
            export_name=f"{self.hyphen_project}-security-alarm-count",
        )
        CfnOutput(
            self,
            "SecurityAlarmTopicArn",
            value=self.alarm_topic.topic_arn,
            export_name=f"{self.hyphen_project}-security-alarm-topic-arn",
        )
