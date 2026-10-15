"""AsyncTrainingStack — async-training plane, deployable without a Runtime image.

Carved out of RuntimeStack so the SageMaker integration validation can run before
`agentcore launch` builds the agent runtime container. Resources:

    - S3 session bucket (training inputs + artifacts)
    - DynamoDB training-jobs table (GSI on runtime_session_id)
    - IAM SageMaker execution role
    - IAM Lambda callback role
    - Lambda callback (arm64, python3.14) — reads JOBS_TABLE + AGENTCORE_RUNTIME_ARN at runtime
    - EventBridge rule on SageMaker state-change → Lambda

`AGENTCORE_RUNTIME_ARN` is a CfnParameter that defaults to "" so the stack deploys
cleanly without the parent stack present. The Lambda's `invoke_agent_runtime` call
will no-op (or fail soft) until that parameter is updated to a real ARN — but the
DDB-status-update half of the callback runs first, which is what the integration
test verifies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aws_cdk import (
    CfnCondition,
    CfnOutput,
    CfnParameter,
    Duration,
    Fn,
    RemovalPolicy,
    Stack,
)
from aws_cdk import (
    aws_dynamodb as dynamodb,
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

from ._iam import (
    acknowledge_iam5_resources,
    cfn_getatt_arn,
    grant_bucket_read_write,
    grant_table_read_write_data,
)
from ._kms import make_data_key
from ._s3 import make_access_log_bucket
from ._tags import apply_cost_tags

# Lambda code asset root. Resolved from ``__file__`` (repo root = parents[2] of
# ``infra/cdk_stacks/<stack>.py``) rather than left as the bare relative ``"src"``,
# which CDK resolves against the PROCESS CWD — so a synth from anywhere but the repo
# root raised «CannotFindAsset». Content-addressed asset hashing means the resolved
# directory is identical to what the old CWD-relative path produced from the repo
# root, so deploys are unaffected.
_SRC_ASSET = str(Path(__file__).resolve().parents[2] / "src")


class AsyncTrainingStack(Stack):
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
        # WA-12: cost-allocation tags (project/env/owner) for chargeback.
        apply_cost_tags(self)

        # WA-01: customer-managed CMK (rotation on) for the session bucket + jobs table.
        self.data_key = make_data_key(
            self, "AsyncDataKey", key_arn=kms_key_arn,
            alias=f"alias/{hyphen_project}-async-training", for_logs=True,
            description="Amlas async-training CMK (rotation on) — session/jobs/logs at rest",
        )

        self.runtime_arn_param = CfnParameter(
            self,
            "AgentRuntimeArn",
            type="String",
            default="",
            description="ARN of the AgentCore Runtime to resume on training callback. "
            "Leave blank if Runtime isn't deployed yet — Lambda will still update DDB.",
        )

        self.access_log_bucket = make_access_log_bucket(self)
        self.session_bucket = self._build_session_bucket()
        self.jobs_table = self._build_jobs_table()
        self.sagemaker_role = self._build_sagemaker_role()
        self.callback_log_group = self._build_callback_log_group()
        self.lambda_role = self._build_lambda_role()
        self.callback_fn = self._build_callback_lambda()
        self.event_rule = self._build_event_rule()

        self._emit_outputs()

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------

    def _build_session_bucket(self) -> s3.Bucket:
        # No `bucket_name=` — CDK generates a unique name per stack so redeploys after
        # teardown does not hit S3's global-namespace settling period. Integration validation
        # discovers the bucket via CFN output `SessionBucketOut`.
        return s3.Bucket(
            self,
            "SessionBucket",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.data_key,
            bucket_key_enabled=True,
            versioned=True,
            enforce_ssl=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            server_access_logs_bucket=self.access_log_bucket,
            server_access_logs_prefix="session-bucket/",
            removal_policy=RemovalPolicy.RETAIN,
        )

    def _build_jobs_table(self) -> dynamodb.Table:
        table = dynamodb.Table(
            self,
            "TrainingJobsTable",
            table_name=f"{self.hyphen_project}-training-jobs",
            partition_key=dynamodb.Attribute(name="job_name", type=dynamodb.AttributeType.STRING),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            point_in_time_recovery_specification=dynamodb.PointInTimeRecoverySpecification(
                point_in_time_recovery_enabled=True
            ),
            encryption=dynamodb.TableEncryption.CUSTOMER_MANAGED,
            encryption_key=self.data_key,
            removal_policy=RemovalPolicy.RETAIN,
        )
        table.add_global_secondary_index(
            index_name="SessionIdIndex",
            partition_key=dynamodb.Attribute(
                name="runtime_session_id", type=dynamodb.AttributeType.STRING
            ),
        )
        return table

    def _build_sagemaker_role(self) -> iam.Role:
        role = iam.Role(
            self,
            "SageMakerExecutionRole",
            role_name=f"{self.hyphen_project}-sagemaker-exec",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="SageMaker training job execution role - S3 + CW Logs",
        )
        grant_bucket_read_write(
            role,
            self.session_bucket,
            encryption_key=self.data_key,
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                    "logs:DescribeLogStreams",
                ],
                resources=[f"arn:aws:logs:{self.region_name}:{self.account}:log-group:/aws/sagemaker/*"],
            )
        )
        # Phase 5 → Phase 6 fold: SageMaker training/processing jobs read inputs from
        # and write artifacts to the medallion LAKE bucket (separate from this stack's
        # session bucket). Phase 5 added this as a live INLINE policy
        # (`AmlasLakeS3ReadWrite`) after a runtime S3 AccessDenied; Phase 6 folds it
        # into the CDK so it is permanent IaC (no manual inline policy). The lake
        # bucket is owned by LakeStack; reference it by its conventional
        # account/region-tokenized name (SEC-01: account is a CDK token, not a literal).
        # Phase 7 fold: SageMaker Feature Store's OFFLINE store additionally requires
        # bucket-level s3:GetBucketAcl / s3:GetBucketLocation and object ACL/tagging on
        # this same lake bucket (the AWS-managed AmazonSageMakerFeatureStoreAccess only
        # covers *sagemaker*-named buckets, not our lake). A real CreateFeatureGroup
        # returned AccessDenied(s3:GetBucketAcl) until these were granted; proven live
        # via a real online+offline FG round-trip, then folded here as permanent IaC.
        lake_bucket_arn = (
            f"arn:aws:s3:::{self.hyphen_project}-lake-{self.account}-{self.region_name}"
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="AmlasLakeS3ReadWrite",
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket",
                    # Feature Store offline-store requirements (Phase 7):
                    "s3:GetBucketAcl",
                    "s3:GetBucketLocation",
                    "s3:PutObjectAcl",
                    "s3:PutObjectTagging",
                ],
                resources=[lake_bucket_arn, f"{lake_bucket_arn}/*"],
            )
        )
        # FIX-14 (Phase 6): the lake bucket is SSE-KMS with LakeStack's own CMK, so a
        # REAL SageMaker training/processing job (which downloads the encrypted training
        # input under this exec role) needs kms:Decrypt/GenerateDataKey on that lake CMK
        # — the S3 grant above is necessary but NOT sufficient. The earlier local-sklearn
        # e2e never surfaced this (no container-side decrypt); a real CreateTrainingJob
        # failed with "AccessDenied (403): ... kms:Decrypt on resource: <lake CMK> because
        # no identity-based policy allows the kms:Decrypt action". The lake CMK is owned by
        # a SEPARATE stack and its key-id is generated (not a fixed name), so scope by the
        # ``kms:ViaService s3.<region>`` condition (the AWS-recommended pattern when the CMK
        # ARN is unknown at authoring time) — bounded to S3-mediated crypto, which is exactly
        # what training-input download, Feature Store offline write, and gold write use.
        # Mirrors the runtime FIX-11 ``SseKmsForBuckets`` ViaService-s3 statement.
        role.add_to_policy(
            iam.PolicyStatement(
                sid="AmlasLakeKmsViaS3",
                actions=[
                    "kms:Decrypt",
                    "kms:Encrypt",
                    "kms:ReEncryptFrom",
                    "kms:ReEncryptTo",
                    "kms:GenerateDataKey",
                    "kms:GenerateDataKeyWithoutPlaintext",
                    "kms:DescribeKey",
                ],
                resources=[f"arn:aws:kms:{self.region_name}:{self.account}:key/*"],
                conditions={
                    "StringEquals": {
                        "kms:ViaService": f"s3.{self.region_name}.amazonaws.com"
                    }
                },
            )
        )
        # WA-07 (least-privilege IAM): replace the broad ``AmazonSageMakerFullAccess`` managed
        # policy with resource-scoped statements (mirrors the hand-scoped runtime role in
        # runtime_stack.py). Control-plane Create*/List* have no per-resource form (the resource
        # is the job being created), so they stay on "*" like the proven runtime-role pattern;
        # describe/stop are scoped to the training-job ARN. SEC-01: account/region are CDK tokens.
        role.add_to_policy(
            iam.PolicyStatement(
                sid="SageMakerTrainingControlPlane",
                actions=[
                    "sagemaker:CreateTrainingJob",
                    "sagemaker:CreateProcessingJob",
                    "sagemaker:ListTrainingJobs",
                    "sagemaker:ListProcessingJobs",
                ],
                resources=["*"],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="SageMakerTrainingJobScoped",
                actions=[
                    "sagemaker:DescribeTrainingJob",
                    "sagemaker:StopTrainingJob",
                    "sagemaker:DescribeProcessingJob",
                    "sagemaker:StopProcessingJob",
                    "sagemaker:AddTags",
                ],
                resources=[
                    f"arn:aws:sagemaker:{self.region_name}:{self.account}:training-job/*",
                    f"arn:aws:sagemaker:{self.region_name}:{self.account}:processing-job/*",
                ],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="EcrPullAuth",
                actions=["ecr:GetAuthorizationToken"],
                # ECR authorization tokens do not support resource-level IAM.
                resources=["*"],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="EcrRepositoryPull",
                actions=[
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:BatchGetImage",
                ],
                resources=[
                    f"arn:aws:ecr:{self.region_name}:"  # nosemgrep
                    f"{self.account}:repository/*"
                ],
            )
        )
        acknowledge_iam5_resources(
            role,
            {
                "*": (
                    "SageMaker create/list and ECR authorization-token APIs lack "
                    "resource-level IAM support; actions are explicit and pull APIs are separate"
                ),
                cfn_getatt_arn(self.session_bucket, suffix="/*"): (
                    "object-key wildcard is confined to the stack-owned encrypted session bucket"
                ),
                f"arn:aws:ecr:{self.region_name}:{self.account}:repository/*": (
                    "training images may come from any repository in this account and region; "
                    "only the three read-only ECR pull APIs are allowed"
                ),
                f"arn:aws:kms:{self.region_name}:{self.account}:key/*": (
                    "the lake CMK is created by a separate stack with a generated key ID; "
                    "kms:ViaService confines use to S3 in this region"
                ),
                f"arn:aws:logs:{self.region_name}:{self.account}:log-group:/aws/sagemaker/*": (
                    "log writes are confined to this account's SageMaker log-group namespace"
                ),
                f"arn:aws:s3:::{self.hyphen_project}-lake-{self.account}-{self.region_name}/*": (
                    "object-key wildcard is confined to the single convention-named lake bucket"
                ),
                f"arn:aws:sagemaker:{self.region_name}:{self.account}:processing-job/*": (
                    "describe, stop, and tag are confined to processing jobs in this account and region"
                ),
                f"arn:aws:sagemaker:{self.region_name}:{self.account}:training-job/*": (
                    "describe, stop, and tag are confined to training jobs in this account and region"
                ),
            },
        )
        return role

    def _build_callback_log_group(self) -> logs.LogGroup:
        return logs.LogGroup(
            self,
            "CallbackLogGroup",
            log_group_name=f"/aws/lambda/{self.hyphen_project}-training-callback",
            retention=logs.RetentionDays.ONE_MONTH,
            encryption_key=self.data_key,
            removal_policy=RemovalPolicy.DESTROY,
        )

    def _build_lambda_role(self) -> iam.Role:
        role = iam.Role(
            self,
            "CallbackLambdaRole",
            role_name=f"{self.hyphen_project}-training-callback-role",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
        )
        # The callback log group is pre-created, so grant only stream creation and
        # event writes on that group instead of attaching AWSLambdaBasicExecutionRole.
        self.callback_log_group.grant_write(role)
        grant_table_read_write_data(
            role,
            self.jobs_table,
            encryption_key=self.data_key,
        )
        # InvokeAgentRuntime — SEC-06 / R-008: scoped wide ONLY because this stack is
        # intentionally deployable without a known Runtime ARN (the runtime ARN is a
        # CfnParameter defaulting to ""). When a concrete `AGENTCORE_RUNTIME_ARN` is
        # supplied, the resource is narrowed to that single runtime ARN; otherwise it
        # falls back to a region/account-scoped runtime pattern (NOT a bare "*").
        #
        # BUGFIX (LIVE-0): `value_as_string` is a DEPLOY-TIME token that is ALWAYS
        # truthy under a synth-time Python `if`, so the old ternary always emitted the
        # raw parameter (the empty default) → IAM rejected an empty Resource
        # ("Resource must be in ARN format or '*'"). The branch MUST be a deploy-time
        # CfnCondition + Fn.condition_if so the empty-vs-supplied decision happens in
        # CloudFormation, never at synth time.
        runtime_arn_supplied = CfnCondition(
            self,
            "RuntimeArnSupplied",
            expression=Fn.condition_not(
                Fn.condition_equals(self.runtime_arn_param.value_as_string, "")
            ),
        )
        runtime_fallback = (
            f"arn:aws:bedrock-agentcore:{self.region}:{self.account}:runtime/*"
        )
        invoke_runtime_resource = Fn.condition_if(
            runtime_arn_supplied.logical_id,
            self.runtime_arn_param.value_as_string,
            runtime_fallback,
        ).to_string()
        role.add_to_policy(
            iam.PolicyStatement(
                sid="InvokeAgentRuntime",
                actions=["bedrock-agentcore:InvokeAgentRuntime"],
                resources=[invoke_runtime_resource],
            )
        )
        conditional_resource = json.dumps(
            self.resolve(invoke_runtime_resource),
            separators=(",", ":"),
        )
        acknowledge_iam5_resources(
            role,
            {
                conditional_resource: (
                    "deployment uses the supplied single runtime ARN when present; "
                    "the fallback is limited to runtimes in this account and deployment region"
                )
            },
        )
        return role

    def _build_callback_lambda(self) -> _lambda.Function:
        return _lambda.Function(
            self,
            "CallbackFunction",
            function_name=f"{self.hyphen_project}-training-callback",
            runtime=_lambda.Runtime.PYTHON_3_14,
            architecture=_lambda.Architecture.ARM_64,
            handler="amlas.async_callback.lambda_handler.handler",
            code=_lambda.Code.from_asset(_SRC_ASSET),
            role=self.lambda_role,
            log_group=self.callback_log_group,
            timeout=Duration.minutes(2),
            memory_size=512,
            environment={
                "AGENTCORE_RUNTIME_ARN": self.runtime_arn_param.value_as_string,
                "JOBS_TABLE": self.jobs_table.table_name,
            },
        )

    def _build_event_rule(self) -> events.Rule:
        rule = events.Rule(
            self,
            "SageMakerStateChangeRule",
            rule_name=f"{self.hyphen_project}-sagemaker-state-change",
            description="Catches SageMaker training job state transitions and resumes the agent",
            event_pattern=events.EventPattern(
                source=["aws.sagemaker"],
                detail_type=["SageMaker Training Job State Change"],
                detail={
                    "TrainingJobStatus": ["Completed", "Failed", "Stopped"],
                },
            ),
        )
        rule.add_target(targets.LambdaFunction(self.callback_fn))
        return rule

    def _emit_outputs(self) -> None:
        CfnOutput(
            self, "JobsTableNameOut", value=self.jobs_table.table_name,
            export_name=f"{self.hyphen_project}-jobs-table",
        )
        CfnOutput(
            self, "SageMakerExecutionRoleArnOut", value=self.sagemaker_role.role_arn,
            export_name=f"{self.hyphen_project}-sagemaker-role-arn",
        )
        CfnOutput(
            self, "SessionBucketOut", value=self.session_bucket.bucket_name,
            export_name=f"{self.hyphen_project}-async-session-bucket",
        )
        CfnOutput(
            self, "CallbackFunctionNameOut", value=self.callback_fn.function_name,
            export_name=f"{self.hyphen_project}-callback-function-name",
        )
