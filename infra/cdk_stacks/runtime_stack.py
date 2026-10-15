"""RuntimeStack — provisions everything AgentCore Runtime needs.

Resources:
    - S3 session bucket (encrypted, versioned, public-access-blocked)
    - DynamoDB training-jobs table w/ GSI on runtime_session_id
    - IAM runtime execution role (trust = bedrock-agentcore service principal)
    - IAM SageMaker execution role (trust = sagemaker.amazonaws.com)
    - IAM Lambda callback role (trust = lambda.amazonaws.com, invoke runtime perm)
    - CloudWatch log groups (APPLICATION_LOGS + USAGE_LOGS dual delivery)
    - AgentCore Memory (L1 Cfn, 3 strategies, 90-day expiry)
    - AgentCore CodeInterpreter (L1 Cfn, shared session)
    - AgentCore Runtime (L1 Cfn, accepts container image URI as CfnParameter)
    - Lambda function (callback handler, arm64)
    - EventBridge rule (SageMaker state-change → Lambda)
    - PatchRuntimeOtel AwsCustomResource (non-optional for Bedrock eval API)

Naming constraints satisfied:
    - project_name, memory name, code_interpreter name, runtime name match [a-zA-Z0-9_]+
    - DDB table + S3 bucket can be hyphenated (their services allow it)
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aws_cdk import (
    CfnOutput,
    CfnParameter,
    CfnResource,
    Duration,
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
from aws_cdk import (
    custom_resources as cr,
)
from constructs import Construct

# Safe to import from amlas.config: pure data module with zero boto3 imports.
from amlas.config.runtime import bedrock_invoke_resources

from ._iam import (
    acknowledge_iam5_resources,
    cfn_getatt_arn,
    grant_bucket_read_write,
    grant_table_read_write_data,
)
from ._kms import make_data_key
from ._s3 import make_access_log_bucket
from ._tags import apply_cost_tags, deployment_tags

# Lambda code asset root. Resolved from ``__file__`` (repo root = parents[2] of
# ``infra/cdk_stacks/<stack>.py``) rather than left as the bare relative ``"src"``,
# which CDK resolves against the PROCESS CWD — so a synth from anywhere but the repo
# root raised «CannotFindAsset». Content-addressed asset hashing means the resolved
# directory is identical to what the old CWD-relative path produced from the repo
# root, so deploys are unaffected.
_SRC_ASSET = str(Path(__file__).resolve().parents[2] / "src")


class RuntimeStack(Stack):
    def __init__(
        self,
        scope: Construct,
        cid: str,
        *,
        project_name: str = "amlas",
        hyphen_project: str = "amlas",
        tavily_secret_id: str | None = None,
        model_package_group_name: str | None = None,
        region_name: str | None = None,
        event_expiry_days: int = 90,
        kms_key_arn: str = "",
        runtime_network_mode: str = "PUBLIC",
        capture_content: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, cid, **kwargs)
        assert project_name.replace("_", "").isalnum(), (
            "project_name must match [a-zA-Z0-9_]+ (AgentCore constraint)"
        )

        self.project_name = project_name
        self.hyphen_project = hyphen_project
        self.tavily_secret_id = (
            tavily_secret_id if tavily_secret_id is not None else f"{hyphen_project}/tavily-api-key"
        )
        self.model_package_group_name = (
            model_package_group_name if model_package_group_name is not None else f"{hyphen_project}-models"
        )
        self.region_name = region_name or self.region
        if self.region_name != self.region:
            raise ValueError(
                "region_name must match the stack environment; configure the deployment region through env."
            )
        # WA-05 (Security/Network): the AgentCore Runtime + Code Interpreter NetworkMode is a
        # configurable seam. Default PUBLIC ONLY because AgentCore VPC-mode support is the
        # external blocker (audit §7 / R-009); an adopter flips this to a private mode
        # (env AMLAS_RUNTIME_NETWORK_MODE → infra.app) once VPC-mode is available — no code
        # change, no account-specific value hardcoded.
        self.runtime_network_mode = runtime_network_mode
        # WA-04 (privacy): OTEL message-content capture — redact-by-default (False).
        self.capture_content = capture_content
        # WA-12: cost-allocation tags (project/env/owner) for chargeback.
        apply_cost_tags(self)

        # WA-01: customer-managed CMK (rotation on) — created by default, or the
        # adopter's shared/BYO key imported when ``kms_key_arn`` is supplied. Used to
        # encrypt the session bucket, the DDB jobs table, AND the CloudWatch log groups
        # (``for_logs=True`` adds the Logs service-principal key-policy grant).
        self.data_key = make_data_key(
            self,
            "DataKey",
            key_arn=kms_key_arn,
            alias=f"alias/{hyphen_project}-runtime",
            for_logs=True,
            description="Amlas runtime CMK (rotation on) — session/jobs/logs at rest",
        )

        # Parameters — user supplies Runtime image URI after CodeBuild
        self.runtime_image_uri = CfnParameter(
            self,
            "RuntimeImageUri",
            type="String",
            default="",
            description="ECR URI of the agent runtime container (linux/arm64). "
            "Leave blank on first deploy — populate after `agentcore launch` builds the image.",
        )

        self.access_log_bucket = make_access_log_bucket(self)
        self.session_bucket = self._build_session_bucket()
        self.jobs_table = self._build_jobs_table()
        self.sagemaker_role = self._build_sagemaker_role()
        self.runtime_role = self._build_runtime_role()
        (
            self.app_log_group,
            self.usage_log_group,
            self.callback_log_group,
        ) = self._build_log_groups()
        self.memory = self._build_memory(event_expiry_days)
        self.code_interpreter = self._build_code_interpreter()
        # CodeInterpreter + Memory IAM statements reference self.code_interpreter
        # and self.memory via get_att(...), so they can only be attached to the
        # runtime role AFTER those L1 resources exist (see task 4.3).
        self._attach_runtime_scoped_policies()
        self.runtime = self._build_runtime()
        # lambda_role MUST be built after self.runtime so _build_lambda_role
        # can scope bedrock-agentcore:InvokeAgentRuntime to
        # self.runtime.get_att("AgentRuntimeArn") (see task 4.5).
        self.lambda_role = self._build_lambda_role()
        self.callback_fn = self._build_callback_lambda()
        self.event_rule = self._build_event_rule()
        self.patch_runtime_otel = self._build_patch_runtime_otel()

        self._emit_outputs()

    # ------------------------------------------------------------------
    # Data layer
    # ------------------------------------------------------------------

    def _build_session_bucket(self) -> s3.Bucket:
        return s3.Bucket(
            self,
            "SessionBucket",
            bucket_name=f"{self.hyphen_project}-{self.account}-{self.region_name}",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=self.data_key,
            bucket_key_enabled=True,
            versioned=True,
            enforce_ssl=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            server_access_logs_bucket=self.access_log_bucket,
            server_access_logs_prefix="runtime-session-bucket/",
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
            partition_key=dynamodb.Attribute(name="runtime_session_id", type=dynamodb.AttributeType.STRING),
        )
        return table

    # ------------------------------------------------------------------
    # IAM roles
    # ------------------------------------------------------------------

    def _build_sagemaker_role(self) -> iam.Role:
        role = iam.Role(
            self,
            "SageMakerExecutionRole",
            role_name=f"{self.hyphen_project}-sagemaker-exec",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="SageMaker training job execution role - S3 + CW Logs + MLflow",
        )
        # Explicit action allow-list for SageMaker training + session-bucket access.
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
        # and write artifacts to the medallion LAKE bucket (owned by LakeStack, separate
        # from this stack's session bucket). Phase 5 discovered+fixed this as a live
        # INLINE policy (`AmlasLakeS3ReadWrite`) after a runtime S3 AccessDenied; this
        # folds it into the CDK so it is permanent IaC. SEC-01: account is a CDK token.
        lake_bucket_arn = f"arn:aws:s3:::{self.hyphen_project}-lake-{self.account}-{self.region_name}"
        role.add_to_policy(
            iam.PolicyStatement(
                sid="AmlasLakeS3ReadWrite",
                actions=[
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket",
                ],
                resources=[lake_bucket_arn, f"{lake_bucket_arn}/*"],
            )
        )
        # FIX-14 (Phase 6): the lake bucket is SSE-KMS with LakeStack's own CMK, so a
        # REAL SageMaker training/processing job (downloading the encrypted training input
        # under this exec role) needs kms:Decrypt/GenerateDataKey on that lake CMK — the S3
        # grant above is necessary but NOT sufficient (a real CreateTrainingJob failed with
        # "AccessDenied (403): ... kms:Decrypt on resource: <lake CMK>"). The lake CMK is
        # owned by a SEPARATE stack with a generated key-id, so scope by the
        # ``kms:ViaService s3.<region>`` condition (bounded to S3-mediated crypto — exactly
        # what training-input download / gold write use). Mirrors FIX-11 ``SseKmsForBuckets``.
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
                conditions={"StringEquals": {"kms:ViaService": f"s3.{self.region_name}.amazonaws.com"}},
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
                resources=[f"arn:aws:ecr:{self.region_name}:{self.account}:repository/*"],
            )
        )
        acknowledge_iam5_resources(
            role,
            {
                "*": (
                    "ECR GetAuthorizationToken has no resource-level IAM support; "
                    "all repository pull APIs are in a separate account/region-scoped statement"
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
            },
        )
        return role

    def _bedrock_resources(self) -> list[str]:
        """Return the Bedrock ARNs required to invoke every allowed inference profile.

        AWS Bedrock requires BOTH the inference-profile ARN (region+account scoped)
        and the underlying foundation-model ARN (cross-region via `*`) in the
        `Resource` list for `bedrock:InvokeModel*`; without the FM ARN the call fails
        with AccessDeniedException (see design §3, Hypothesized Root Cause for C3_a).

        Sourced from the canonical ``BEDROCK_INVOKE_PROFILES`` (config.runtime) so
        the IAM allowlist stays in lockstep with the per-role model tiers — when
        Opus 4.8 became the ML-work default the deployed runtime's policy gained it
        automatically (no silent in-cloud AccessDenied).
        """
        return bedrock_invoke_resources(self.account, self.region_name)

    def _build_runtime_role(self) -> iam.Role:
        role = iam.Role(
            self,
            "RuntimeExecutionRole",
            role_name=f"{self.hyphen_project}-runtime-exec",
            assumed_by=iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
            description="AgentCore Runtime execution role for amlas",
        )

        # Bedrock InvokeModel (Opus / Sonnet / Haiku)
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                resources=self._bedrock_resources(),
            )
        )

        # AgentCore Code Interpreter + Memory statements are attached by
        # `_attach_runtime_scoped_policies()` once `self.code_interpreter` and
        # `self.memory` exist — their ARNs come from `get_att(...)` on L1
        # resources that are constructed after this role (see task 4.3).

        # Session data and jobs use explicit API allow-lists. The KMS grant is
        # attached once through the bucket helper because both stores share the key.
        grant_bucket_read_write(
            role,
            self.session_bucket,
            encryption_key=self.data_key,
        )
        grant_table_read_write_data(role, self.jobs_table)

        # SageMaker training — Create is SCOPED to the amlas-* training-job ARN
        # pattern (least-privilege; mirrors the deployed p6 SageMakerCreateScoped
        # policy). Only List (a list action with no per-resource semantics) needs "*".
        role.add_to_policy(
            iam.PolicyStatement(
                sid="AmlasCreateTrainingJobScoped",
                actions=["sagemaker:CreateTrainingJob"],
                resources=[
                    f"arn:aws:sagemaker:{self.region}:{self.account}:training-job/{self.hyphen_project}-*"
                ],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="AmlasListTrainingJobs",
                actions=["sagemaker:ListTrainingJobs"],
                resources=["*"],  # List is account-wide; no per-resource ARN exists.
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "sagemaker:DescribeTrainingJob",
                    "sagemaker:StopTrainingJob",
                ],
                resources=[f"arn:aws:sagemaker:{self.region_name}:{self.account}:training-job/*"],
            )
        )

        # iam:PassRole — scoped to the SageMaker exec role only, conditioned on service
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["iam:PassRole"],
                resources=[self.sagemaker_role.role_arn],
                conditions={"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}},
            )
        )

        # CloudWatch Logs
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
                resources=[
                    f"arn:aws:logs:{self.region_name}:{self.account}:log-group:/aws/bedrock-agentcore/*"
                ],
            )
        )

        # EventBridge (for emitting our own events if needed)
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["events:PutEvents"],
                resources=[f"arn:aws:events:{self.region_name}:{self.account}:event-bus/default"],
            )
        )

        # ---- MLOps de-mock extensions (design C.2.8) --------------------
        # Model Registry + Clarify/Processing + Feature Store + Monitor +
        # PassRole-for-processing — the runtime invokes these tools for real in
        # the de-stubbed suite. Control-plane actions stay on "*" (Create/List
        # have no per-resource semantics); runtime put/get is feature-group-scoped.
        role.add_to_policy(
            iam.PolicyStatement(
                sid="MlopsControlPlane",
                actions=[
                    "sagemaker:CreateProcessingJob",
                    "sagemaker:DescribeProcessingJob",
                    "sagemaker:StopProcessingJob",
                    "sagemaker:ListProcessingJobs",
                    "sagemaker:CreatePipeline",
                    "sagemaker:UpdatePipeline",
                    "sagemaker:DescribePipeline",
                    "sagemaker:StartPipelineExecution",
                    "sagemaker:DescribePipelineExecution",
                    "sagemaker:CreateModelPackage",
                    "sagemaker:CreateModelPackageGroup",
                    "sagemaker:UpdateModelPackage",
                    "sagemaker:DescribeModelPackage",
                    "sagemaker:DescribeModelPackageGroup",
                    "sagemaker:ListModelPackages",
                    "sagemaker:ListModelPackageGroups",
                    "sagemaker:CreateFeatureGroup",
                    "sagemaker:DescribeFeatureGroup",
                    "sagemaker:DeleteFeatureGroup",
                    "sagemaker:ListFeatureGroups",
                    "sagemaker:CreateMonitoringSchedule",
                    "sagemaker:DescribeMonitoringSchedule",
                    "sagemaker:DeleteMonitoringSchedule",
                    "sagemaker:ListMonitoringSchedules",
                ],
                resources=["*"],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="FeatureStoreRuntime",
                actions=[
                    "sagemaker:PutRecord",
                    "sagemaker:GetRecord",
                    "sagemaker:DeleteRecord",
                ],
                resources=[f"arn:aws:sagemaker:{self.region_name}:{self.account}:feature-group/*"],
            )
        )
        # Secrets Manager — read the Tavily key secret (value set out-of-band).
        role.add_to_policy(
            iam.PolicyStatement(
                sid="TavilySecretRead",
                actions=["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"],
                resources=[
                    f"arn:aws:secretsmanager:{self.region_name}:{self.account}:secret:{self.tavily_secret_id}-*"
                ],
            )
        )

        residual_findings = {
            "*": (
                "explicit SageMaker create/list control-plane APIs lack a stable "
                "pre-existing resource ARN; PassRole is separately ARN and service scoped"
            ),
            cfn_getatt_arn(self.session_bucket, suffix="/*"): (
                "object-key wildcard is confined to the stack-owned encrypted session bucket"
            ),
            f"arn:aws:logs:{self.region_name}:{self.account}:log-group:/aws/bedrock-agentcore/*": (
                "log writes are confined to this account's AgentCore log-group namespace"
            ),
            f"arn:aws:sagemaker:{self.region_name}:{self.account}:feature-group/*": (
                "online feature-store APIs are confined to feature groups in this account and region"
            ),
            f"arn:aws:sagemaker:{self.region_name}:{self.account}:training-job/*": (
                "describe and stop are confined to training jobs in this account and region"
            ),
            (f"arn:aws:secretsmanager:{self.region_name}:{self.account}:secret:{self.tavily_secret_id}-*"): (
                "Secrets Manager appends an unpredictable suffix; the secret path is fixed"
            ),
        }
        for resource in self._bedrock_resources():
            if resource.startswith("arn:aws:bedrock:*::foundation-model/"):
                residual_findings[resource] = (
                    "Bedrock cross-region inference requires the same allow-listed model ID "
                    "in each destination region; service and model are fixed"
                )
        acknowledge_iam5_resources(role, residual_findings)
        return role

    def _attach_runtime_scoped_policies(self) -> None:
        """Attach IAM statements that require `self.memory` / `self.code_interpreter`.

        These statements must be scoped to the stack-owned Memory / Code
        Interpreter ARNs (Properties P4 and P5 / bug conditions C3_b, C3_c),
        but those L1 resources are constructed AFTER `self.runtime_role`.
        The `__init__` ordering therefore builds the role first, then the two
        resources, and finally calls this method to attach the scoped
        statements via `runtime_role.add_to_policy(...)`.
        """
        # AgentCore Code Interpreter — scoped to the stack-owned CI's ARN.
        self.runtime_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock-agentcore:CreateCodeInterpreter",
                    "bedrock-agentcore:StartCodeInterpreterSession",
                    "bedrock-agentcore:InvokeCodeInterpreter",
                    "bedrock-agentcore:StopCodeInterpreterSession",
                    "bedrock-agentcore:DeleteCodeInterpreter",
                    "bedrock-agentcore:ListCodeInterpreters",
                    "bedrock-agentcore:GetCodeInterpreter",
                ],
                resources=[self.code_interpreter.get_att("CodeInterpreterArn").to_string()],
            )
        )

        # AgentCore Memory — scoped to the stack-owned Memory's ARN.
        self.runtime_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock-agentcore:CreateMemory",
                    "bedrock-agentcore:GetMemory",
                    "bedrock-agentcore:UpdateMemory",
                    "bedrock-agentcore:DeleteMemory",
                    "bedrock-agentcore:CreateEvent",
                    "bedrock-agentcore:GetEvent",
                    "bedrock-agentcore:ListEvents",
                    "bedrock-agentcore:GetLastKTurns",
                    "bedrock-agentcore:ListSessions",
                ],
                resources=[self.memory.get_att("MemoryArn").to_string()],
            )
        )

    def _build_lambda_role(self) -> iam.Role:
        role = iam.Role(
            self,
            "CallbackLambdaRole",
            role_name=f"{self.hyphen_project}-callback-lambda",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Callback Lambda role - invoke AgentCore Runtime + read DDB",
        )
        # The callback log group is pre-created, so grant only stream creation and
        # event writes on that group instead of attaching AWSLambdaBasicExecutionRole.
        self.callback_log_group.grant_write(role)
        grant_table_read_write_data(
            role,
            self.jobs_table,
            encryption_key=self.data_key,
        )
        role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock-agentcore:InvokeAgentRuntime"],
                resources=[self.runtime.get_att("AgentRuntimeArn").to_string()],
            )
        )
        return role

    # ------------------------------------------------------------------
    # Observability — dual log delivery
    # ------------------------------------------------------------------

    def _build_log_groups(
        self,
    ) -> tuple[logs.LogGroup, logs.LogGroup, logs.LogGroup]:
        app_lg = logs.LogGroup(
            self,
            "RuntimeAppLogGroup",
            log_group_name=f"/aws/bedrock-agentcore/{self.project_name}/app",
            retention=logs.RetentionDays.ONE_MONTH,
            encryption_key=self.data_key,  # WA-01: CMK-encrypt agent logs at rest
            removal_policy=RemovalPolicy.DESTROY,
        )
        usage_lg = logs.LogGroup(
            self,
            "RuntimeUsageLogGroup",
            log_group_name=f"/aws/bedrock-agentcore/{self.project_name}/usage",
            retention=logs.RetentionDays.THREE_MONTHS,
            encryption_key=self.data_key,  # WA-01: CMK-encrypt agent logs at rest
            removal_policy=RemovalPolicy.DESTROY,
        )
        callback_lg = logs.LogGroup(
            self,
            "CallbackLogGroup",
            log_group_name=f"/aws/lambda/{self.hyphen_project}-training-callback",
            retention=logs.RetentionDays.ONE_MONTH,
            encryption_key=self.data_key,
            removal_policy=RemovalPolicy.DESTROY,
        )
        return app_lg, usage_lg, callback_lg

    # ------------------------------------------------------------------
    # AgentCore primitives (L1 CfnResource — version-portable)
    # ------------------------------------------------------------------

    def _build_memory(self, event_expiry_days: int) -> CfnResource:
        return CfnResource(
            self,
            "AgentCoreMemory",
            type="AWS::BedrockAgentCore::Memory",
            properties={
                "Name": f"{self.project_name}_memory",
                "Tags": deployment_tags(self.hyphen_project),
                "Description": "Amlas agents — 3-strategy long-term memory",
                "EventExpiryDuration": event_expiry_days,
                "MemoryStrategies": [
                    {
                        "UserPreferenceMemoryStrategy": {
                            "Name": "UserPreference",
                            "Description": "Per-project ML preferences, favored algos, thresholds",
                        }
                    },
                    {
                        "SemanticMemoryStrategy": {
                            "Name": "SemanticFact",
                            "Description": "Schema facts, leakage patterns, known-bad features",
                        }
                    },
                    {
                        "SummaryMemoryStrategy": {
                            "Name": "Summary",
                            "Description": "Per-run summaries for iteration carry-over",
                        }
                    },
                ],
            },
        )

    def _build_code_interpreter(self) -> CfnResource:
        return CfnResource(
            self,
            "AgentCoreCodeInterpreter",
            type="AWS::BedrockAgentCore::CodeInterpreterCustom",
            properties={
                "Name": f"{self.project_name}_code_interpreter",
                "Tags": deployment_tags(self.hyphen_project),
                "Description": "Shared code interpreter for agent tool calls",
                "ExecutionRoleArn": self.runtime_role.role_arn,
                "NetworkConfiguration": {"NetworkMode": self.runtime_network_mode},
            },
        )

    def _build_runtime(self) -> CfnResource:
        """AgentCore Runtime — references container image via CfnParameter.

        Image URI is supplied at deploy time (after `agentcore launch` builds + pushes it),
        e.g. `cdk deploy --parameters RuntimeImageUri=<uri>`.
        """
        return CfnResource(
            self,
            "AgentCoreRuntime",
            type="AWS::BedrockAgentCore::Runtime",
            properties={
                "AgentRuntimeName": self.project_name,
                "Tags": deployment_tags(self.hyphen_project),
                "Description": "Amlas engineering agents — main runtime",
                "AgentRuntimeArtifact": {
                    "ContainerConfiguration": {
                        "ContainerUri": self.runtime_image_uri.value_as_string,
                    }
                },
                "RoleArn": self.runtime_role.role_arn,
                "NetworkConfiguration": {"NetworkMode": self.runtime_network_mode},
                "EnvironmentVariables": {
                    "OTEL_PYTHON_DISTRO": "aws_distro",
                    "OTEL_PYTHON_CONFIGURATOR": "aws_configurator",
                    "OTEL_PYTHON_DISABLED_INSTRUMENTATIONS": "http,starlette",
                    "OTEL_TRACES_SAMPLER": "always_on",
                    "OTEL_TRACES_SAMPLER_ARG": "1.0",
                    "OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT": "4096",
                    # WA-04 (privacy): redact-by-default — message-content capture is OFF
                    # unless an adopter deliberately enables it (capture_content=True ←
                    # AMLAS_CAPTURE_CONTENT). Capturing full prompt/response content aids
                    # debugging but is a PII/compliance risk (audit §8 decision #3).
                    "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": (
                        "true" if self.capture_content else "false"
                    ),
                    "OTEL_INSTRUMENTATION_GENAI_EXPERIMENTAL_MODE": "default",
                    "AGENT_OBSERVABILITY_ENABLED": "true",
                    "AMLAS_AWS_REGION": self.region_name,
                    "AMLAS_DEPLOYMENT_PREFIX": self.hyphen_project,
                    "JOBS_TABLE": self.jobs_table.table_name,
                    "SESSION_BUCKET": self.session_bucket.bucket_name,
                    # ARN below is a resource identifier, not a credential; exposure in
                    # env/logs does not grant access (IAM policy still gates AssumeRole).
                    "SAGEMAKER_EXECUTION_ROLE_ARN": self.sagemaker_role.role_arn,
                    "MEMORY_ID": self.memory.ref,
                    "CODE_INTERPRETER_ID": self.code_interpreter.ref,
                    # ---- MLOps de-mock config (design C.2.8) ----
                    # Lake prefixes point at the LakeStack bucket (same account/region
                    # token pattern); registry/secret are the deployed real names.
                    # Settings/LakeConfig/MLOpsConfig read these env vars verbatim.
                    "AGENTICML_LAKE_SILVER_PREFIX": (
                        f"s3://{self.hyphen_project}-lake-{self.account}-{self.region_name}/silver/"
                    ),
                    "AGENTICML_LAKE_GOLD_PREFIX": (
                        f"s3://{self.hyphen_project}-lake-{self.account}-{self.region_name}/gold/"
                    ),
                    "AGENTICML_MODEL_PACKAGE_GROUP": self.model_package_group_name,
                    "AGENTICML_FEATURE_STORE_DB": f"{self.hyphen_project.replace('-', '_')}_features",
                    "TAVILY_SECRET_ID": self.tavily_secret_id,
                    # Fresh ISO-8601 UTC timestamp on every synth so CFN
                    # detects drift on pure-code image updates (README
                    # gotcha #12 / PLAN §13). Deliberate noise in `cdk diff`.
                    "BUILD_TIMESTAMP": datetime.now(UTC).isoformat(),
                },
            },
        )

    # ------------------------------------------------------------------
    # Async training: Lambda callback + EventBridge rule
    # ------------------------------------------------------------------

    def _build_callback_lambda(self) -> _lambda.Function:
        return _lambda.Function(
            self,
            "CallbackFunction",
            function_name=f"{self.hyphen_project}-training-callback",
            runtime=_lambda.Runtime.PYTHON_3_14,
            architecture=_lambda.Architecture.ARM_64,
            handler="amlas.async_callback.lambda_handler.handler",
            code=_lambda.Code.from_asset(_SRC_ASSET),  # stdlib + Lambda-provided boto3 only
            role=self.lambda_role,
            log_group=self.callback_log_group,
            timeout=Duration.minutes(2),
            memory_size=512,
            environment={
                "AGENTCORE_RUNTIME_ARN": self.runtime.get_att("AgentRuntimeArn").to_string(),
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

    # ------------------------------------------------------------------
    # PatchRuntimeOtel — non-optional for Bedrock eval API
    # ------------------------------------------------------------------

    def _build_patch_runtime_otel(self) -> cr.AwsCustomResource:
        """Post-create hook that flips Runtime observability to the 'default' GenAI mode.

        This is a documented requirement captured in research 04:
        `sample-mlops-agent/cdk/lib/stacks/backend/agentcore-stack.ts` — without this the
        Bedrock evaluation API won't accept Runtime traces. CDK custom resource calls
        `bedrock-agentcore-control:UpdateAgentRuntime`.
        """
        provider_role = iam.Role(
            self,
            "PatchRuntimeOtelRole",
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            description="Inline-only role for the Runtime OTEL patch custom resource",
        )
        provider_log_group = logs.LogGroup(
            self,
            "PatchRuntimeOtelLogGroup",
            log_group_name=f"/aws/lambda/{self.hyphen_project}-patch-runtime-otel",
            retention=logs.RetentionDays.ONE_MONTH,
            encryption_key=self.data_key,
            removal_policy=RemovalPolicy.RETAIN,
        )
        return cr.AwsCustomResource(
            self,
            "PatchRuntimeOtel",
            role=provider_role,
            log_group=provider_log_group,
            # AgentCore Control is newer than the SDK bundled in some Lambda
            # runtimes, so installation is an explicit compatibility decision.
            install_latest_aws_sdk=True,
            on_create=cr.AwsSdkCall(
                service="bedrock-agentcore-control",
                action="updateAgentRuntime",
                parameters={
                    "agentRuntimeId": self.runtime.ref,
                    "logConfiguration": {
                        "logTypes": ["APPLICATION_LOGS", "USAGE_LOGS"],
                        "cloudWatchLogsDeliveryConfiguration": {
                            "logGroupName": self.app_log_group.log_group_name,
                        },
                    },
                },
                physical_resource_id=cr.PhysicalResourceId.of("patch-runtime-otel"),
            ),
            on_update=cr.AwsSdkCall(
                service="bedrock-agentcore-control",
                action="updateAgentRuntime",
                parameters={"agentRuntimeId": self.runtime.ref},
                physical_resource_id=cr.PhysicalResourceId.of("patch-runtime-otel"),
            ),
            policy=cr.AwsCustomResourcePolicy.from_statements(
                [
                    iam.PolicyStatement(
                        sid="PatchRuntimeOtelUpdate",
                        actions=["bedrock-agentcore-control:UpdateAgentRuntime"],
                        resources=[self.runtime.get_att("AgentRuntimeArn").to_string()],
                    )
                ]
            ),
        )

    # ------------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------------

    def _emit_outputs(self) -> None:
        # CFN export names must match [A-Za-z0-9:-]+ — NO underscores. Use hyphen_project here.
        CfnOutput(
            self,
            "RuntimeArnOut",
            value=self.runtime.get_att("AgentRuntimeArn").to_string(),
            export_name=f"{self.hyphen_project}-runtime-arn",
        )
        CfnOutput(self, "MemoryIdOut", value=self.memory.ref, export_name=f"{self.hyphen_project}-memory-id")
        CfnOutput(
            self,
            "CodeInterpreterIdOut",
            value=self.code_interpreter.ref,
            export_name=f"{self.hyphen_project}-ci-id",
        )
        CfnOutput(
            self,
            "SessionBucketOut",
            value=self.session_bucket.bucket_name,
            export_name=f"{self.hyphen_project}-bucket",
        )
        CfnOutput(
            self,
            "JobsTableOut",
            value=self.jobs_table.table_name,
            export_name=f"{self.hyphen_project}-jobs-table",
        )
