"""BackendIamStack (NEW, M8 §8.7) — per-backend least-privilege IAM roles.

The hardening milestone (roadmap M8; doc 05 §8.7) gives each backend/adapter family its
OWN scoped IAM role + inline policy, so the managed adapters run with exactly the
permissions their port needs — never a shared, over-broad role. Four roles, one per
backend family that touches AWS (mirrors the ``backends/adapters/`` G1-G4 grouping):

    - TrainingBackend role   → ``sagemaker:CreateTrainingJob`` (+ describe/stop) scoped to
      the project training-job ARN, + scoped S3 read/write on the lake bucket, + a
      service-conditioned ``iam:PassRole``.
    - DataQualityBackend role → Glue DataBrew (``databrew:*JobRun``) + Glue Data Quality
      (``glue:*DataQualityRuleset`` / ``StartDataQualityRulesetEvaluationRun``) scoped to
      the project Glue/DataBrew ARNs + scoped lake S3.
    - DeploymentBackend role  → SageMaker hosting (``CreateEndpoint`` / ``CreateModel`` /
      ``CreateEndpointConfig`` + invoke) scoped to the project endpoint/model ARNs.
    - MonitoringBackend role  → SageMaker Model Monitor (``CreateMonitoringSchedule`` …)
      scoped to the project monitoring-schedule ARN + CloudWatch metric put (namespace-
      conditioned) + scoped Logs read.

Hardening invariant (the M8 anti-wildcard rule, doc 05 §8.7): every **sensitive** action
(``Create*`` / ``StartJobRun`` / ``StartDataQualityRulesetEvaluationRun``) is scoped to a
concrete resource ARN — NEVER ``Resource: "*"``. A offline validation
(``validation coverage``) asserts this for every statement in the stack, so a
future over-broad grant fails CI. Low-risk, resource-level-unsupported actions
(``cloudwatch:PutMetricData``, ``sagemaker:ListTrainingJobs``) may use ``*`` but are
condition-scoped (namespace) where the API allows it.

SEC-01: every ARN is composed from ``self.account`` / ``self.region`` CDK tokens (or a
service-scoped wildcard suffix) — NO hardcoded account IDs, no plaintext secrets. The
stack is ``cdk synth`` / unit-test only — it is NOT wired into ``app.py``'s deploy set
(M8 verifies offline; deployment is deferred).
"""

from __future__ import annotations

from typing import Any

from aws_cdk import (
    CfnOutput,
    Stack,
)
from aws_cdk import (
    aws_iam as iam,
)
from constructs import Construct

from ._iam import acknowledge_iam5_resources

# The sensitive actions that MUST always be resource-scoped (never Resource "*").
# The offline validation imports this so the stack and its guard share one source of truth.
SENSITIVE_ACTION_PREFIXES: tuple[str, ...] = (
    "sagemaker:CreateTrainingJob",
    "sagemaker:CreateProcessingJob",
    "sagemaker:CreateModel",
    "sagemaker:CreateEndpoint",
    "sagemaker:CreateEndpointConfig",
    "sagemaker:CreateMonitoringSchedule",
    "glue:StartDataQualityRulesetEvaluationRun",
    "glue:CreateDataQualityRuleset",
    "databrew:StartJobRun",
)


class BackendIamStack(Stack):
    """Per-backend least-privilege IAM roles for the managed adapters (M8 §8.7)."""

    def __init__(
        self,
        scope: Construct,
        cid: str,
        *,
        hyphen_project: str = "amlas",
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, cid, **kwargs)
        self.hyphen_project = hyphen_project

        # Scoped S3 ARNs — the medallion lake bucket only (LakeStack naming).
        self._lake_bucket_arns = [
            f"arn:aws:s3:::{hyphen_project}-lake-{self.account}-*",
            f"arn:aws:s3:::{hyphen_project}-lake-{self.account}-*/*",
        ]

        self.training_role = self._build_training_role()
        self.dataquality_role = self._build_dataquality_role()
        self.deployment_role = self._build_deployment_role()
        self.monitoring_role = self._build_monitoring_role()

        self._emit_outputs()

    # ── TrainingBackend role (sagemaker:CreateTrainingJob + scoped S3) ───────────
    def _build_training_role(self) -> iam.Role:
        role = iam.Role(
            self,
            "TrainingBackendRole",
            role_name=f"{self.hyphen_project}-be-training",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Least-priv role for the TrainingBackend managed adapter (M8 §8.7)",
        )
        training_arn = f"arn:aws:sagemaker:{self.region}:{self.account}:training-job/{self.hyphen_project}-*"
        role.add_to_policy(
            iam.PolicyStatement(
                sid="TrainingCreateScoped",
                actions=[
                    "sagemaker:CreateTrainingJob",
                    "sagemaker:DescribeTrainingJob",
                    "sagemaker:StopTrainingJob",
                    "sagemaker:AddTags",
                ],
                resources=[training_arn],
            )
        )
        # List is resource-level-unsupported → narrow but unavoidable * (NOT a Create*).
        role.add_to_policy(
            iam.PolicyStatement(
                sid="TrainingList",
                actions=["sagemaker:ListTrainingJobs"],
                resources=["*"],
            )
        )
        role.add_to_policy(self._scoped_lake_s3("TrainingLakeS3"))
        role.add_to_policy(self._pass_self_to_sagemaker("TrainingPassRole"))
        acknowledge_iam5_resources(
            role,
            {
                "*": "SageMaker ListTrainingJobs has no resource-level IAM support",
                training_arn: (
                    "training operations are confined to project-prefixed jobs in this account and region"
                ),
                f"arn:aws:iam::{self.account}:role/{self.hyphen_project}-be-*": (
                    "PassRole is confined to project backend roles and conditioned on SageMaker"
                ),
                **self._lake_iam5_findings(),
            },
        )
        return role

    # ── DataQualityBackend role (Glue DataBrew + Glue Data Quality) ──────────────
    def _build_dataquality_role(self) -> iam.Role:
        role = iam.Role(
            self,
            "DataQualityBackendRole",
            role_name=f"{self.hyphen_project}-be-dataquality",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("glue.amazonaws.com"),
                iam.ServicePrincipal("databrew.amazonaws.com"),
            ),
            description="Least-priv role for the DataQualityBackend managed adapter (M8 §8.7)",
        )
        databrew_arn = f"arn:aws:databrew:{self.region}:{self.account}:job/{self.hyphen_project}-*"
        glue_ruleset_arn = f"arn:aws:glue:{self.region}:{self.account}:dataQualityRuleset/{self.hyphen_project}-*"
        glue_ctx_arns = [
            f"arn:aws:glue:{self.region}:{self.account}:catalog",
            f"arn:aws:glue:{self.region}:{self.account}:database/{self.hyphen_project}*",
            f"arn:aws:glue:{self.region}:{self.account}:table/{self.hyphen_project}*/*",
        ]
        role.add_to_policy(
            iam.PolicyStatement(
                sid="DataBrewJobRunScoped",
                actions=[
                    "databrew:StartJobRun",
                    "databrew:DescribeJobRun",
                    "databrew:ListJobRuns",
                    "databrew:CreateProfileJob",
                    "databrew:CreateRecipeJob",
                ],
                resources=[databrew_arn],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="GlueDataQualityScoped",
                actions=[
                    "glue:CreateDataQualityRuleset",
                    "glue:StartDataQualityRulesetEvaluationRun",
                    "glue:GetDataQualityRulesetEvaluationRun",
                    "glue:GetDataQualityResult",
                ],
                resources=[glue_ruleset_arn, *glue_ctx_arns],
            )
        )
        role.add_to_policy(self._scoped_lake_s3("DataQualityLakeS3"))
        acknowledge_iam5_resources(
            role,
            {
                databrew_arn: (
                    "DataBrew operations are confined to project-prefixed jobs in this account and region"
                ),
                glue_ruleset_arn: (
                    "Glue DQ operations are confined to project-prefixed rulesets in this account and region"
                ),
                glue_ctx_arns[1]: "Glue context is confined to project-prefixed databases",
                glue_ctx_arns[2]: "Glue context is confined to tables in project-prefixed databases",
                **self._lake_iam5_findings(),
            },
        )
        return role

    # ── DeploymentBackend role (SageMaker hosting endpoint actions) ──────────────
    def _build_deployment_role(self) -> iam.Role:
        role = iam.Role(
            self,
            "DeploymentBackendRole",
            role_name=f"{self.hyphen_project}-be-deployment",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Least-priv role for the DeploymentBackend managed adapter (M8 §8.7)",
        )
        ep = f"arn:aws:sagemaker:{self.region}:{self.account}:endpoint/{self.hyphen_project}-*"
        ep_cfg = f"arn:aws:sagemaker:{self.region}:{self.account}:endpoint-config/{self.hyphen_project}-*"
        model = f"arn:aws:sagemaker:{self.region}:{self.account}:model/{self.hyphen_project}-*"
        transform = f"arn:aws:sagemaker:{self.region}:{self.account}:transform-job/{self.hyphen_project}-*"
        role.add_to_policy(
            iam.PolicyStatement(
                sid="DeploymentHostingScoped",
                actions=[
                    "sagemaker:CreateModel",
                    "sagemaker:CreateEndpoint",
                    "sagemaker:CreateEndpointConfig",
                    "sagemaker:UpdateEndpoint",
                    "sagemaker:DescribeEndpoint",
                    "sagemaker:DescribeEndpointConfig",
                    "sagemaker:DescribeModel",
                    "sagemaker:DeleteEndpoint",
                    "sagemaker:DeleteEndpointConfig",
                    "sagemaker:DeleteModel",
                    "sagemaker:InvokeEndpoint",
                    "sagemaker:CreateTransformJob",
                    "sagemaker:DescribeTransformJob",
                    "sagemaker:AddTags",
                ],
                resources=[ep, ep_cfg, model, transform],
            )
        )
        role.add_to_policy(self._scoped_lake_s3("DeploymentLakeS3"))
        role.add_to_policy(self._pass_self_to_sagemaker("DeploymentPassRole"))
        acknowledge_iam5_resources(
            role,
            {
                ep: "hosting operations are confined to project-prefixed endpoints",
                ep_cfg: "hosting operations are confined to project-prefixed endpoint configs",
                model: "hosting operations are confined to project-prefixed models",
                transform: "batch operations are confined to project-prefixed transform jobs",
                f"arn:aws:iam::{self.account}:role/{self.hyphen_project}-be-*": (
                    "PassRole is confined to project backend roles and conditioned on SageMaker"
                ),
                **self._lake_iam5_findings(),
            },
        )
        return role

    # ── MonitoringBackend role (Model Monitor + CloudWatch + Logs) ───────────────
    def _build_monitoring_role(self) -> iam.Role:
        role = iam.Role(
            self,
            "MonitoringBackendRole",
            role_name=f"{self.hyphen_project}-be-monitoring",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            description="Least-priv role for the MonitoringBackend managed adapter (M8 §8.7)",
        )
        sched = f"arn:aws:sagemaker:{self.region}:{self.account}:monitoring-schedule/{self.hyphen_project}-*"
        role.add_to_policy(
            iam.PolicyStatement(
                sid="ModelMonitorScoped",
                actions=[
                    "sagemaker:CreateMonitoringSchedule",
                    "sagemaker:DescribeMonitoringSchedule",
                    "sagemaker:UpdateMonitoringSchedule",
                    "sagemaker:DeleteMonitoringSchedule",
                    "sagemaker:StartMonitoringSchedule",
                    "sagemaker:StopMonitoringSchedule",
                ],
                resources=[sched],
            )
        )
        # CloudWatch PutMetricData has NO resource-level support → * but namespace-conditioned.
        role.add_to_policy(
            iam.PolicyStatement(
                sid="CloudWatchPutNamespaced",
                actions=["cloudwatch:PutMetricData"],
                resources=["*"],
                conditions={"StringEquals": {"cloudwatch:namespace": "Amlas/ML"}},
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="CloudWatchAlarmScoped",
                actions=["cloudwatch:PutMetricAlarm", "cloudwatch:DescribeAlarms", "cloudwatch:DeleteAlarms"],
                resources=[f"arn:aws:cloudwatch:{self.region}:{self.account}:alarm:{self.hyphen_project}-*"],
            )
        )
        logs_arn = (
            f"arn:aws:logs:{self.region}:{self.account}:log-group:/aws/sagemaker/*"
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="MonitoringLogsRead",
                actions=[
                    "logs:GetLogEvents",
                    "logs:FilterLogEvents",
                    "logs:DescribeLogStreams",
                    "logs:DescribeLogGroups",
                ],
                resources=[logs_arn],
            )
        )
        role.add_to_policy(self._scoped_lake_s3("MonitoringLakeS3"))
        acknowledge_iam5_resources(
            role,
            {
                "*": (  # nosemgrep
                    "CloudWatch PutMetricData has no resource-level IAM support and is "
                    "conditioned on the Amlas/ML namespace"
                ),
                sched: "monitoring operations are confined to project-prefixed schedules",
                f"arn:aws:cloudwatch:{self.region}:{self.account}:alarm:{self.hyphen_project}-*": (
                    "alarm operations are confined to project-prefixed alarms"
                ),
                logs_arn: "log reads are confined to this account's SageMaker log namespace",
                **self._lake_iam5_findings(),
            },
        )
        return role

    # ── shared statement builders ────────────────────────────────────────────────
    def _lake_iam5_findings(self) -> dict[str, str]:
        """Return evidence for the project/account-bounded lake bucket namespaces."""
        return {
            self._lake_bucket_arns[0]: (
                "bucket wildcard is limited to convention-named lake buckets for this account"
            ),
            self._lake_bucket_arns[1]: (
                "object wildcard is limited to convention-named lake buckets for this account"
            ),
        }

    def _scoped_lake_s3(self, sid: str) -> iam.PolicyStatement:
        """Scoped S3 read/write on the medallion lake bucket only (never a global ``s3:*``)."""
        return iam.PolicyStatement(
            sid=sid,
            actions=[
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket",
                "s3:GetBucketLocation",
            ],
            resources=self._lake_bucket_arns,
        )

    def _pass_self_to_sagemaker(self, sid: str) -> iam.PolicyStatement:
        """Service-conditioned ``iam:PassRole`` — only to SageMaker, only project roles."""
        return iam.PolicyStatement(
            sid=sid,
            actions=["iam:PassRole"],
            resources=[f"arn:aws:iam::{self.account}:role/{self.hyphen_project}-be-*"],
            conditions={"StringEquals": {"iam:PassedToService": "sagemaker.amazonaws.com"}},
        )

    def _emit_outputs(self) -> None:
        for label, role in (
            ("TrainingBackendRoleArn", self.training_role),
            ("DataQualityBackendRoleArn", self.dataquality_role),
            ("DeploymentBackendRoleArn", self.deployment_role),
            ("MonitoringBackendRoleArn", self.monitoring_role),
        ):
            CfnOutput(self, label, value=role.role_arn,
                      export_name=f"{self.hyphen_project}-{label}")
