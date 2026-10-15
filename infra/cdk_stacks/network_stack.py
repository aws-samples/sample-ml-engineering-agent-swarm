"""NetworkStack (NEW, R-010 / AMLAS-SEC-003 / BSC54) — PrivateLink VPC endpoints.

Provisions network isolation so AWS service traffic is confined to the AWS
network (not public regional endpoints), satisfying **BSC54** and strengthening
**BSC16** (encrypt-in-transit / network isolation).

Resources:
    - aws_ec2.Vpc (2 AZs, private-isolated + private-with-egress subnets) — the
      private subnets host SageMaker training/processing jobs so they have no
      direct internet path.
    - Interface VPC endpoints (PrivateLink) for the control/data-plane services the
      platform calls: bedrock-runtime, bedrock-agentcore, sagemaker.api,
      sagemaker.runtime, secretsmanager, sts, ecr.api, ecr.dkr, logs, monitoring.
      Each carries a restrictive **endpoint policy** scoped to this account.
    - Gateway VPC endpoints for s3 + dynamodb (no hourly cost; route-table based).
    - SSM params publishing the VPC id + subnet ids for consuming stacks/jobs.

SEC-01: NO hardcoded account IDs — the endpoint policies scope `aws:PrincipalAccount`
to `self.account` (a CDK token), never a literal.

Design note (minimal-viable scaffolding): this stack stands up the VPC + endpoints
+ scoped policies as IaC so BSC54 is satisfied and the constructs are testable via
`cdk synth`. Wiring the existing serverless workloads (AgentCore runtime / SageMaker
jobs) to run *inside* this VPC is a follow-up integration step tracked in
REMEDIATION_PLAN.md R-010 (it depends on AgentCore VPC-mode support, R-009).
"""

from __future__ import annotations

from typing import Any, ClassVar

from aws_cdk import (
    CfnOutput,
    RemovalPolicy,
    Stack,
)
from aws_cdk import (
    aws_ec2 as ec2,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_logs as logs,
)
from aws_cdk import (
    aws_ssm as ssm,
)
from constructs import Construct

from ._kms import make_data_key

_VPC_CIDR = "10.0.0.0/16"

# Interface endpoints (PrivateLink) for the services the platform calls. Names are
# the service short-names; `ec2.InterfaceVpcEndpointAwsService` builds the full
# `com.amazonaws.<region>.<service>` name per the deploy region.
_INTERFACE_SERVICES: dict[str, ec2.InterfaceVpcEndpointAwsService] = {
    "bedrock-runtime": ec2.InterfaceVpcEndpointAwsService.BEDROCK_RUNTIME,
    "sagemaker-api": ec2.InterfaceVpcEndpointAwsService.SAGEMAKER_API,
    "sagemaker-runtime": ec2.InterfaceVpcEndpointAwsService.SAGEMAKER_RUNTIME,
    "secretsmanager": ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER,
    "sts": ec2.InterfaceVpcEndpointAwsService.STS,
    "ecr-api": ec2.InterfaceVpcEndpointAwsService.ECR,
    "ecr-dkr": ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER,
    "logs": ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS,
    "monitoring": ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_MONITORING,
}


class NetworkStack(Stack):
    def __init__(
        self,
        scope: Construct,
        cid: str,
        *,
        hyphen_project: str = "amlas",
        ssm_prefix: str | None = None,
        max_azs: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, cid, **kwargs)
        self.hyphen_project = hyphen_project
        self.ssm_prefix = ssm_prefix if ssm_prefix is not None else f"/{hyphen_project}"
        available_zones = self.availability_zones
        if len(available_zones) < max_azs:
            raise ValueError(
                f"NetworkStack requires {max_azs} availability zones in {self.region}; "
                f"CDK resolved {len(available_zones)}. Choose a region with enough available zones."
            )

        self.flow_log_key = make_data_key(
            self,
            "FlowLogKey",
            alias=f"alias/{hyphen_project}-vpc-flow-logs",
            for_logs=True,
            enable_rotation=True,  # explicit at call site (default also True)
            description="Amlas VPC flow-log CMK (rotation on)",
        )
        self.vpc = self._build_vpc(max_azs)
        self.endpoint_security_group = self._build_endpoint_security_group()
        self.flow_log_group = self._build_flow_log_group()
        self.flow_log = self.vpc.add_flow_log(
            "VpcFlowLog",
            destination=ec2.FlowLogDestination.to_cloud_watch_logs(self.flow_log_group),
            traffic_type=ec2.FlowLogTrafficType.ALL,
        )
        self.interface_endpoints = self._build_interface_endpoints()
        self._build_gateway_endpoints()

        self._write_ssm_params()
        self._emit_outputs()

    # ------------------------------------------------------------------
    # VPC
    # ------------------------------------------------------------------

    def _build_vpc(self, max_azs: int) -> ec2.Vpc:
        return ec2.Vpc(
            self,
            "Vpc",
            ip_addresses=ec2.IpAddresses.cidr(_VPC_CIDR),
            max_azs=max_azs,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="private-isolated",
                    subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    name="private-egress",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    name="public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
            ],
        )

    def _build_endpoint_security_group(self) -> ec2.SecurityGroup:
        group = ec2.SecurityGroup(
            self,
            "InterfaceEndpointSecurityGroup",
            vpc=self.vpc,
            allow_all_outbound=False,
            description="TLS access to AMLAS PrivateLink endpoints from the VPC",
        )
        group.add_ingress_rule(
            ec2.Peer.ipv4(_VPC_CIDR),
            ec2.Port.tcp(443),
            "TLS from workloads inside the AMLAS VPC",
        )
        return group

    def _build_flow_log_group(self) -> logs.LogGroup:
        return logs.LogGroup(
            self,
            "VpcFlowLogGroup",
            log_group_name=f"/aws/vpc-flow-logs/{self.hyphen_project}",
            retention=logs.RetentionDays.THREE_MONTHS,
            encryption_key=self.flow_log_key,
            removal_policy=RemovalPolicy.RETAIN,
        )

    # ------------------------------------------------------------------
    # Endpoints + scoped policies
    # ------------------------------------------------------------------

    # Security review (network_stack HIGH x3): per-service least-privilege
    # endpoint-policy actions. Each endpoint grants ONLY the data-plane actions
    # the platform uses through it (plus the same-account principal condition),
    # replacing the previous single `Action:*` statement shared by all endpoints.
    _ENDPOINT_ACTIONS: ClassVar[dict[str, list[str]]] = {
        "bedrock-runtime": [
            "bedrock:InvokeModel",
            "bedrock:InvokeModelWithResponseStream",
        ],
        "sagemaker-api": [
            "sagemaker:CreateTrainingJob",
            "sagemaker:CreateProcessingJob",
            "sagemaker:CreateModel",
            "sagemaker:CreateEndpoint",
            "sagemaker:CreateEndpointConfig",
            "sagemaker:Describe*",
            "sagemaker:List*",
            "sagemaker:StopTrainingJob",
            "sagemaker:StopProcessingJob",
            "sagemaker:DeleteModel",
            "sagemaker:DeleteEndpoint",
            "sagemaker:DeleteEndpointConfig",
            "sagemaker:AddTags",
        ],
        "sagemaker-runtime": ["sagemaker:InvokeEndpoint"],
        "secretsmanager": [
            "secretsmanager:GetSecretValue",
            "secretsmanager:DescribeSecret",
        ],
        "sts": ["sts:AssumeRole", "sts:GetCallerIdentity"],
        "ecr-api": [
            "ecr:GetAuthorizationToken",
            "ecr:BatchCheckLayerAvailability",
            "ecr:BatchGetImage",
        ],
        "ecr-dkr": [
            "ecr:GetDownloadUrlForLayer",
            "ecr:BatchGetImage",
            "ecr:BatchCheckLayerAvailability",
        ],
        "logs": [
            "logs:CreateLogGroup",
            "logs:CreateLogStream",
            "logs:PutLogEvents",
            "logs:DescribeLogGroups",
            "logs:DescribeLogStreams",
        ],
        "monitoring": ["cloudwatch:PutMetricData", "cloudwatch:GetMetricData"],
        # Gateway endpoints:
        "s3": [
            "s3:GetObject",
            "s3:PutObject",
            "s3:DeleteObject",
            "s3:ListBucket",
            "s3:GetBucketLocation",
            "s3:AbortMultipartUpload",
            "s3:ListMultipartUploadParts",
        ],
        "dynamodb": [
            "dynamodb:GetItem",
            "dynamodb:PutItem",
            "dynamodb:UpdateItem",
            "dynamodb:DeleteItem",
            "dynamodb:Query",
            "dynamodb:BatchGetItem",
            "dynamodb:BatchWriteItem",
            "dynamodb:DescribeTable",
        ],
    }

    def _account_scoped_statement(self, endpoint_name: str) -> iam.PolicyStatement:
        """Endpoint policy statement: same-account principals + per-service actions.

        SEC-01: `aws:PrincipalAccount` is conditioned on `self.account` (a CDK
        token), confining the endpoint to project-owned use and blocking
        cross-account data exfiltration through the PrivateLink path.
        Least-privilege (Security review): actions are the per-service
        allow-list from :data:`_ENDPOINT_ACTIONS` — never ``*``. Resources stay
        ``*`` because the endpoint policy is a NETWORK guardrail; identity
        policies (e.g. the P6 runtime role) provide the resource-level scoping.
        """
        actions = self._ENDPOINT_ACTIONS.get(endpoint_name)
        if not actions:
            raise ValueError(f"no endpoint action allow-list for {endpoint_name!r}")
        return iam.PolicyStatement(
            sid="AllowSameAccountScopedActions",
            effect=iam.Effect.ALLOW,
            principals=[iam.AnyPrincipal()],
            actions=actions,
            resources=["*"],
            conditions={"StringEquals": {"aws:PrincipalAccount": self.account}},
        )

    def _build_interface_endpoints(self) -> dict[str, ec2.InterfaceVpcEndpoint]:
        endpoints: dict[str, ec2.InterfaceVpcEndpoint] = {}
        for name, service in _INTERFACE_SERVICES.items():
            ep = self.vpc.add_interface_endpoint(
                f"Endpoint-{name}",
                service=service,
                open=False,
                private_dns_enabled=True,
                security_groups=[self.endpoint_security_group],
                subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED),
            )
            # SEC-01 / R-010: same-account access + per-service action allow-list.
            ep.add_to_policy(self._account_scoped_statement(name))
            endpoints[name] = ep
        return endpoints

    def _build_gateway_endpoints(self) -> None:
        # Gateway endpoints (S3 + DynamoDB) — route-table based, no hourly cost.
        # Scoped to same-account access via endpoint policy.
        s3_ep = self.vpc.add_gateway_endpoint(
            "Endpoint-s3",
            service=ec2.GatewayVpcEndpointAwsService.S3,
            subnets=[ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED)],
        )
        s3_ep.add_to_policy(self._account_scoped_statement("s3"))
        ddb_ep = self.vpc.add_gateway_endpoint(
            "Endpoint-dynamodb",
            service=ec2.GatewayVpcEndpointAwsService.DYNAMODB,
            subnets=[ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED)],
        )
        ddb_ep.add_to_policy(self._account_scoped_statement("dynamodb"))

    # ------------------------------------------------------------------
    # Outputs / SSM
    # ------------------------------------------------------------------

    def _write_ssm_params(self) -> None:
        ssm.StringParameter(
            self,
            "VpcIdParam",
            parameter_name=f"{self.ssm_prefix}/vpc-id",
            string_value=self.vpc.vpc_id,
            description="VPC id hosting PrivateLink endpoints for the ML platform",
        )
        private_subnet_ids = ",".join(
            self.vpc.select_subnets(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED).subnet_ids
        )
        ssm.StringParameter(
            self,
            "PrivateSubnetIdsParam",
            parameter_name=f"{self.ssm_prefix}/private-subnet-ids",
            string_value=private_subnet_ids,
            description="Comma-separated private-isolated subnet ids for SageMaker jobs",
        )

    def _emit_outputs(self) -> None:
        CfnOutput(
            self,
            "VpcId",
            value=self.vpc.vpc_id,
            export_name=f"{self.hyphen_project}-vpc-id",
        )
        CfnOutput(
            self,
            "InterfaceEndpointCount",
            value=str(len(self.interface_endpoints)),
            export_name=f"{self.hyphen_project}-interface-endpoint-count",
        )
