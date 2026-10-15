"""AgentCoreRuntimeStack (NEW) — CDK-owned ECR repo for the agent runtime image.

Per design Part C.2.1: replaces the starter-toolkit's out-of-band ECR
(`ecr_auto_create: false` in `.bedrock_agentcore.yaml`) with a CDK-owned repo so
the runtime container image (`linux/arm64`) lives in IaC.

Resources:
    - aws_ecr.Repository `amlas-runtime`
        - image_scan_on_push=True
        - lifecycle rule: keep last 10 images (cost control)
        - removal_policy=DESTROY + empty_on_delete=True so `cdk destroy` is clean
          (no image is pushed until Phase 6; we favor a clean teardown over RETAIN
          for this de-mock phase — see teardown doc).
    - SSM param `/amlas/runtime-ecr-uri` → feeds RuntimeImageUri.
    - CfnOutput RepositoryUri / RepositoryArn / RepositoryName.

SEC-01: no hardcoded account IDs — repo URI is composed from CDK tokens
(`self.account` / `self.region`) only.
"""

from __future__ import annotations

from typing import Any

from aws_cdk import (
    CfnOutput,
    RemovalPolicy,
    Stack,
)
from aws_cdk import (
    aws_ecr as ecr,
)
from aws_cdk import (
    aws_ssm as ssm,
)
from constructs import Construct


class AgentCoreRuntimeStack(Stack):
    def __init__(
        self,
        scope: Construct,
        cid: str,
        *,
        hyphen_project: str = "amlas",
        ssm_prefix: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, cid, **kwargs)
        self.hyphen_project = hyphen_project
        self.ssm_prefix = ssm_prefix if ssm_prefix is not None else f"/{hyphen_project}"

        self.repository = ecr.Repository(
            self,
            "RuntimeRepository",
            repository_name=f"{hyphen_project}-runtime",
            image_scan_on_push=True,
            empty_on_delete=True,
            removal_policy=RemovalPolicy.DESTROY,
            lifecycle_rules=[
                ecr.LifecycleRule(
                    description="Keep only the last 10 images",
                    max_image_count=10,
                    rule_priority=1,
                )
            ],
        )

        # SSM: runtime image repo URI (consumed by the deploy of RuntimeStack)
        ssm.StringParameter(
            self,
            "RuntimeEcrUriParam",
            parameter_name=f"{self.ssm_prefix}/runtime-ecr-uri",
            string_value=self.repository.repository_uri,
            description="ECR repository URI for the AgentCore runtime image (linux/arm64)",
        )

        self._emit_outputs()

    def _emit_outputs(self) -> None:
        CfnOutput(
            self,
            "RepositoryUri",
            value=self.repository.repository_uri,
            export_name=f"{self.hyphen_project}-runtime-ecr-uri",
        )
        CfnOutput(
            self,
            "RepositoryArn",
            value=self.repository.repository_arn,
            export_name=f"{self.hyphen_project}-runtime-ecr-arn",
        )
        CfnOutput(
            self,
            "RepositoryName",
            value=self.repository.repository_name,
            export_name=f"{self.hyphen_project}-runtime-ecr-name",
        )
