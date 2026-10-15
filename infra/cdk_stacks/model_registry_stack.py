"""ModelRegistryStack (NEW) — governed SageMaker Model Package Group.

Per design Part C.2.4. The design names it `amlas_models`, but a LIVE probe of
the default SageMaker API rejected underscores:

    ValidationException: Value 'wbprobe_underscore_x' at 'modelPackageGroupName'
    failed to satisfy constraint: Member must satisfy regular expression pattern:
    [a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}

So we deploy the registry group as **`amlas-models`** (hyphenated) — the real,
API-valid name — and publish it to SSM/env so `MLOpsConfig.model_package_group`
reads the deployed value rather than the invalid underscore form.

Resources:
    - aws_sagemaker.CfnModelPackageGroup `amlas-models`
      (removal_policy=RETAIN — registry is the governance system of record;
       `cdk destroy` leaves it. group create is effectively idempotent/upsert.)
    - SSM param `/amlas/model-package-group`.
    - CfnOutput ModelPackageGroupName / ModelPackageGroupArn.

SEC-01: ARN built from CDK tokens only.
"""

from __future__ import annotations

from typing import Any

from aws_cdk import (
    CfnOutput,
    RemovalPolicy,
    Stack,
)
from aws_cdk import (
    aws_sagemaker as sagemaker,
)
from aws_cdk import (
    aws_ssm as ssm,
)
from constructs import Construct


class ModelRegistryStack(Stack):
    def __init__(
        self,
        scope: Construct,
        cid: str,
        *,
        hyphen_project: str = "amlas",
        ssm_prefix: str | None = None,
        model_package_group_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, cid, **kwargs)
        self.hyphen_project = hyphen_project
        self.ssm_prefix = ssm_prefix if ssm_prefix is not None else f"/{hyphen_project}"
        self.model_package_group_name = (
            model_package_group_name if model_package_group_name is not None else f"{hyphen_project}-models"
        )

        self.model_package_group = sagemaker.CfnModelPackageGroup(
            self,
            "ModelPackageGroup",
            model_package_group_name=self.model_package_group_name,
            model_package_group_description=(
                "Amlas governed model registry — versioned model packages "
                "with approval workflow (PendingManualApproval → Approved)."
            ),
        )
        self.model_package_group.apply_removal_policy(RemovalPolicy.RETAIN)

        ssm.StringParameter(
            self,
            "ModelPackageGroupParam",
            parameter_name=f"{self.ssm_prefix}/model-package-group",
            string_value=self.model_package_group_name,
            description="SageMaker Model Registry package group name (API-valid, hyphenated)",
        )

        self._emit_outputs()

    def _emit_outputs(self) -> None:
        CfnOutput(
            self,
            "ModelPackageGroupName",
            value=self.model_package_group_name,
            export_name=f"{self.hyphen_project}-model-package-group",
        )
        CfnOutput(
            self,
            "ModelPackageGroupArn",
            value=(
                f"arn:aws:sagemaker:{self.region}:{self.account}:"
                f"model-package-group/{self.model_package_group_name}"
            ),
            export_name=f"{self.hyphen_project}-model-package-group-arn",
        )
