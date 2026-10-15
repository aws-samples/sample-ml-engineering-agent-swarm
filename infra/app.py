"""CDK app entry point.

Standing foundational stacks (Phase 2 real-infra — low/zero idle cost):
    - AgentCoreRuntimeStack     — ECR repo for the runtime image
    - LakeStack                 — medallion S3 bucket + Glue DB
    - ModelRegistryStack        — SageMaker model-package-group (amlas-models)
    - SecretsStack              — Tavily secret (imported) + SSM id

Existing stacks:
    - RuntimeStack       — full stack, requires RuntimeImageUri parameter
                                  (NOT deployed in Phase 2 — needs the built image, Phase 6).
    - AsyncTrainingStack  — async-training plane (already deployed in default).

Deploy (standing foundational — Phase 2):
    AWS_PROFILE=default uv run cdk synth
    AWS_PROFILE=default uv run cdk deploy \\
        AgentCoreRuntimeStack LakeStack ModelRegistryStack SecretsStack \\
        --require-approval never
"""

from __future__ import annotations

import os

from aws_cdk import App, Environment, Tags, Validations

from amlas.config._env_compat import env_with_compat
from amlas.config.aws_region import resolve_aws_region
from amlas.config.deployment import deployment_stack_name, resolve_deployment_prefix, resolve_ssm_prefix

from .cdk_stacks import (
    AgentCoreRuntimeStack,
    AsyncTrainingStack,
    BackendIamStack,
    DetectionsStack,
    GuardrailStack,
    HarnessStack,
    LakeStack,
    ModelRegistryStack,
    NetworkStack,
    RuntimeStack,
    SecretsStack,
)

deployment_prefix = resolve_deployment_prefix()
ssm_prefix = resolve_ssm_prefix()
tavily_secret_id = os.environ.get("TAVILY_SECRET_ID", "").strip() or None
model_package_group_name = env_with_compat("AMLAS_MODEL_PACKAGE_GROUP", "").strip() or None
app = App()
Tags.of(app).add("amlas:deployment-prefix", deployment_prefix)
deployment_id = os.environ.get("AMLAS_DEPLOYMENT_ID", "").strip()
if deployment_id:
    Tags.of(app).add("amlas:deployment-id", deployment_id)
# cdk-nag v3 runs as a native CDK policy-validation
# plugin. Gated on AMLAS_CDK_NAG so normal deploys are unaffected; set
# AMLAS_CDK_NAG=1 to run the security-review synth and emit its report.
if os.environ.get("AMLAS_CDK_NAG", "").strip().lower() in ("1", "true", "yes", "on"):
    # Import lazily so normal synths do not initialize the review-only plugin.
    from cdk_nag import AwsSolutionsChecks

    Validations.of(app).add_plugins(
        AwsSolutionsChecks(
            app,
            verbose=True,
            write_suppressions_to_cloud_formation=True,
        )
    )
env = Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=resolve_aws_region(os.environ.get("CDK_DEFAULT_REGION"), required=False),
)

# WA-01 (customer-managed KMS) reuse / BYO-key seam (audit §8 open decision #1):
# by DEFAULT each data stack mints its OWN rotation-enabled CMK (the secure default).
# An adopter who wants a SINGLE shared / bring-your-own CMK supplies its ARN here via
# ``AMLAS_KMS_KEY_ARN`` — every data stack then IMPORTS that one key instead of
# creating new ones. Empty default = "create per-stack CMKs" (no account-specific
# value hardcoded; the ARN, if any, is supplied entirely by the adopter's environment).
kms_key_arn = env_with_compat("AMLAS_KMS_KEY_ARN", "").strip()
# Guard against an UNRESOLVED template placeholder (or any non-ARN value) being read as
# a real BYO key. `.env.example` historically shipped a placeholder that LOOKS like an
# ARN (``arn:aws:kms:us-west-2:<AWS_ACCOUNT_ID>:key/<KMS_KEY_ID>``); after ``cp
# .env.example .env`` that non-empty string was imported as a BYO CMK, so every data
# stack referenced a NON-EXISTENT key — DetectionsStack failed CFN LogGroup ``KmsKeyId``
# validation (``#/KmsKeyId: failed constraint [pattern]`` + "KMS Key Id could not be
# found") and Lake/Async silently pointed at a bogus key. A real ARN never contains
# ``<``; treat any value with ``<`` or that doesn't start with ``arn:aws:kms:`` as empty
# so each stack mints its OWN rotation-enabled CMK (the secure default).
if "<" in kms_key_arn or not kms_key_arn.startswith("arn:aws:kms:"):
    kms_key_arn = ""

# WA-05 (Security/Network) runtime NetworkMode seam: default PUBLIC because AgentCore VPC-mode
# is the external blocker (audit §7 / R-009). An adopter flips this to a private mode via
# ``AMLAS_RUNTIME_NETWORK_MODE`` once VPC-mode is available — no code change.
runtime_network_mode = env_with_compat("AMLAS_RUNTIME_NETWORK_MODE", "PUBLIC")

# WA-04 (privacy) OTEL message-content capture seam: redact-by-default (OFF). An adopter
# deliberately enables full prompt/response capture via ``AMLAS_CAPTURE_CONTENT=true`` after a
# data-governance decision on retention/redaction (audit §8 decision #3).
capture_content = env_with_compat("AMLAS_CAPTURE_CONTENT", "").strip().lower() in ("1", "true", "yes", "on")


def stack_options(logical_id: str) -> dict:
    """Keep CLI logical IDs stable while isolating physical deployments."""
    options = {
        "env": env,
        "hyphen_project": deployment_prefix,
        "stack_name": deployment_stack_name(logical_id, deployment_prefix),
    }
    if logical_id in {
        "AgentCoreRuntimeStack",
        "LakeStack",
        "ModelRegistryStack",
        "SecretsStack",
        "NetworkStack",
        "GuardrailStack",
        "HarnessStack",
    }:
        options["ssm_prefix"] = ssm_prefix
    return options


# --- Standing foundational stacks (Phase 2 — negligible idle cost) ---
AgentCoreRuntimeStack(app, "AgentCoreRuntimeStack", **stack_options("AgentCoreRuntimeStack"))
# Use the resolved deployment region; LakeStack also derives it from its env
# for direct callers, preserving environment-agnostic CloudFormation tokens.
LakeStack(app, "LakeStack", region_name=env.region, kms_key_arn=kms_key_arn, **stack_options("LakeStack"))
ModelRegistryStack(
    app,
    "ModelRegistryStack",
    model_package_group_name=model_package_group_name,
    **stack_options("ModelRegistryStack"),
)
SecretsStack(
    app,
    "SecretsStack",
    tavily_secret_id=tavily_secret_id,
    **stack_options("SecretsStack"),
)

# --- Per-backend least-privilege IAM (synth-only; not in the deploy set) ---
BackendIamStack(app, "BackendIamStack", **stack_options("BackendIamStack"))

# --- Phase 3 security stacks (remediation) ---
# NetworkStack (R-010 / BSC54): VPC + PrivateLink interface/gateway endpoints with
#   account-scoped endpoint policies. DetectionsStack (R-011): CloudTrail trail +
#   security metric-filter alarms. Both are standalone IaC; wiring existing
#   serverless workloads into the VPC is the R-009/R-010 follow-up.
NetworkStack(app, "NetworkStack", **stack_options("NetworkStack"))
DetectionsStack(app, "DetectionsStack", kms_key_arn=kms_key_arn, **stack_options("DetectionsStack"))

# GuardrailStack (WA-02 / AGENTSEC04+08): provisions the Bedrock Guardrail bound in-band
# on every model turn (config.guardrails reads its id/version from SSM). Conservative
# responsible-AI defaults (HIGH content filters, PII anonymize) are CfnParameter-tunable.
GuardrailStack(app, "GuardrailStack", **stack_options("GuardrailStack"))

# HarnessStack (ADR-0007 phase 1, T-HARN-13): one AWS::BedrockAgentCore::Harness per
# harness-hosted Lead, built from its offline HarnessSpec, with each harness id published to
# SSM under /amlas/harness/. Its execution role is where ADR-0002 is enforced — no
# InvokeAgentRuntimeCommand and no code-interpreter actions, ever.
if os.environ.get("AMLAS_SKIP_HARNESS_STACK", "").strip().lower() not in ("1", "true", "yes", "on"):
    HarnessStack(app, "HarnessStack", **stack_options("HarnessStack"))

# --- Existing stacks ---
RuntimeStack(
    app,
    "RuntimeStack",
    project_name=deployment_prefix.replace("-", "_"),
    tavily_secret_id=tavily_secret_id,
    model_package_group_name=model_package_group_name,
    **stack_options("RuntimeStack"),
    kms_key_arn=kms_key_arn,
    runtime_network_mode=runtime_network_mode,
    capture_content=capture_content,
)
AsyncTrainingStack(app, "AsyncTrainingStack", kms_key_arn=kms_key_arn, **stack_options("AsyncTrainingStack"))

app.synth()
