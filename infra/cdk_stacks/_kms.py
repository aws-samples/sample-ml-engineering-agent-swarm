"""Shared customer-managed KMS (CMK) helper — WA-01 / WA-04 (data protection).

The audit's single highest-severity systemic finding was that **no customer-managed
KMS key (CMK) existed anywhere in `infra/`** — every S3 bucket / DynamoDB table used
AWS-managed (SSE-S3 / `AWS_MANAGED`) encryption, with no key rotation / key-policy
segregation. This module is the one place the remediation creates (or imports) a CMK
so every data stack encrypts at rest with a rotation-enabled customer-managed key.

Portability / OSS-adopter seam (the audit §8 open decision #1 — CMK strategy):
    * **Secure default (ON):** each data stack creates its OWN `kms.Key` with
      ``enable_key_rotation=True`` and ``RemovalPolicy.RETAIN`` (a key is the system
      of record for the data it protects — never auto-delete).
    * **Key reuse / BYO key:** when an adopter supplies a key ARN (via the
      ``AMLAS_KMS_KEY_ARN`` env read in :mod:`infra.app`, threaded as the
      ``kms_key_arn`` kwarg), ALL stacks IMPORT that single shared/BYO key instead of
      minting their own (the "single shared CMK acceptable for v1" decision). No
      account/org-specific value is ever hardcoded — the ARN is supplied by the adopter.

CloudWatch Logs note: a log group encrypted with a CMK requires the key policy to grant
the regional CloudWatch Logs service principal; :func:`make_data_key` adds that grant
(scoped by the ``kms:EncryptionContext`` to this account's log ARNs) when ``for_logs=True``.

SEC-01: every ARN/condition is composed from CDK tokens (``stack.account`` /
``stack.region``) — never a hardcoded literal.
"""

from __future__ import annotations

from aws_cdk import (
    RemovalPolicy,
    Stack,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_kms as kms,
)


def make_data_key(
    stack: Stack,
    cid: str,
    *,
    key_arn: str = "",
    alias: str | None = None,
    enable_rotation: bool = True,
    for_logs: bool = False,
    description: str = "Amlas customer-managed CMK (rotation on) — data-at-rest",
) -> kms.IKey:
    """Return a customer-managed CMK for a data stack — create by default, import if BYO.

    Args:
        stack: the owning CDK stack (``self``).
        cid: construct id for the created/imported key.
        key_arn: an adopter-supplied existing key ARN. When non-empty the key is
            IMPORTED (the shared/BYO-key reuse seam); when empty a NEW rotation-enabled
            CMK is created (the secure default).
        alias: optional KMS alias for the created key (ignored on import).
        enable_rotation: annual key rotation — defaults ON (the secure default).
        for_logs: when True, attach the CloudWatch-Logs service-principal key-policy
            grant required to encrypt a ``logs.LogGroup`` with this CMK.
        description: created-key description.

    Returns:
        A ``kms.IKey`` usable as ``encryption_key=`` on S3 / DDB / Logs / SNS.
    """
    if key_arn:
        # Reuse / BYO: import the adopter-supplied key (single shared CMK for v1).
        return kms.Key.from_key_arn(stack, cid, key_arn)

    key = kms.Key(
        stack,
        cid,
        description=description,
        enable_key_rotation=enable_rotation,
        alias=alias,
        # A CMK is the system of record for the data it protects; retain on teardown
        # so a `cdk destroy` can never strand encrypted data behind a deleted key.
        removal_policy=RemovalPolicy.RETAIN,
    )
    if for_logs:
        # CloudWatch Logs encryption with a CMK requires the regional Logs service
        # principal to be granted on the KEY POLICY (CDK does NOT add this implicitly).
        # `resources=["*"]` in a KEY POLICY refers to the key itself (the standard
        # idiom) and is NOT subject to the BackendIamStack no-Resource:"*" CI gate.
        key.add_to_resource_policy(
            iam.PolicyStatement(
                sid="AllowCloudWatchLogs",
                effect=iam.Effect.ALLOW,
                principals=[iam.ServicePrincipal(f"logs.{stack.region}.amazonaws.com")],
                actions=[
                    "kms:Encrypt",
                    "kms:Decrypt",
                    "kms:ReEncrypt*",
                    "kms:GenerateDataKey*",
                    "kms:DescribeKey",
                ],
                resources=["*"],
                conditions={
                    "ArnLike": {
                        "kms:EncryptionContext:aws:logs:arn": (
                            f"arn:aws:logs:{stack.region}:{stack.account}:*"
                        )
                    }
                },
            )
        )
    return key
