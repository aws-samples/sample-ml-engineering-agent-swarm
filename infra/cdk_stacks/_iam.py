"""Least-privilege IAM grants with explicit action names.

CDK's convenience grant helpers intentionally use action-name wildcards such as
``s3:GetObject*`` and ``kms:GenerateDataKey*``.  These helpers keep the same
resource boundaries while spelling out only the APIs used by AMLAS workloads.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from aws_cdk import CfnResource, Stack, Validations
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_iam as iam
from aws_cdk import aws_kms as kms
from aws_cdk import aws_s3 as s3
from constructs import Construct

_S3_BUCKET_READ_WRITE_ACTIONS = [
    "s3:GetBucketLocation",
    "s3:ListBucket",
    "s3:ListBucketMultipartUploads",
]
_S3_OBJECT_READ_WRITE_ACTIONS = [
    "s3:AbortMultipartUpload",
    "s3:DeleteObject",
    "s3:GetObject",
    "s3:ListMultipartUploadParts",
    "s3:PutObject",
]
_DYNAMODB_READ_WRITE_ACTIONS = [
    "dynamodb:BatchGetItem",
    "dynamodb:BatchWriteItem",
    "dynamodb:ConditionCheckItem",
    "dynamodb:DeleteItem",
    "dynamodb:DescribeTable",
    "dynamodb:GetItem",
    "dynamodb:PutItem",
    "dynamodb:Query",
    "dynamodb:Scan",
    "dynamodb:UpdateItem",
]
_KMS_ENCRYPT_DECRYPT_ACTIONS = [
    "kms:Decrypt",
    "kms:DescribeKey",
    "kms:Encrypt",
    "kms:GenerateDataKey",
    "kms:GenerateDataKeyWithoutPlaintext",
    "kms:ReEncryptFrom",
    "kms:ReEncryptTo",
]


def cfn_getatt_arn(resource: Construct, *, suffix: str = "") -> str:
    """Return cdk-nag's canonical ``<LogicalId.Arn>`` rendering for an L2."""
    cfn_resource = resource.node.default_child
    if not isinstance(cfn_resource, CfnResource):
        raise TypeError(f"{resource.node.path} has no CfnResource default child")
    logical_id = Stack.of(resource).get_logical_id(cfn_resource)
    return f"<{logical_id}.Arn>{suffix}"


def _render_cdk_nag_token(value: object) -> str | None:
    """Render CDK token fragments the same way cdk-nag names IAM findings."""
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return None

    ref = value.get("Ref")
    if len(value) == 1 and isinstance(ref, str):
        return f"<{ref}>"

    get_att = value.get("Fn::GetAtt")
    if (
        len(value) == 1
        and isinstance(get_att, list)
        and all(isinstance(part, str) for part in get_att)
    ):
        return f"<{'.'.join(get_att)}>"

    join = value.get("Fn::Join")
    if len(value) == 1 and isinstance(join, list) and len(join) == 2:
        delimiter, fragments = join
        if isinstance(delimiter, str) and isinstance(fragments, list):
            rendered = [_render_cdk_nag_token(fragment) for fragment in fragments]
            if all(fragment is not None for fragment in rendered):
                return delimiter.join(fragment for fragment in rendered if fragment is not None)
    return None


def _canonical_finding_resource(scope: Construct, resource: str) -> str:
    """Render a token-bearing IAM resource as cdk-nag's static finding text."""
    resolved = Stack.of(scope).resolve(resource)
    rendered = _render_cdk_nag_token(resolved)
    if rendered is not None:
        return rendered
    return json.dumps(resolved, separators=(",", ":"))


def acknowledge_iam5_resources(
    scope: Construct,
    findings: Mapping[str, str],
) -> None:
    """Acknowledge exact residual IAM5 resource patterns with review evidence.

    CDK 2.261 rejects granular cdk-nag IDs through ``Validations.acknowledge``
    because both APIs reserve ``::`` differently. cdk-nag 3.0.1 reads the
    supported acknowledged-rules metadata directly and removes the native
    ``annotation::`` qualifier before matching. Recording the same metadata
    explicitly preserves exact finding IDs and enables its CloudFormation
    evidence writer without broad rule suppression.
    """
    acknowledgments = {}
    for resource, reason in findings.items():
        canonical_resource = _canonical_finding_resource(scope, resource)
        acknowledgments[
            f"annotation::AwsSolutions-IAM5[Resource::{canonical_resource}]"
        ] = reason
    scope.node.add_metadata(
        Validations.ACKNOWLEDGED_RULES_METADATA_KEY,
        acknowledgments,
    )


def grant_key_encrypt_decrypt(role: iam.Role, key: kms.IKey) -> None:
    """Allow the exact KMS data-plane APIs required for encrypted data stores."""
    role.add_to_policy(
        iam.PolicyStatement(
            sid="ExplicitKmsEncryptDecrypt",
            actions=_KMS_ENCRYPT_DECRYPT_ACTIONS,
            resources=[key.key_arn],
        )
    )


def grant_bucket_read_write(
    role: iam.Role,
    bucket: s3.IBucket,
    *,
    encryption_key: kms.IKey | None = None,
) -> None:
    """Grant bucket/object read-write without wildcard action names."""
    role.add_to_policy(
        iam.PolicyStatement(
            sid="ExplicitBucketReadWrite",
            actions=_S3_BUCKET_READ_WRITE_ACTIONS,
            resources=[bucket.bucket_arn],
        )
    )
    role.add_to_policy(
        iam.PolicyStatement(
            sid="ExplicitObjectReadWrite",
            actions=_S3_OBJECT_READ_WRITE_ACTIONS,
            resources=[bucket.arn_for_objects("*")],
        )
    )
    if encryption_key is not None:
        grant_key_encrypt_decrypt(role, encryption_key)


def grant_table_read_write_data(
    role: iam.Role,
    table: dynamodb.ITable,
    *,
    encryption_key: kms.IKey | None = None,
    index_names: tuple[str, ...] = ("SessionIdIndex",),
) -> None:
    """Grant exact DynamoDB item/query APIs on a table and its named indexes."""
    role.add_to_policy(
        iam.PolicyStatement(
            sid="ExplicitTableReadWrite",
            actions=_DYNAMODB_READ_WRITE_ACTIONS,
            resources=[
                table.table_arn,
                *(f"{table.table_arn}/index/{name}" for name in index_names),
            ],
        )
    )
    if encryption_key is not None:
        grant_key_encrypt_decrypt(role, encryption_key)
