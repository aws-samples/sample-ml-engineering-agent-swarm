"""Shared secure S3 constructs for independently deployable stacks."""

from __future__ import annotations

from aws_cdk import Acknowledgment, Duration, RemovalPolicy, Validations
from aws_cdk import aws_s3 as s3
from constructs import Construct


def make_access_log_bucket(
    scope: Construct,
    construct_id: str = "AccessLogBucket",
) -> s3.Bucket:
    """Create a retained, private sink for S3 server-access logs.

    The sink intentionally does not log to itself: recursive S3 access logging is
    unsupported and would create an unbounded log-delivery loop. Every data bucket
    using this sink must configure ``server_access_logs_bucket`` explicitly.
    """
    bucket = s3.Bucket(
        scope,
        construct_id,
        encryption=s3.BucketEncryption.S3_MANAGED,
        versioned=True,
        enforce_ssl=True,
        block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
        object_ownership=s3.ObjectOwnership.OBJECT_WRITER,
        removal_policy=RemovalPolicy.RETAIN,
        lifecycle_rules=[
            s3.LifecycleRule(
                id="ExpireAccessLogs",
                expiration=Duration.days(365),
                noncurrent_version_expiration=Duration.days(90),
                abort_incomplete_multipart_upload_after=Duration.days(7),
            )
        ],
    )
    Validations.of(bucket).acknowledge(
        Acknowledgment(
            id="AwsSolutions-S1",
            reason=(
                "This dedicated access-log sink must not recursively deliver "
                "server-access logs to itself."
            ),
        )
    )
    return bucket
