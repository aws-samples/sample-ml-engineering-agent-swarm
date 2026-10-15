"""AWS health check — uses boto3 standard credential chain to verify connectivity.

Does NOT import boto3 at module level so callers can run without it installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .aws_region import resolve_aws_region
from .settings import Settings


@dataclass
class AWSHealthStatus:
    """Result of an AWS connectivity health check."""

    aws_connected: bool = False
    account_id: str = ""
    arn: str = ""
    bedrock_available: bool = False
    s3_available: bool = False
    s3_bucket_accessible: bool = False
    profile_used: str = "default chain"
    region: str = ""
    bedrock_region: str = ""
    error: str = ""
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def summary_line(self) -> str:
        """One-line summary for TUI display."""
        if not self.aws_connected:
            return f"Not connected: {self.error}"
        parts = [f"Account: {self.account_id}"]
        if self.profile_used != "default chain":
            parts.append(f"Profile: {self.profile_used}")
        return " | ".join(parts)


def check_aws_health(settings: Settings | None = None) -> AWSHealthStatus:
    """Check AWS connectivity using boto3 standard credential chain.

    Checks performed:
    1. STS get_caller_identity — verifies credentials exist and are valid
    2. Bedrock client creation — verifies bedrock-runtime is accessible
    3. S3 bucket access — if a bucket is configured, verifies it exists

    Args:
        settings: Settings instance (loads defaults if None)

    Returns:
        AWSHealthStatus with connectivity results
    """
    if settings is None:
        settings = Settings.load()

    status = AWSHealthStatus(
        region=settings.aws.region,
        bedrock_region=settings.aws.effective_bedrock_region,
        profile_used=settings.aws.profile or "default chain",
    )

    try:
        import boto3
        from botocore.exceptions import (
            BotoCoreError,
            ClientError,
            NoCredentialsError,
            NoRegionError,
        )
    except ImportError:
        status.error = "boto3 is unavailable; rerun scripts/setup.sh"
        return status

    # --- 1. Check STS (credentials exist and are valid) ---
    try:
        session = boto3.Session(
            profile_name=settings.aws.effective_profile,
            region_name=resolve_aws_region(settings.aws.region, profile_name=settings.aws.effective_profile),
        )
        sts = session.client("sts")
        identity = sts.get_caller_identity()
        status.aws_connected = True
        status.account_id = identity.get("Account", "")
        status.arn = identity.get("Arn", "")
    except NoCredentialsError:
        status.error = "No AWS credentials found"
        return status
    except NoRegionError:
        status.error = "No AWS region configured"
        return status
    except ClientError as e:
        status.error = f"AWS credential error: {e.response['Error']['Message']}"
        return status
    except BotoCoreError as e:
        status.error = f"AWS connection error: {e!s}"
        return status
    except Exception as e:
        status.error = f"Unexpected error: {e!s}"
        return status

    # --- 2. Check Bedrock access ---
    try:
        bedrock_region = settings.aws.effective_bedrock_region
        # Client creation success means the SDK accepts region + service combo.
        session.client("bedrock-runtime", region_name=resolve_aws_region(bedrock_region, profile_name=settings.aws.effective_profile))
        status.bedrock_available = True
    except Exception as e:
        status.bedrock_available = False
        status.errors["bedrock"] = str(e)

    # --- 3. Check S3 ---
    try:
        s3 = session.client("s3")
        # Basic S3 access test
        s3.list_buckets()
        status.s3_available = True

        # If a specific bucket is configured, verify access
        if settings.aws.s3_bucket:
            try:
                s3.head_bucket(Bucket=settings.aws.s3_bucket)
                status.s3_bucket_accessible = True
            except ClientError:
                status.s3_bucket_accessible = False
                status.errors["s3_bucket"] = f"Cannot access bucket: {settings.aws.s3_bucket}"
        else:
            # No bucket configured = not an error, just unconfigured
            status.s3_bucket_accessible = True
    except Exception as e:
        status.s3_available = False
        status.errors["s3"] = str(e)

    return status
