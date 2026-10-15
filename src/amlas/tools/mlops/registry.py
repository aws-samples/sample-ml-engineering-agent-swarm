"""SageMaker Model Registry tools (spec sheet 2.8, master plan §8 D2).

``register_model_sm`` creates the Model Package Group (if absent) and registers a
versioned Model Package with metric metadata + an approval status;
``approve_model_sm`` transitions a package PendingManualApproval → Approved (the
audit-logged governance gate); ``list_model_packages_sm`` enumerates the versions
in a group (the registry table the TUI renders).

SageMaker supports ``create_model_package_group`` / ``create_model_package`` /
``update_model_package`` / ``list_model_packages`` — so these tools are exercised
against the REAL standing ``amlas-models`` group (ephemeral package versions,
deleted on teardown). Injectable ``sm_client`` is a real boto3 client; no faking.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from ...errors import tool_error_boundary
from ._common import make_sm_client, sanitize_name

# FIX-15 (Phase 6) + SEC-A01: a real ``CreateModelPackage`` validates the inference
# image EXISTS + is pullable IN-REGION, so a cross-region DLC account (e.g. a us-east-1
# account while ``region`` defaults to us-west-2) is rejected with "Access denied for
# registry ID: ... check if your ECR image exists". The public DLC account differs per
# region and is therefore DERIVED AT RUNTIME rather than baked into a hardcoded
# per-region account map on this shipped surface (SEC-A01: no literal account ids). We
# prefer the SageMaker SDK's bundled, offline per-region resolver
# (``image_uris.retrieve`` — authoritative for EVERY region) and fall back to the
# canonical public map in ``pipelines._sklearn_image`` when the slim runtime does not
# ship the high-level ``sagemaker`` SDK. Callers may still pass an explicit ``image=``.


def _default_sklearn_image(region: str) -> str:
    """Return the region-correct public SKLearn inference image URI (FIX-15 / SEC-A01)."""
    try:
        # sagemaker ships in the optional ``agents`` extra; absent from the offline dev venv
        # (pyright cannot resolve it there). Guarded by the enclosing try/except fallback.
        from sagemaker import image_uris  # pyright: ignore[reportMissingImports]

        return str(
            image_uris.retrieve(
                framework="sklearn",
                region=region,
                version="1.2-1",
                image_scope="inference",
            )
        )
    except Exception:
        # Slim runtime (no high-level ``sagemaker`` SDK) — reuse the canonical public
        # DLC-account resolver so the default stays a real in-region image URI.
        from .pipelines import _sklearn_image

        return _sklearn_image(region)


def _ensure_group(sm: Any, group: str) -> None:
    """Create the Model Package Group if it does not already exist (idempotent)."""
    try:
        sm.describe_model_package_group(ModelPackageGroupName=group)
        return
    except Exception:  # nosec B110
        pass
    try:
        sm.create_model_package_group(
            ModelPackageGroupName=group,
            ModelPackageGroupDescription="Amlas MLOps model package group",
        )
    except Exception:  # nosec B110
        pass


@tool
@tool_error_boundary(source="tools.mlops.registry")
def register_model_sm(
    artifact: str,
    group: str,
    metrics: dict[str, Any] | None = None,
    approval: str = "PendingManualApproval",
    *,
    image: str | None = None,
    region: str | None = None,
    sm_client: Any = None,
) -> dict[str, Any]:
    """Register a model package into a Model Package Group (master plan §8 D2).

    Creates the group if absent, then registers a versioned package pointing at
    the ``artifact`` (``s3://.../model.tar.gz``) with the inference container,
    attaching ``metrics`` as customer metadata and setting the approval gate.

    Args:
        artifact: S3 URI of the model artifact (``ModelDataUrl``).
        group: Model Package Group name (sanitized to SageMaker naming).
        metrics: Optional ``{metric: value}`` map attached as CustomerMetadata.
        approval: Initial approval status (default ``PendingManualApproval``).
        image: Optional inference image URI (defaults to a CPU sklearn image).
        region: AWS region.
        sm_client: Injected SageMaker client (real-AWS-backed in tests).

    Returns:
        ``{"model_package_arn","version","group","approval_status"}`` or
        ``{"success": False, "error": ...}``.
    """
    try:
        grp = sanitize_name(group)
        sm = make_sm_client(region, sm_client)
        _ensure_group(sm, grp)
        container: dict[str, Any] = {
            "Image": image or _default_sklearn_image(region),
        }
        if artifact:
            container["ModelDataUrl"] = artifact
        kwargs: dict[str, Any] = {
            "ModelPackageGroupName": grp,
            "ModelApprovalStatus": approval,
            "InferenceSpecification": {
                "Containers": [container],
                "SupportedContentTypes": ["text/csv"],
                "SupportedResponseMIMETypes": ["text/csv"],
            },
        }
        if metrics:
            kwargs["CustomerMetadataProperties"] = {
                str(k): str(v) for k, v in metrics.items()
            }
        resp = sm.create_model_package(**kwargs)
        arn = resp.get("ModelPackageArn", "")
        version = _version_from_arn(arn)
        return {
            "model_package_arn": arn,
            "version": version,
            "group": grp,
            "approval_status": approval,
            "metrics": dict(metrics or {}),
        }
    except Exception as e:
        return {"success": False, "error": f"register_model_sm failed: {e}"}


@tool
@tool_error_boundary(source="tools.mlops.registry")
def approve_model_sm(
    arn: str,
    *,
    status: str = "Approved",
    region: str | None = None,
    sm_client: Any = None,
) -> dict[str, Any]:
    """Transition a registered model package's approval status (master plan §8 D2).

    The audit-logged gate: PendingManualApproval → Approved (or Rejected).

    Args:
        arn: The ModelPackageArn to transition.
        status: Target status — ``Approved`` (default) or ``Rejected``.
        region: AWS region.
        sm_client: Injected SageMaker client (real-AWS-backed in tests).

    Returns:
        ``{"model_package_arn","status"}`` or ``{"success": False, "error": ...}``.
    """
    try:
        if status not in ("Approved", "Rejected", "PendingManualApproval"):
            return {"success": False, "error": f"invalid approval status: {status!r}"}
        sm = make_sm_client(region, sm_client)
        sm.update_model_package(ModelPackageArn=arn, ModelApprovalStatus=status)
        return {"model_package_arn": arn, "status": status}
    except Exception as e:
        return {"success": False, "error": f"approve_model_sm failed: {e}"}


@tool
@tool_error_boundary(source="tools.mlops.registry")
def list_model_packages_sm(
    group: str,
    *,
    region: str | None = None,
    sm_client: Any = None,
) -> dict[str, Any]:
    """List the model package versions in a group (the registry table; master plan §8).

    Args:
        group: Model Package Group name.
        region: AWS region.
        sm_client: Injected SageMaker client (real-AWS-backed in tests).

    Returns:
        ``{"group","versions":[{"model_package_arn","version","approval_status"}],"count"}``
        or ``{"success": False, "error": ...}``.
    """
    try:
        grp = sanitize_name(group)
        sm = make_sm_client(region, sm_client)
        resp = sm.list_model_packages(ModelPackageGroupName=grp)
        versions = [
            {
                "model_package_arn": p.get("ModelPackageArn", ""),
                "version": p.get("ModelPackageVersion", _version_from_arn(p.get("ModelPackageArn", ""))),
                "approval_status": p.get("ModelApprovalStatus", ""),
            }
            for p in resp.get("ModelPackageSummaryList", [])
        ]
        return {"group": grp, "versions": versions, "count": len(versions)}
    except Exception as e:
        return {"success": False, "error": f"list_model_packages_sm failed: {e}"}


def _version_from_arn(arn: str) -> int:
    """Extract the trailing integer version from a model-package ARN (``.../group/3``)."""
    try:
        return int(str(arn).rstrip("/").rsplit("/", 1)[-1])
    except (ValueError, IndexError):
        return 1
