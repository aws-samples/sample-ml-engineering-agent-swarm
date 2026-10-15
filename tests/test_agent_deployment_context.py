"""Selected deployment context must precede agent resource decisions."""

from __future__ import annotations

import pytest

from amlas.config import externalized
from amlas.control_plane.aws_resource_policy import ResourceControlPolicy, ResourceRequest, authorize
from amlas.flows._util import raw
from amlas.tools.aws_control import describe_aws_readiness


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "AMLAS_RESOURCE_PREFIX",
        "AMLAS_DEPLOYMENT_PREFIX",
        "AMLAS_SSM_PREFIX",
        "SAGEMAKER_EXECUTION_ROLE_ARN",
        "AMLAS_SAGEMAKER_ROLE_ARN",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
    ):
        monkeypatch.delenv(name, raising=False)
    for key in externalized.CONFIG_KEYS.values():
        for name in key.env:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setattr(externalized, "_default_ssm_getter", lambda *_: "")


def test_readiness_exposes_selected_references_without_claiming_existence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "trial-blue")
    monkeypatch.setenv("AMLAS_SESSION_BUCKET", "explicit-session")
    monkeypatch.setenv("SAGEMAKER_EXECUTION_ROLE_ARN", "configured-role")
    requests: list[tuple[str, str]] = []
    published = {
        "/trial-blue/lake-bucket": "trial-blue-lake",
        "/trial-blue/glue-database": "trial_blue_lake",
        "/trial-blue/agentcore-runtime-arn": "configured-runtime",
    }

    def getter(key: str, region: str) -> str:
        requests.append((key, region))
        return published.get(key, "")

    monkeypatch.setattr(externalized, "_default_ssm_getter", getter)
    result = raw(describe_aws_readiness)(region="eu-west-1")
    assert "deployment" in result, "readiness must expose selected deployment configuration"
    context = result["deployment"]
    assert context["prefix"] == "trial-blue"
    assert context["ssm_prefix"] == "/trial-blue"
    assert context["region"] == "eu-west-1"
    assert context["resources"]["lake_bucket"] == "trial-blue-lake"
    assert context["resources"]["session_bucket"] == "explicit-session"
    assert context["resources"]["sagemaker_execution_role_arn"] == "configured-role"
    assert context["resources"]["agentcore_runtime_arn"] == "configured-runtime"
    assert context["verification"] == "configuration_only"
    assert ("/trial-blue/lake-bucket", "eu-west-1") in requests
    assert not any("/amlas/" in key or key.endswith("/session-bucket") for key, _ in requests)


def test_missing_references_are_explicit_and_never_guessed() -> None:
    result = raw(describe_aws_readiness)(region="eu-west-1")
    assert "deployment" in result, "readiness must distinguish missing references from policy"
    context = result["deployment"]
    for name in ("lake_bucket", "session_bucket", "agentcore_runtime_arn", "sagemaker_execution_role_arn"):
        assert context["resources"][name] == ""
        assert name in context["unresolved"]
    assert context["verification"] == "configuration_only"


def test_default_resource_policy_is_narrowed_to_selected_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "trial-blue")
    policy = ResourceControlPolicy.from_env()
    assert policy.resource_prefix == "trial-blue-"
    own = ResourceRequest("s3", "PutObject", "trial-blue-lake/data/", "eu-west-1")
    foreign = ResourceRequest("s3", "PutObject", "amlas-other-lake/data/", "eu-west-1")
    assert authorize(own, policy=policy).allowed
    assert not authorize(foreign, policy=policy).allowed
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "trial-green")
    assert ResourceControlPolicy.from_env().resource_prefix == "trial-green-"


def test_legacy_prefix_and_explicit_owner_override_remain_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert ResourceControlPolicy.from_env().resource_prefix == "amlas-"
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "trial-blue")
    monkeypatch.setenv("AMLAS_RESOURCE_PREFIX", "chosen-scope-")
    assert ResourceControlPolicy.from_env().resource_prefix == "chosen-scope-"
