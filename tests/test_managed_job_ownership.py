"""Atomic native job ownership and accepted-handle regressions; no AWS network."""

from __future__ import annotations

from typing import Any

import pytest
from botocore.exceptions import ClientError
from test_region_client_metadata import RecordingSageMaker

UUID = "00000000-0000-0000-0000-aaaaaaaaaa33"
SESSION = "owned-runtime-session-0000000000000001"


@pytest.fixture(autouse=True)
def owned_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    for name in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AMLAS_AWS_PROFILE", "AGENTICML_AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-credentials"))
    monkeypatch.setenv("AMLAS_DEPLOYMENT_ID", UUID)
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "trial-blue")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


class NativeSageMaker(RecordingSageMaker):
    def __init__(self, *, describe_failure: bool = False) -> None:
        super().__init__()
        self.describe_failure = describe_failure
        self.accepted: dict[str, Any] = {}

    def create_training_job(self, **kwargs: Any) -> dict[str, str]:
        self._record("CreateTrainingJob", kwargs)
        self.accepted = kwargs
        return {
            "TrainingJobArn": "arn:aws:sagemaker:eu-west-1:000000000000:training-job/"
            + kwargs["TrainingJobName"]
        }

    def create_processing_job(self, **kwargs: Any) -> dict[str, str]:
        self._record("CreateProcessingJob", kwargs)
        self.accepted = kwargs
        return {
            "ProcessingJobArn": "arn:aws:sagemaker:eu-west-1:000000000000:processing-job/"
            + kwargs["ProcessingJobName"]
        }

    def describe_processing_job(self, **kwargs: Any) -> dict[str, str]:
        if self.describe_failure:
            raise ClientError(
                {"Error": {"Code": "InternalFailure", "Message": "fixture failure"}}, "DescribeProcessingJob"
            )
        return super().describe_processing_job(**kwargs)


class DynamoDB:
    def __init__(self, *, update_failure: bool = False) -> None:
        self.update_failure = update_failure

    def put_item(self, **kwargs: Any) -> dict:
        return {}

    def update_item(self, **kwargs: Any) -> dict:
        if self.update_failure:
            raise ClientError(
                {"Error": {"Code": "InternalFailure", "Message": "fixture failure"}}, "UpdateItem"
            )
        return {}


def training(sm: NativeSageMaker, ddb: DynamoDB) -> Any:
    from amlas.tools.sagemaker import submit_training_job

    return submit_training_job(
        training_type="xgboost",
        config={},
        runtime_session_id=SESSION,
        execution_role_arn="arn:aws:iam::000000000000:role/owned-exec",
        session_bucket="owned-session",
        sagemaker_client=sm,
        dynamodb_client=ddb,
    )


def tag_map(request: dict[str, Any]) -> dict[str, str]:
    return {item["Key"]: item["Value"] for item in request["Tags"]}


def test_training_tags_are_atomic_with_native_creation() -> None:
    sm = NativeSageMaker()
    result = training(sm, DynamoDB())
    assert tag_map(sm.accepted) == {
        "amlas:deployment-id": UUID,
        "amlas:deployment-prefix": "trial-blue",
        "amlas:runtime-session-id": SESSION,
    }
    assert result.training_job_arn.endswith("/" + result.job_name)


def test_training_accepted_handle_survives_post_create_tracking_failure() -> None:
    from amlas.tools.sagemaker import SageMakerSubmitError

    sm = NativeSageMaker()
    with pytest.raises(SageMakerSubmitError) as caught:
        training(sm, DynamoDB(update_failure=True))
    accepted = caught.value.accepted_job
    assert accepted is not None
    assert accepted.job_name == sm.accepted["TrainingJobName"]
    assert accepted.runtime_session_id == SESSION
    assert accepted.training_job_arn.endswith("/" + accepted.job_name)
    assert tag_map(sm.accepted)["amlas:runtime-session-id"] == SESSION


def test_processing_tags_and_accepted_handle_survive_describe_failure() -> None:
    from amlas.backends.adapters.etl import ProcessingJobBackend

    sm = NativeSageMaker(describe_failure=True)
    result = ProcessingJobBackend(
        role_arn="arn:aws:iam::000000000000:role/owned-exec", sagemaker_client=sm
    ).run_transform(
        input_uri="s3://owned-session/input/",
        output="s3://owned-session/output/",
        transform_spec={"op": "normalize"},
        resources={"runtime_session_id": SESSION},
    )
    assert result["accepted"] is True
    assert result["arn"].endswith("/" + result["job_name"])
    assert result["runtime_session_id"] == SESSION
    assert result["observation_error"] == "ClientError"
    assert tag_map(sm.accepted) == {
        "amlas:deployment-id": UUID,
        "amlas:deployment-prefix": "trial-blue",
        "amlas:runtime-session-id": SESSION,
    }


def test_invalid_deployment_uuid_fails_before_any_sdk_write(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AMLAS_DEPLOYMENT_ID", "invalid")
    sm = NativeSageMaker()
    with pytest.raises(ValueError, match="deployment"):
        training(sm, DynamoDB())
    assert sm.requests == []
