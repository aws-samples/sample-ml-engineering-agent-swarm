"""Region metadata regressions; no credentials, network, or cloud writes.

Profile grammar is checked against botocore.configloader.build_profile_map:
https://github.com/boto/botocore/blob/develop/botocore/configloader.py
SDK request fakes validate the installed SageMaker service model before recording.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import botocore.configloader
import botocore.session
import pytest
from botocore.handlers import generate_idempotent_uuid
from botocore.validate import validate_parameters

from amlas.config.aws_region import MissingRegionError, resolve_aws_region
from amlas.config.sagemaker_images import (
    clarify_image,
    model_monitor_image,
    sklearn_processing_image,
    sklearn_training_image,
    xgboost_image,
)


@pytest.fixture(autouse=True)
def no_ambient_region(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in (
        "AWS_REGION",
        "AWS_DEFAULT_REGION",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AMLAS_AWS_REGION",
        "AGENTICML_AWS_REGION",
        "AMLAS_AWS_PROFILE",
        "AGENTICML_AWS_PROFILE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "missing-aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "must-not-read-credentials"))
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


@pytest.mark.parametrize(
    ("section", "profile"),
    [
        ('profile "team dev"', "team dev"),
        ("profile 'team dev'", "team dev"),
        (r"profile team\ dev", "team dev"),
        ("profile team", "team"),
        ("default", "default"),
        ('profile "default"', "default"),
    ],
)
def test_profile_section_grammar_matches_botocore(
    section: str, profile: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "aws-config"
    config.write_text(f"[{section}]\nregion = eu-west-1\ncredential_process = must-not-execute\n")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    assert botocore.configloader.load_config(str(config))["profiles"][profile]["region"] == "eu-west-1"
    try:
        actual = resolve_aws_region(profile_name=profile)
    except MissingRegionError as error:
        pytest.fail(f"AWS-supported profile did not resolve: {error}")
    assert actual == "eu-west-1"


@pytest.mark.parametrize("section", ['profile "unfinished', "profile team dev", "profile"])
def test_invalid_profile_sections_do_not_become_profiles(
    section: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "aws-config"
    config.write_text(f"[{section}]\nregion = eu-west-1\n")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    assert "team dev" not in botocore.configloader.load_config(str(config))["profiles"]
    assert resolve_aws_region(profile_name="team dev", required=False) is None
    with pytest.raises(MissingRegionError, match="No AWS region selected"):
        resolve_aws_region(profile_name="team dev")


class RecordingSageMaker:
    """Strict request-shape recorder with concrete endpoint metadata."""

    def __init__(self, region: str | None = "eu-west-1") -> None:
        self.meta = SimpleNamespace(region_name=region)
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.model = botocore.session.get_session().get_service_model("sagemaker")

    def _record(self, operation: str, kwargs: dict[str, Any]) -> None:
        model = self.model.operation_model(operation)
        # The real boto3 call runs this handler before request validation.
        generate_idempotent_uuid(kwargs, model)
        validate_parameters(kwargs, model.input_shape)
        self.requests.append((operation, kwargs))

    def create_processing_job(self, **kwargs: Any) -> dict[str, str]:
        self._record("CreateProcessingJob", kwargs)
        return {"ProcessingJobArn": "arn:aws:sagemaker:eu-west-1:000000000000:processing-job/test"}

    def describe_processing_job(self, **kwargs: Any) -> dict[str, str]:
        self._record("DescribeProcessingJob", kwargs)
        return {"ProcessingJobStatus": "InProgress"}

    def create_model(self, **kwargs: Any) -> dict[str, str]:
        self._record("CreateModel", kwargs)
        return {"ModelArn": "arn:aws:sagemaker:eu-west-1:000000000000:model/test"}

    def create_hyper_parameter_tuning_job(self, **kwargs: Any) -> dict[str, str]:
        self._record("CreateHyperParameterTuningJob", kwargs)
        return {
            "HyperParameterTuningJobArn": "arn:aws:sagemaker:eu-west-1:000000000000:hyper-parameter-tuning-job/test"
        }

    def describe_hyper_parameter_tuning_job(self, **kwargs: Any) -> dict[str, str]:
        self._record("DescribeHyperParameterTuningJob", kwargs)
        return {"HyperParameterTuningJobStatus": "InProgress"}

    def create_training_job(self, **kwargs: Any) -> dict[str, str]:
        self._record("CreateTrainingJob", kwargs)
        return {"TrainingJobArn": "arn:aws:sagemaker:eu-west-1:000000000000:training-job/test"}

    def create_pipeline(self, **kwargs: Any) -> dict[str, str]:
        self._record("CreatePipeline", kwargs)
        return {"PipelineArn": "arn:aws:sagemaker:eu-west-1:000000000000:pipeline/test"}

    def create_monitoring_schedule(self, **kwargs: Any) -> dict[str, str]:
        self._record("CreateMonitoringSchedule", kwargs)
        return {"MonitoringScheduleArn": "arn:aws:sagemaker:eu-west-1:000000000000:monitoring-schedule/test"}


def call_image_boundary(
    boundary: str, client: RecordingSageMaker, *, region: str | None = None, image: str | None = None
) -> str:
    if boundary == "etl":
        from amlas.backends.adapters.etl import ProcessingJobBackend

        result = ProcessingJobBackend(region=region, sagemaker_client=client).run_transform(
            input_uri="s3://input-bucket/data.csv",
            output="s3://output-bucket/data/",
            transform_spec={"op": "normalize"},
            resources={"image": image} if image else {},
        )
        assert result["success"] is True
        return client.requests[0][1]["AppSpecification"]["ImageUri"]
    if boundary == "deployment":
        from amlas.backends.adapters.deployment import SageMakerDeploymentBackend
        from amlas.backends.types import ModelRef

        SageMakerDeploymentBackend(region=region, sagemaker_client=client).package(
            model_ref=ModelRef(uri="s3://model-bucket/model.tar.gz", algo="xgboost"),
            image=image,
        )
        return client.requests[0][1]["PrimaryContainer"]["Image"]
    if boundary == "tuning":
        from amlas.backends.adapters.tuning import SageMakerAMTBackend

        SageMakerAMTBackend(region=region, sagemaker_client=client).search(
            base_training={
                "train_uri": "s3://input-bucket/train/",
                "output_uri": "s3://output-bucket/models/",
                **({"image": image} if image else {}),
            },
            ranges={"eta": {"type": "float", "low": 0.1, "high": 0.5}},
            objective="validation:auc",
        )
        return client.requests[0][1]["TrainingJobDefinition"]["AlgorithmSpecification"]["TrainingImage"]
    if boundary == "clarify":
        from amlas.tools.mlops.clarify import run_clarify

        result = run_clarify(
            "test-model",
            "s3://input-bucket/data.csv",
            {"label": "target", "output_path": "s3://output-bucket/clarify/"},
            region=region,
            image=image,
            sm_client=client,
        )
        if result.get("success") is False:
            raise ValueError(result["error"])
        return client.requests[0][1]["AppSpecification"]["ImageUri"]
    if boundary == "clarify-adapter":
        from amlas.backends.adapters.explainability import ClarifyBackend

        ClarifyBackend(region=region, sm_client=client).bias_and_shap(
            dataset_uri="s3://input-bucket/data.csv",
            target="label",
        )
        return client.requests[0][1]["AppSpecification"]["ImageUri"]
    if boundary == "training":
        from amlas.tools.sagemaker import _create_training_job

        _create_training_job(
            sm_client=client,
            job_name="region-test",
            training_type="sklearn",
            config={"skip_source_bundle": True, **({"image_uri": image} if image else {})},
            execution_role_arn="arn:aws:iam::000000000000:role/test",
            session_bucket="model-bucket",
            runtime_session_id="region-test",
            region=region,
        )
        return client.requests[0][1]["AlgorithmSpecification"]["TrainingImage"]
    if boundary == "pipeline":
        from amlas.tools.mlops.pipelines import build_sagemaker_pipeline

        result = build_sagemaker_pipeline(
            {
                "name": "region-test",
                "training_type": "sklearn",
                "session_bucket": "model-bucket",
                "role_arn": "arn:aws:iam::000000000000:role/test",
            },
            region=region,
            sm_client=client,
        )
        if result.get("success") is False:
            raise ValueError(result["error"])
        steps = result["definition"]["Steps"]
        training = next(step for step in steps if step["Name"] == "Training")
        assert training["Arguments"]["AlgorithmSpecification"]["TrainingImage"] == sklearn_training_image(
            "eu-west-1"
        )
        return steps[0]["Arguments"]["AppSpecification"]["ImageUri"]
    if boundary == "monitor":
        from amlas.tools.mlops.monitor import setup_model_monitor

        result = setup_model_monitor(
            "test-endpoint",
            "s3://input-bucket/baseline/",
            region=region,
            sm_client=client,
            output_path="s3://output-bucket/monitor/",
        )
        if result.get("success") is False:
            raise ValueError(result["error"])
        return client.requests[0][1]["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringAppSpecification"
        ]["ImageUri"]
    raise AssertionError(f"unknown image boundary {boundary}")


IMAGE_BOUNDARIES = [
    "etl",
    "deployment",
    "tuning",
    "clarify",
    "monitor",
    "clarify-adapter",
    "training",
    "pipeline",
]


@pytest.mark.parametrize("boundary", IMAGE_BOUNDARIES)
@pytest.mark.parametrize("ambient", [None, "us-west-2"])
def test_injected_endpoint_region_controls_default_image(
    boundary: str, ambient: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if ambient:
        monkeypatch.setenv("AWS_REGION", ambient)
    client = RecordingSageMaker()
    monkeypatch.setattr(
        "boto3.client", lambda *args, **kwargs: pytest.fail("injected path constructed an SDK client")
    )
    expected = {
        "etl": sklearn_processing_image("eu-west-1"),
        "deployment": xgboost_image("eu-west-1"),
        "tuning": xgboost_image("eu-west-1"),
        "clarify": clarify_image("eu-west-1"),
        "monitor": model_monitor_image("eu-west-1"),
        "clarify-adapter": clarify_image("eu-west-1"),
        "training": sklearn_training_image("eu-west-1"),
        "pipeline": sklearn_processing_image("eu-west-1"),
    }
    try:
        actual = call_image_boundary(boundary, client)
    except (ValueError, RuntimeError) as error:
        pytest.fail(f"Injected endpoint metadata did not resolve: {error}")
    assert actual == expected[boundary]


@pytest.mark.parametrize("boundary", IMAGE_BOUNDARIES)
def test_explicit_endpoint_region_conflict_fails_before_writes(boundary: str) -> None:
    client = RecordingSageMaker()
    with pytest.raises((ValueError, RuntimeError), match=r"region.*differs|region.*conflict"):
        call_image_boundary(boundary, client, region="us-west-2")
    assert client.requests == []


@pytest.mark.parametrize("boundary", ["etl", "deployment", "tuning", "clarify", "training"])
@pytest.mark.parametrize("image_region", ["eu-west-1", "us-west-2"])
def test_custom_regional_image_is_checked_against_endpoint(
    boundary: str, image_region: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    client = RecordingSageMaker()
    image = f"000000000000.dkr.ecr.{image_region}.amazonaws.com/custom:test"
    if image_region == "eu-west-1":
        try:
            actual = call_image_boundary(boundary, client, image=image)
        except (ValueError, RuntimeError) as error:
            pytest.fail(f"Same-region override was rejected: {error}")
        assert actual == image
    else:
        with pytest.raises(ValueError, match=r"Image region.*differs"):
            call_image_boundary(boundary, client, image=image)
        assert client.requests == []


@pytest.mark.parametrize("boundary", IMAGE_BOUNDARIES)
def test_missing_endpoint_metadata_requires_explicit_region_for_defaults(boundary: str) -> None:
    client = RecordingSageMaker(None)
    with pytest.raises((ValueError, RuntimeError), match="No AWS region selected"):
        call_image_boundary(boundary, client)
    assert client.requests == []


@pytest.mark.parametrize("boundary", ["etl", "deployment", "tuning", "clarify", "training"])
def test_explicit_custom_image_allows_injected_client_without_region_metadata(boundary: str) -> None:
    client = RecordingSageMaker(None)
    image = "000000000000.dkr.ecr.eu-west-1.amazonaws.com/custom:test"
    assert call_image_boundary(boundary, client, image=image) == image


def test_actual_source_bundle_contains_region_dependency_and_imports_standalone(tmp_path: Path) -> None:
    from amlas.tools.sagemaker_source_bundle import build_source_tarball

    with tarfile.open(fileobj=io.BytesIO(build_source_tarball()), mode="r:gz") as archive:
        assert "amlas/config/aws_region.py" in archive.getnames()
        archive.extractall(tmp_path, filter="data")
    code = """
import pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
sys.path[:] = [str(root)] + [p for p in sys.path if 'site-packages' in p or p.startswith(sys.base_prefix)]
import train
import amlas.config.aws_region as region
import amlas.sagemaker_entrypoints.tabular_train as entry
assert pathlib.Path(region.__file__).resolve().is_relative_to(root)
assert pathlib.Path(entry.__file__).resolve().is_relative_to(root)
assert callable(train.main)
print('BUNDLE_IMPORT_PASS')
"""
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(tmp_path)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "BUNDLE_IMPORT_PASS"


class RecordingDynamoDB:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.meta = SimpleNamespace(region_name="eu-west-1")
        self.model = botocore.session.get_session().get_service_model("dynamodb")

    def put_item(self, **kwargs: Any) -> dict[str, Any]:
        validate_parameters(kwargs, self.model.operation_model("PutItem").input_shape)
        self.requests.append(("PutItem", kwargs))
        return {}

    def update_item(self, **kwargs: Any) -> dict[str, Any]:
        validate_parameters(kwargs, self.model.operation_model("UpdateItem").input_shape)
        self.requests.append(("UpdateItem", kwargs))
        return {}


@pytest.mark.parametrize("boundary", ["pipeline", "monitor", "training"])
@pytest.mark.parametrize("ambient", [None, "us-west-2"])
def test_adapter_keeps_ambient_region_distinct_from_explicit_selection(
    boundary: str, ambient: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if ambient:
        monkeypatch.setenv("AWS_REGION", ambient)
    client = RecordingSageMaker()
    monkeypatch.setattr("boto3.client", lambda *args, **kwargs: pytest.fail("injected path built a client"))
    if boundary == "pipeline":
        import json

        from amlas.backends.adapters.pipeline import SageMakerPipelineBackend

        try:
            SageMakerPipelineBackend(sm_client=client).build(
                spec={
                    "name": "region-test",
                    "training_type": "sklearn",
                    "session_bucket": "model-bucket",
                    "role_arn": "arn:aws:iam::000000000000:role/test",
                }
            )
        except RuntimeError as error:
            pytest.fail(f"Injected pipeline endpoint was overridden: {error}")
        steps = json.loads(client.requests[0][1]["PipelineDefinition"])["Steps"]
        image = steps[0]["Arguments"]["AppSpecification"]["ImageUri"]
        expected = sklearn_processing_image("eu-west-1")
    elif boundary == "monitor":
        from amlas.backends.adapters.monitoring import SageMakerModelMonitorBackend

        schedule = SageMakerModelMonitorBackend(sm_client=client).setup(
            endpoint="test-endpoint",
            baseline_uri="s3://input-bucket/baseline/",
        )
        assert schedule.arn, "injected monitor endpoint was overridden and degraded"
        image = client.requests[0][1]["MonitoringScheduleConfig"]["MonitoringJobDefinition"][
            "MonitoringAppSpecification"
        ]["ImageUri"]
        expected = model_monitor_image("eu-west-1")
    else:
        from amlas.backends.adapters.training import SageMakerTrainingBackend
        from amlas.config.sagemaker_images import pytorch_training_image

        ddb = RecordingDynamoDB()
        try:
            SageMakerTrainingBackend(
                sagemaker_client=client,
                dynamodb_client=ddb,
                session_bucket="model-bucket",
                execution_role_arn="arn:aws:iam::000000000000:role/test",
                runtime_session_id="region-test",
            ).submit(algo="train", train_uri="s3://input-bucket/train/")
        except ValueError as error:
            pytest.fail(f"Injected training endpoint was overridden: {error}")
        image = client.requests[0][1]["AlgorithmSpecification"]["TrainingImage"]
        expected = pytorch_training_image("eu-west-1", gpu=False)
    assert image == expected


def test_training_region_conflict_is_rejected_before_ddb_tracking_write() -> None:
    from amlas.tools.sagemaker import submit_training_job

    client = RecordingSageMaker()
    ddb = RecordingDynamoDB()
    with pytest.raises(ValueError, match=r"region.*differs"):
        submit_training_job(
            training_type="sklearn",
            config={"skip_source_bundle": True},
            region="us-west-2",
            runtime_session_id="region-test",
            execution_role_arn="arn:aws:iam::000000000000:role/test",
            session_bucket="model-bucket",
            sagemaker_client=client,
            dynamodb_client=ddb,
        )
    assert ddb.requests == []
    assert client.requests == []
