"""Closed AWS-boundary regressions for isolated deployment discovery and jobs."""

from __future__ import annotations

import fnmatch
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from test_region_client_metadata import RecordingSageMaker

from amlas.config import externalized


@pytest.fixture(autouse=True)
def clean_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AMLAS_AWS_PROFILE",
        "AGENTICML_AWS_PROFILE",
        "AMLAS_DEPLOYMENT_PREFIX",
        "AMLAS_SSM_PREFIX",
        "AMLAS_JOBS_TABLE",
        "JOBS_TABLE",
        "TAVILY_SECRET_ID",
        "AMLAS_TAVILY_SECRET_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    for key in externalized.CONFIG_KEYS.values():
        for name in key.env:
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")


def deployment() -> Any:
    try:
        return importlib.import_module("amlas.config.deployment")
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing namespace resolver module: {exc.name}")


def test_default_prefix_and_stack_names_remain_legacy() -> None:
    module = deployment()
    assert module.resolve_deployment_prefix() == "amlas"
    assert module.resolve_ssm_prefix() == "/amlas"
    assert module.deployment_stack_name("LakeStack", "amlas") == "LakeStack"


def test_prefix_and_ssm_selectors_resolve_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    module = deployment()
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "trial-blue")
    assert module.resolve_deployment_prefix() == "trial-blue"
    assert module.resolve_ssm_prefix() == "/trial-blue"
    assert module.deployment_stack_name("LakeStack", "trial-blue") == "trial-blue-LakeStack"
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "trial-green")
    assert module.resolve_deployment_prefix() == "trial-green"
    monkeypatch.setenv("AMLAS_SSM_PREFIX", "/shared/trial-green/")
    assert module.resolve_ssm_prefix() == "/shared/trial-green"


@pytest.mark.parametrize(
    "value",
    [
        "Upper",
        "has_space here",
        "bad_underscore",
        "-leading",
        "trailing-",
        "double--dash",
        "a" * 25,
        "a/branch",
        "1number",
    ],
)
def test_invalid_prefix_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", value)
    with pytest.raises(ValueError, match="deployment prefix"):
        deployment().resolve_deployment_prefix()


@pytest.mark.parametrize("value", ["relative", "/", "/bad space", "/bad//path"])
def test_invalid_ssm_prefix_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("AMLAS_SSM_PREFIX", value)
    with pytest.raises(ValueError, match="SSM prefix"):
        deployment().resolve_ssm_prefix()


@pytest.mark.parametrize(
    "name",
    [
        "lake_bucket",
        "session_bucket",
        "agentcore_runtime_arn",
        "guardrail_id",
        "model_package_group",
        "tavily_secret_id",
    ],
)
def test_externalized_reads_selected_ssm_namespace(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    requests: list[str] = []
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "trial-blue")

    def getter(key: str, region: str) -> str:
        requests.append(key)
        return "published-value"

    assert externalized.resolve(name, ssm_getter=getter) == "published-value"
    assert requests == [externalized.CONFIG_KEYS[name].ssm.replace("/amlas/", "/trial-blue/")]
    monkeypatch.setenv("AMLAS_SSM_PREFIX", "/shared/green")
    requests.clear()
    externalized.resolve(name, ssm_getter=getter)
    assert requests == [externalized.CONFIG_KEYS[name].ssm.replace("/amlas/", "/shared/green/")]


@pytest.mark.parametrize(
    ("name", "suffix"),
    [
        ("model_package_group", "-models"),
        ("feature_store_db", "_features"),
        ("tavily_secret_id", "/tavily-api-key"),
    ],
)
def test_resource_defaults_follow_selected_prefix(
    monkeypatch: pytest.MonkeyPatch, name: str, suffix: str
) -> None:
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "trial-blue")
    prefix = "trial_blue" if name == "feature_store_db" else "trial-blue"
    assert externalized.resolve(name, ssm_getter=lambda *_: "") == prefix + suffix
    monkeypatch.setenv(externalized.CONFIG_KEYS[name].env[0], "explicit-resource")
    assert (
        externalized.resolve(name, ssm_getter=lambda *_: pytest.fail("explicit env must win"))
        == "explicit-resource"
    )


def test_harness_discovery_uses_selected_namespace_and_env_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    from amlas.harness.leads import harness_arn_for, harness_ssm_key

    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "trial-blue")
    assert harness_ssm_key("research_lead") == "/trial-blue/harness/research-lead-id"
    requests: list[str] = []

    def getter(key: str, region: str) -> str:
        requests.append(key)
        return "example"

    result = harness_arn_for(
        "research_lead",
        env={"AMLAS_DEPLOYMENT_PREFIX": "mapped-green"},
        ssm_getter=getter,
        region="eu-west-1",
        account_id="000000000000",
    )
    assert result.endswith("/example")
    assert requests == ["/mapped-green/harness/research-lead-id"]


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        (None, "trial-blue/tavily-api-key"),
        ("AMLAS_TAVILY_SECRET_ID", "chosen-secret"),
        ("TAVILY_SECRET_ID", "chosen-secret"),
    ],
)
def test_secret_discovery_respects_selected_prefix_and_overrides(
    monkeypatch: pytest.MonkeyPatch, override: str | None, expected: str
) -> None:
    from amlas.cloud.secrets import resolve_tavily_key

    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "trial-blue")
    if override:
        monkeypatch.setenv(override, "chosen-secret")
    ids: list[str] = []

    class SecretClient:
        def get_secret_value(self, **kwargs: str) -> dict[str, str]:
            assert set(kwargs) == {"SecretId"}
            ids.append(kwargs["SecretId"])
            fixture_value = "".join(("place", "holder"))  # Noncredential placeholder recognition fixture.
            return {"SecretString": fixture_value}

    assert resolve_tavily_key(client=SecretClient()) is None
    assert ids == [expected]


class RecordingDynamoDB:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def put_item(self, **kwargs: Any) -> dict[str, Any]:
        self.requests.append(kwargs)
        return {}

    def update_item(self, **kwargs: Any) -> dict[str, Any]:
        self.requests.append(kwargs)
        return {}

    def get_item(self, **kwargs: Any) -> dict[str, Any]:
        self.requests.append(kwargs)
        return {"Item": {"status": {"S": "RUNNING"}}}


@pytest.mark.parametrize(
    ("prefix", "table_override"), [("amlas", None), ("trial-blue", None), ("trial-blue", "explicit-jobs")]
)
def test_training_submit_and_status_resolve_jobs_table_at_call_time(
    monkeypatch: pytest.MonkeyPatch, prefix: str, table_override: str | None
) -> None:
    from amlas.tools.sagemaker import get_job_status, submit_training_job

    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", prefix)
    if table_override:
        monkeypatch.setenv("AMLAS_JOBS_TABLE", table_override)
    ddb, sm = RecordingDynamoDB(), RecordingSageMaker()
    result = submit_training_job(
        training_type="xgboost",
        config={},
        runtime_session_id="example-session",
        execution_role_arn="arn:aws:iam::000000000000:role/example",
        session_bucket="example-bucket",
        dynamodb_client=ddb,
        sagemaker_client=sm,
    )
    assert fnmatch.fnmatchcase(result.job_name, prefix + "-*")
    assert re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", result.job_name)
    get_job_status(result.job_name, dynamodb_client=ddb)
    assert {request["TableName"] for request in ddb.requests} == {table_override or prefix + "-training-jobs"}
    get_job_status(result.job_name, jobs_table="argument-jobs", dynamodb_client=ddb)
    assert ddb.requests[-1]["TableName"] == "argument-jobs"


def test_training_backend_resource_override_is_preserved_for_status(monkeypatch: pytest.MonkeyPatch) -> None:
    from amlas.backends.adapters.training import SageMakerTrainingBackend

    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "trial-blue")
    ddb, sm = RecordingDynamoDB(), RecordingSageMaker()
    backend = SageMakerTrainingBackend(
        execution_role_arn="arn:aws:iam::000000000000:role/example",
        session_bucket="example-bucket",
        dynamodb_client=ddb,
        sagemaker_client=sm,
    )
    handle = backend.submit(
        algo="xgboost", train_uri="s3://example-bucket/train/", resources={"jobs_table": "resource-jobs"}
    )
    backend.status(handle)
    assert {request["TableName"] for request in ddb.requests} == {"resource-jobs"}


@pytest.mark.parametrize("prefix", ["amlas", "trial-blue", "a" * 24])
def test_processing_job_name_matches_runtime_iam_scope_and_api_limit(
    monkeypatch: pytest.MonkeyPatch, prefix: str
) -> None:
    from amlas.backends.adapters.etl import ProcessingJobBackend

    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", prefix)
    sm = RecordingSageMaker()
    result = ProcessingJobBackend(
        role_arn="arn:aws:iam::000000000000:role/example", sagemaker_client=sm
    ).run_transform(
        input_uri="s3://example-bucket/data.csv",
        output="s3://example-bucket/result/",
        transform_spec={"op": "normalize " + "feature" * 15},
    )
    assert fnmatch.fnmatchcase(result["job_name"], prefix + "-*")
    assert len(result["job_name"]) <= 63
    assert re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", result["job_name"])


def test_training_tool_passes_explicit_jobs_table(monkeypatch: pytest.MonkeyPatch) -> None:
    from amlas.backends.adapters._util import raw_tool
    from amlas.tools import training_jobs

    seen: list[dict[str, Any]] = []

    def submit(**kwargs: Any) -> Any:
        from amlas.tools.sagemaker import SubmitResult

        seen.append(kwargs)
        return SubmitResult("example", "session", "RUNNING")

    monkeypatch.setattr(training_jobs, "submit_training_job", submit)
    raw_tool(training_jobs.submit_sagemaker_job)(
        training_type="sklearn",
        config={"execution_role_arn": "role", "session_bucket": "bucket", "jobs_table": "explicit-table"},
    )
    assert seen[0]["jobs_table"] == "explicit-table"


def test_dispatch_preflight_uses_selected_jobs_table(monkeypatch: pytest.MonkeyPatch) -> None:
    from amlas.backends.adapters._util import raw_tool
    from amlas.tools import dispatch_preflight

    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "trial-blue")
    seen: list[str] = []
    for name in ("check_session_bucket", "check_pass_role", "check_staged_test_alignment"):
        monkeypatch.setattr(dispatch_preflight, name, lambda *args, **kwargs: {"status": "ok"})

    def check(**kwargs: Any) -> dict[str, str]:
        seen.append(kwargs["jobs_table"])
        return {"status": "ok"}

    monkeypatch.setattr(dispatch_preflight, "check_kms_jobs_table", check)
    raw_tool(dispatch_preflight.dispatch_preflight)()
    assert seen == ["trial-blue-training-jobs"]


def test_namespace_cli_runs_without_project_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "trial-blue")
    script = Path(__file__).resolve().parents[1] / "src/amlas/config/deployment.py"
    assert script.is_file(), "missing stdlib namespace resolver"
    result = subprocess.run(
        [sys.executable, "-I", str(script)],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"deployment_prefix": "trial-blue", "ssm_prefix": "/trial-blue"}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model_package_group", "amlas-models"),
        ("feature_store_db", "amlas_features"),
        ("tavily_secret_id", "amlas/tavily-api-key"),
    ],
)
def test_default_resources_remain_legacy(name: str, expected: str) -> None:
    assert externalized.resolve(name, ssm_getter=lambda *_: "") == expected


def test_default_training_backend_reads_managed_runtime_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    from amlas.backends.adapters.training import SageMakerTrainingBackend
    from amlas.tools import sagemaker

    monkeypatch.setenv("AMLAS_DEPLOYMENT_PREFIX", "trial-blue")
    monkeypatch.setenv("SAGEMAKER_EXECUTION_ROLE_ARN", "arn:aws:iam::000000000000:role/runtime-job")
    monkeypatch.setenv("AMLAS_SESSION_BUCKET", "trial-blue-session")
    seen: list[dict[str, Any]] = []

    def submit(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return sagemaker.SubmitResult("example", "session", "RUNNING")

    monkeypatch.setattr(sagemaker, "submit_training_job", submit)
    SageMakerTrainingBackend().submit(algo="xgboost", train_uri="s3://example-bucket/train/")
    assert seen[0]["execution_role_arn"] == "arn:aws:iam::000000000000:role/runtime-job"
    assert seen[0]["session_bucket"] == "trial-blue-session"
    assert seen[0]["jobs_table"] == "trial-blue-training-jobs"
